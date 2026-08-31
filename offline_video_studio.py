import os
import sys
import json
import math
import shutil
import pathlib
import tempfile
import datetime as dt
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional, List

import numpy as np
from PIL import Image

import torch
from diffusers import AutoPipelineForText2Image

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QTextEdit,
    QPushButton, QComboBox, QProgressBar, QGroupBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QSpinBox, QDoubleSpinBox
)

# -----------------------------
# Helpers
# -----------------------------
def now_slug() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def short(text: str, n: int = 80) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else (t[:n - 1] + "…")


def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def open_path(path: str):
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        os.system(f'open "{path}"')
    else:
        os.system(f'xdg-open "{path}"')


def pick_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# -----------------------------
# Prompt templates
# -----------------------------
TEMPLATES = {
    "Cinematic Documentary (Realistic)": (
        "DOCUMENTARY-STYLE CINEMATIC SHOT.\n"
        "Subject: {subject}\n"
        "Setting: {setting}\n"
        "Lighting: {lighting}\n"
        "Camera: {camera}\n"
        "Action: {action}\n"
        "Mood: {mood}\n"
        "Quality: realistic faces, high detail, natural motion\n"
        "Avoid: text, watermark, logo, glitches\n"
    ),
    "YouTube Intro (Bold, Clean)": (
        "A clean, high-production YouTube intro shot.\n"
        "Brand vibe: {brand_vibe}\n"
        "Scene: {setting}\n"
        "Camera: {camera}\n"
        "Action: {action}\n"
        "Lighting: {lighting}\n"
        "Mood: {mood}\n"
        "No on-screen text, no watermark.\n"
    ),
    "Action Scene (Grounded)": (
        "Grounded action scene with realistic physics.\n"
        "Location: {setting}\n"
        "Characters/subject: {subject}\n"
        "Action beats: {action}\n"
        "Camera: {camera}\n"
        "Lighting: {lighting}\n"
        "Mood: {mood}\n"
        "Avoid: gore, text, watermark, distortion\n"
    ),
    "Product Ad (Studio)": (
        "High-end product commercial.\n"
        "Product: {subject}\n"
        "Studio setup: {setting}\n"
        "Lighting: softbox highlights, glossy reflections\n"
        "Camera: smooth slider move, macro details\n"
        "Action: slow rotation / reveal, premium feel\n"
        "Avoid: text, watermark, distortion\n"
    ),
}


# -----------------------------
# History
# -----------------------------
@dataclass
class HistoryItem:
    created_at: str
    mode: str
    status: str
    model_id: str
    width: int
    height: int
    fps: int
    seconds: int
    steps: int
    guidance: float
    seed: int
    prompt_preview: str
    output_path: str
    stitched_path: str = ""
    error: str = ""


class HistoryStore:
    def __init__(self, out_dir: str):
        self.out_dir = pathlib.Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "history.json"
        self.items: List[HistoryItem] = []
        self.load()

    def load(self):
        if not self.path.exists():
            self.items = []
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.items = [HistoryItem(**x) for x in data]
        except Exception:
            self.items = []

    def save(self):
        data = [asdict(x) for x in self.items]
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add(self, item: HistoryItem):
        self.items.insert(0, item)
        self.save()


# -----------------------------
# Local pipeline: Text -> Image -> Video (Ken Burns)
# -----------------------------
@dataclass
class LocalGenConfig:
    model_id: str = "stabilityai/sd-turbo"

    # ✅ SAFE CPU TEST DEFAULTS (still changeable in GUI)
    width: int = 768
    height: int = 512
    fps: int = 24
    seconds: int = 4
    steps: int = 4
    guidance: float = 0.0
    seed: int = 42

    negative_prompt: str = "text, watermark, logo, glitch, deformed, blurry, extra fingers"

    # motion (safe)
    zoom_start: float = 1.00
    zoom_end: float = 1.08
    drift_px: int = 10
    rotate_deg: float = 0.0

    # encoding
    crf: int = 18
    preset: str = "medium"


def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _generate_keyframe(prompt: str, cfg: LocalGenConfig, device: torch.device, log_fn=None) -> Image.Image:
    _set_seed(cfg.seed)

    if log_fn:
        log_fn(f"Loading model: {cfg.model_id} (first run may download weights)")

    pipe = AutoPipelineForText2Image.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        use_safetensors=True
    ).to(device)

    if device.type == "cpu":
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    if log_fn:
        log_fn(f"Generating keyframe: {cfg.width}x{cfg.height}, steps={cfg.steps}, guidance={cfg.guidance}, seed={cfg.seed}")

    img = pipe(
        prompt=prompt,
        negative_prompt=cfg.negative_prompt if cfg.negative_prompt.strip() else None,
        num_inference_steps=int(cfg.steps),
        guidance_scale=float(cfg.guidance),
        width=int(cfg.width),
        height=int(cfg.height),
    ).images[0]

    return img


def _crop_zoom_pan(img: Image.Image, out_w: int, out_h: int, scale: float, dx: float, dy: float, rot_deg: float) -> Image.Image:
    img = img.convert("RGB")

    if abs(rot_deg) > 1e-6:
        img = img.rotate(rot_deg, resample=Image.Resampling.BICUBIC, expand=True)

    w, h = img.size
    zw, zh = int(w * scale), int(h * scale)
    zimg = img.resize((zw, zh), resample=Image.Resampling.LANCZOS)

    cx = zw // 2 + int(dx)
    cy = zh // 2 + int(dy)

    left = cx - out_w // 2
    top = cy - out_h // 2
    right = left + out_w
    bottom = top + out_h

    # clamp
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > zw:
        shift = right - zw
        left -= shift
        right = zw
    if bottom > zh:
        shift = bottom - zh
        top -= shift
        bottom = zh

    crop = zimg.crop((left, top, right, bottom))
    if crop.size != (out_w, out_h):
        crop = crop.resize((out_w, out_h), resample=Image.Resampling.LANCZOS)
    return crop


def _encode_frames_to_mp4(frames_dir: str, fps: int, out_mp4: str, cfg: LocalGenConfig):
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg first.")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(cfg.crf),
        "-preset", cfg.preset,
        out_mp4
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + proc.stderr[-4000:])


def _render_video_from_keyframe(keyframe: Image.Image, cfg: LocalGenConfig, out_mp4: str, log_fn=None, progress_fn=None, cancel_fn=None):
    n_frames = int(cfg.fps) * int(cfg.seconds)

    base = keyframe
    if base.size != (cfg.width, cfg.height):
        base = base.resize((cfg.width, cfg.height), resample=Image.Resampling.LANCZOS)

    tmpdir = tempfile.mkdtemp(prefix="offline_t2v_")
    try:
        if log_fn:
            log_fn(f"Rendering {n_frames} frames (Ken Burns motion)…")

        for i in range(n_frames):
            if cancel_fn and cancel_fn():
                raise RuntimeError("Cancelled")

            t = i / max(1, (n_frames - 1))

            scale = cfg.zoom_start + (cfg.zoom_end - cfg.zoom_start) * (0.5 - 0.5 * math.cos(math.pi * t))
            dx = cfg.drift_px * math.sin(2 * math.pi * t * 0.6)
            dy = cfg.drift_px * math.cos(2 * math.pi * t * 0.5)
            rot = cfg.rotate_deg * math.sin(2 * math.pi * t)

            frame = _crop_zoom_pan(base, cfg.width, cfg.height, scale, dx, dy, rot)
            frame.save(os.path.join(tmpdir, f"frame_{i:05d}.png"), "PNG")

            if progress_fn:
                progress_fn(int((i + 1) / n_frames * 90))

        if log_fn:
            log_fn("Encoding MP4 with ffmpeg…")

        _encode_frames_to_mp4(tmpdir, cfg.fps, out_mp4, cfg)

        if progress_fn:
            progress_fn(100)

        if log_fn:
            log_fn(f"Saved: {out_mp4}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _stitch_clips(clip_paths: List[str], out_mp4: str, log_fn=None):
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg not found in PATH.")

    out_dir = str(pathlib.Path(out_mp4).parent)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    concat_txt = os.path.join(out_dir, f"concat_{now_slug()}.txt")
    lines = []
    for p in clip_paths:
        safe = p.replace("'", r"'\''")
        lines.append(f"file '{safe}'")
    pathlib.Path(concat_txt).write_text("\n".join(lines), encoding="utf-8")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_txt,
        "-c", "copy",
        out_mp4
    ]
    if log_fn:
        log_fn("Stitching clips (ffmpeg concat)…")
        log_fn(" ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg stitch failed:\n" + proc.stderr[-4000:])


# -----------------------------
# Workers
# -----------------------------
class SingleLocalWorker(QThread):
    log = pyqtSignal(str)
    status = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, prompt: str, out_dir: str, out_name: str, cfg: LocalGenConfig):
        super().__init__()
        self.prompt = prompt.strip()
        self.out_dir = out_dir
        self.out_name = out_name
        self.cfg = cfg
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            if not self.prompt:
                raise ValueError("Prompt is empty.")

            out_dir = pathlib.Path(self.out_dir).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            stamp = now_slug()
            out_path = str(out_dir / f"{self.out_name}_{stamp}.mp4")
            keyframe_path = str(out_dir / f"{self.out_name}_{stamp}_keyframe.png")

            device = pick_device()
            self.log.emit(f"Device: {device} (CPU-only will be slower for keyframe generation)")

            self.status.emit("Generating keyframe image…")
            self.progress.emit(0)

            img = _generate_keyframe(
                prompt=self.prompt,
                cfg=self.cfg,
                device=device,
                log_fn=lambda s: self.log.emit(s)
            )
            img.save(keyframe_path)
            self.log.emit(f"Saved keyframe: {keyframe_path}")

            self.status.emit("Rendering video…")
            _render_video_from_keyframe(
                keyframe=img,
                cfg=self.cfg,
                out_mp4=out_path,
                log_fn=lambda s: self.log.emit(s),
                progress_fn=lambda p: self.progress.emit(p),
                cancel_fn=lambda: self._cancel
            )

            self.status.emit("Done ✅")
            self.done.emit(out_path)

        except Exception as e:
            msg = str(e)
            self.failed.emit("Cancelled" if "Cancelled" in msg else msg)


class BatchLocalWorker(QThread):
    log = pyqtSignal(str)
    status = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal(str, list)  # stitched_path, clip_paths
    failed = pyqtSignal(str)

    def __init__(self, prompts: List[str], out_dir: str, out_name: str, cfg: LocalGenConfig, do_stitch: bool):
        super().__init__()
        self.prompts = [p.strip() for p in prompts if p.strip()]
        self.out_dir = out_dir
        self.out_name = out_name
        self.cfg = cfg
        self.do_stitch = do_stitch
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            if not self.prompts:
                raise ValueError("No scenes to render.")

            out_dir = pathlib.Path(self.out_dir).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            if self.do_stitch and not ffmpeg_exists():
                raise RuntimeError("ffmpeg not found. Install ffmpeg or disable Auto-stitch.")

            device = pick_device()
            self.log.emit(f"Device: {device} (CPU-only will be slower for keyframes)")

            total = len(self.prompts)
            clip_paths: List[str] = []

            for i, prompt in enumerate(self.prompts, start=1):
                if self._cancel:
                    raise RuntimeError("Cancelled")

                self.status.emit(f"Scene {i}/{total}: generating keyframe…")
                self.log.emit(f"\n--- Scene {i}/{total} ---")
                self.log.emit(short(prompt, 200))

                scene_cfg = LocalGenConfig(**asdict(self.cfg))
                scene_cfg.seed = int(self.cfg.seed) + i

                img = _generate_keyframe(
                    prompt=prompt,
                    cfg=scene_cfg,
                    device=device,
                    log_fn=lambda s: self.log.emit(s)
                )

                clip_path = str(out_dir / f"{self.out_name}_{i:02d}_{now_slug()}.mp4")
                self.status.emit(f"Scene {i}/{total}: rendering video…")

                base = int(((i - 1) / total) * 100)
                span = int((1 / total) * 100)

                def overall_progress(local_p):
                    mapped = min(span, int((local_p / 100) * span))
                    self.progress.emit(min(99, base + mapped))

                _render_video_from_keyframe(
                    keyframe=img,
                    cfg=scene_cfg,
                    out_mp4=clip_path,
                    log_fn=lambda s: self.log.emit(s),
                    progress_fn=overall_progress,
                    cancel_fn=lambda: self._cancel
                )

                clip_paths.append(clip_path)

            stitched_path = ""
            if self.do_stitch and clip_paths:
                self.status.emit("Stitching scenes…")
                stitched_path = str(out_dir / f"{self.out_name}_STITCHED_{now_slug()}.mp4")
                _stitch_clips(clip_paths, stitched_path, log_fn=lambda s: self.log.emit(s))

            self.progress.emit(100)
            self.status.emit("Batch done ✅")
            self.done.emit(stitched_path, clip_paths)

        except Exception as e:
            msg = str(e)
            self.failed.emit("Cancelled" if "Cancelled" in msg else msg)


# -----------------------------
# GUI
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Offline Video Studio — Text → Image → Video (PyQt)")
        self.setMinimumSize(1280, 760)

        self.single_worker: Optional[SingleLocalWorker] = None
        self.batch_worker: Optional[BatchLocalWorker] = None

        self.last_output_path: Optional[str] = None
        self.last_stitched_path: Optional[str] = None

        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(14, 14, 14, 14)
        main.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(splitter, 1)

        left = QWidget()
        right = QWidget()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

        left_layout = QVBoxLayout(left)
        right_layout = QVBoxLayout(right)
        left_layout.setSpacing(12)
        right_layout.setSpacing(12)

        # ✅ Label helper to keep alignment consistent (stops overlap)
        def mk_label(text: str) -> QLabel:
            lab = QLabel(text)
            lab.setMinimumWidth(95)
            lab.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            return lab

        # ---------------- Settings (FIXED LAYOUT)
        settings_box = QGroupBox("Local Settings")
        g = QGridLayout(settings_box)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10)
        g.setContentsMargins(12, 14, 12, 12)
        # ✅ Responsive grid columns (stops squashing)
        g.setColumnStretch(0, 0)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(2, 0)
        g.setColumnStretch(3, 1)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "stabilityai/sd-turbo",
            "stabilityai/sdxl-turbo",
        ])
        self.model_combo.setMinimumWidth(280)

        # ✅ SAFE CPU TEST DEFAULTS (changeable)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(256, 2048)
        self.width_spin.setValue(768)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(256, 2048)
        self.height_spin.setValue(512)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(8, 60)
        self.fps_spin.setValue(24)

        self.seconds_spin = QSpinBox()
        self.seconds_spin.setRange(1, 30)
        self.seconds_spin.setValue(4)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 30)
        self.steps_spin.setValue(4)

        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(0.0, 20.0)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setValue(0.0)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(42)

        self.negative_input = QLineEdit("text, watermark, logo, glitch, deformed, blurry, extra fingers")
        self.negative_input.setMinimumWidth(320)

        self.zoom_end_spin = QDoubleSpinBox()
        self.zoom_end_spin.setRange(1.0, 1.5)
        self.zoom_end_spin.setSingleStep(0.01)
        self.zoom_end_spin.setValue(1.08)

        self.drift_spin = QSpinBox()
        self.drift_spin.setRange(0, 120)
        self.drift_spin.setValue(10)

        self.rotate_spin = QDoubleSpinBox()
        self.rotate_spin.setRange(0.0, 2.0)
        self.rotate_spin.setSingleStep(0.1)
        self.rotate_spin.setValue(0.0)

        self.out_dir_input = QLineEdit(str(pathlib.Path.home() / "offline_video_outputs"))
        self.out_dir_input.setMinimumWidth(320)

        self.out_name_input = QLineEdit("scene")
        self.browse_out_btn = QPushButton("Browse…")

        self.stitch_checkbox = QCheckBox("Auto-stitch batch scenes (ffmpeg)")
        self.stitch_checkbox.setChecked(True)

        row = 0
        g.addWidget(mk_label("Model"), row, 0)
        g.addWidget(self.model_combo, row, 1, 1, 3)

        row += 1
        g.addWidget(mk_label("Width"), row, 0)
        g.addWidget(self.width_spin, row, 1)
        g.addWidget(mk_label("Height"), row, 2)
        g.addWidget(self.height_spin, row, 3)

        row += 1
        g.addWidget(mk_label("FPS"), row, 0)
        g.addWidget(self.fps_spin, row, 1)
        g.addWidget(mk_label("Seconds"), row, 2)
        g.addWidget(self.seconds_spin, row, 3)

        row += 1
        g.addWidget(mk_label("Steps"), row, 0)
        g.addWidget(self.steps_spin, row, 1)
        g.addWidget(mk_label("Guidance"), row, 2)
        g.addWidget(self.guidance_spin, row, 3)

        row += 1
        g.addWidget(mk_label("Seed"), row, 0)
        g.addWidget(self.seed_spin, row, 1, 1, 3)

        # ✅ BIG overlap fix: negative prompt gets its own row spanning columns
        row += 1
        g.addWidget(mk_label("Negative"), row, 0)
        g.addWidget(self.negative_input, row, 1, 1, 3)

        row += 1
        g.addWidget(mk_label("Zoom end"), row, 0)
        g.addWidget(self.zoom_end_spin, row, 1)
        g.addWidget(mk_label("Drift (px)"), row, 2)
        g.addWidget(self.drift_spin, row, 3)

        row += 1
        g.addWidget(mk_label("Rotate (deg)"), row, 0)
        g.addWidget(self.rotate_spin, row, 1, 1, 3)

        row += 1
        g.addWidget(mk_label("Output"), row, 0)
        g.addWidget(self.out_dir_input, row, 1, 1, 2)
        g.addWidget(self.browse_out_btn, row, 3)

        row += 1
        g.addWidget(mk_label("Name base"), row, 0)
        g.addWidget(self.out_name_input, row, 1, 1, 3)

        row += 1
        g.addWidget(self.stitch_checkbox, row, 0, 1, 4)

        left_layout.addWidget(settings_box)

        # ---------------- Prompt + templates
        prompt_box = QGroupBox("Single Prompt (one clip)")
        pv = QVBoxLayout(prompt_box)

        template_row = QHBoxLayout()
        self.template_combo = QComboBox()
        self.template_combo.addItems(["(No template)"] + list(TEMPLATES.keys()))
        self.apply_template_btn = QPushButton("Apply Template")
        template_row.addWidget(QLabel("Template:"))
        template_row.addWidget(self.template_combo, 1)
        template_row.addWidget(self.apply_template_btn)

        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(
            "Write your prompt here.\n\n"
            "Example:\n"
            "Cinematic wide shot of a snowy Arctic research station at night, blue lighting, soft fog, documentary style, "
            "realistic, no text, no watermark."
        )
        self.prompt_input.setMinimumHeight(170)

        pv.addLayout(template_row)
        pv.addWidget(self.prompt_input)

        btn_row = QHBoxLayout()
        self.generate_single_btn = QPushButton("Generate Clip (Local)")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        btn_row.addWidget(self.generate_single_btn)
        btn_row.addWidget(self.cancel_btn)
        pv.addLayout(btn_row)

        left_layout.addWidget(prompt_box)

        # ---------------- Batch scenes
        batch_box = QGroupBox("Batch Scenes (one prompt per line)")
        bv = QVBoxLayout(batch_box)
        self.scenes_input = QTextEdit()
        self.scenes_input.setPlaceholderText(
            "One scene prompt per line.\n"
            "Example:\n"
            "Arctic lab exterior, drone shot, snow blowing, cinematic documentary.\n"
            "Inside lab corridor tracking shot, cold blue lighting, subtle haze.\n"
            "Close-up of scientist holding vial, shallow depth of field.\n"
        )
        self.scenes_input.setMinimumHeight(150)
        self.generate_batch_btn = QPushButton("Generate Batch (+ Stitch if enabled)")
        bv.addWidget(self.scenes_input)
        bv.addWidget(self.generate_batch_btn)

        left_layout.addWidget(batch_box)
        left_layout.addStretch(1)

        # ---------------- Right: status + logs + history
        status_box = QGroupBox("Status")
        sv = QVBoxLayout(status_box)
        self.status_label = QLabel("Idle.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        action_row = QHBoxLayout()
        self.open_out_folder_btn = QPushButton("Open Output Folder")
        self.open_last_btn = QPushButton("Open Last Clip")
        self.open_last_btn.setEnabled(False)
        self.open_stitched_btn = QPushButton("Open Stitched Video")
        self.open_stitched_btn.setEnabled(False)
        action_row.addWidget(self.open_out_folder_btn)
        action_row.addWidget(self.open_last_btn)
        action_row.addWidget(self.open_stitched_btn)

        sv.addWidget(self.status_label)
        sv.addWidget(self.progress_bar)
        sv.addLayout(action_row)
        right_layout.addWidget(status_box)

        logs_box = QGroupBox("Logs")
        lv = QVBoxLayout(logs_box)
        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMinimumHeight(210)
        lv.addWidget(self.logs)
        right_layout.addWidget(logs_box)

        history_box = QGroupBox("History")
        hv = QVBoxLayout(history_box)
        self.history_table = QTableWidget(0, 10)
        self.history_table.setHorizontalHeaderLabels([
            "Time", "Mode", "Status", "Model", "WxH", "FPS", "Sec", "Seed", "Prompt", "Output"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        history_btn_row = QHBoxLayout()
        self.refresh_history_btn = QPushButton("Refresh")
        self.open_selected_btn = QPushButton("Open Selected")
        history_btn_row.addWidget(self.refresh_history_btn)
        history_btn_row.addWidget(self.open_selected_btn)

        hv.addWidget(self.history_table)
        hv.addLayout(history_btn_row)
        right_layout.addWidget(history_box, 1)

        # Wiring
        self.apply_template_btn.clicked.connect(self.apply_template)
        self.browse_out_btn.clicked.connect(self.pick_out_dir)
        self.generate_single_btn.clicked.connect(self.start_single)
        self.generate_batch_btn.clicked.connect(self.start_batch)
        self.cancel_btn.clicked.connect(self.cancel_any)
        self.open_out_folder_btn.clicked.connect(self.open_output_folder)
        self.open_last_btn.clicked.connect(self.open_last_clip)
        self.open_stitched_btn.clicked.connect(self.open_last_stitched)
        self.refresh_history_btn.clicked.connect(self.load_history_table)
        self.open_selected_btn.clicked.connect(self.open_selected)

        self.apply_theme()
        self.load_history_table()

        if self.stitch_checkbox.isChecked() and not ffmpeg_exists():
            self.log("⚠️ ffmpeg not found in PATH. Install ffmpeg or disable Auto-stitch.")

        if pick_device().type == "cpu":
            self.log("ℹ️ CPU-only mode: defaults set to 768×512, 4s, 4 steps for safe testing.")

    def apply_theme(self):
        self.setStyleSheet("""
            QWidget { background: #0b1220; color: #e8eefc; }
            QGroupBox {
                border: 1px solid #253046; border-radius: 14px;
                margin-top: 10px; padding: 12px;
                background: rgba(255,255,255,0.03);
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 6px;
                color: #bcd1ff; font-weight: 800;
            }
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #0f1b30; border: 1px solid #253046;
                border-radius: 10px; padding: 8px;
                selection-background-color: #2f5fff;
            }
            QPushButton {
                background: #2f5fff; border: none; border-radius: 12px;
                padding: 10px 12px; font-weight: 800;
            }
            QPushButton:hover { background: #3a67ff; }
            QPushButton:disabled { background: #2a3550; color: #9aa6c1; }
            QProgressBar {
                border: 1px solid #253046; border-radius: 10px;
                text-align: center; background: #0f1b30; height: 18px;
            }
            QProgressBar::chunk { background: #2f5fff; border-radius: 10px; }
            QLabel { color: #e8eefc; }
            QTableWidget {
                background: #0f1b30;
                border: 1px solid #253046;
                border-radius: 10px;
                gridline-color: #253046;
            }
            QHeaderView::section {
                background: #0b1220;
                color: #bcd1ff;
                border: none;
                padding: 6px;
                font-weight: 800;
            }
        """)
        self.setFont(QFont("Segoe UI", 10))

    def log(self, text: str):
        self.logs.append(text)

    def set_status(self, text: str):
        self.status_label.setText(text)

    def pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder", self.out_dir_input.text())
        if d:
            self.out_dir_input.setText(d)
            self.load_history_table()

    def apply_template(self):
        name = self.template_combo.currentText()
        if name == "(No template)":
            return
        self.prompt_input.setPlainText(TEMPLATES.get(name, ""))
        self.log(f"Template applied: {name}")

    def get_cfg(self) -> LocalGenConfig:
        return LocalGenConfig(
            model_id=self.model_combo.currentText(),
            width=int(self.width_spin.value()),
            height=int(self.height_spin.value()),
            fps=int(self.fps_spin.value()),
            seconds=int(self.seconds_spin.value()),
            steps=int(self.steps_spin.value()),
            guidance=float(self.guidance_spin.value()),
            seed=int(self.seed_spin.value()),
            negative_prompt=self.negative_input.text().strip(),
            zoom_start=1.00,
            zoom_end=float(self.zoom_end_spin.value()),
            drift_px=int(self.drift_spin.value()),
            rotate_deg=float(self.rotate_spin.value()),
        )

    def history_store(self) -> HistoryStore:
        return HistoryStore(self.out_dir_input.text().strip())

    def set_busy(self, busy: bool):
        self.generate_single_btn.setEnabled(not busy)
        self.generate_batch_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.open_last_btn.setEnabled((not busy) and bool(self.last_output_path))
        self.open_stitched_btn.setEnabled((not busy) and bool(self.last_stitched_path))

    def start_single(self):
        if (self.single_worker and self.single_worker.isRunning()) or (self.batch_worker and self.batch_worker.isRunning()):
            QMessageBox.warning(self, "Busy", "A generation is already running.")
            return

        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Prompt missing", "Please enter a prompt.")
            return

        cfg = self.get_cfg()
        out_dir = self.out_dir_input.text().strip()
        out_name = self.out_name_input.text().strip() or "scene"

        self.logs.clear()
        self.progress_bar.setValue(0)
        self.set_status("Starting…")
        self.set_busy(True)
        self.last_output_path = None
        self.last_stitched_path = None

        self.single_worker = SingleLocalWorker(prompt=prompt, out_dir=out_dir, out_name=out_name, cfg=cfg)
        self.single_worker.log.connect(self.log)
        self.single_worker.status.connect(self.set_status)
        self.single_worker.progress.connect(self.progress_bar.setValue)
        self.single_worker.done.connect(self.on_single_done)
        self.single_worker.failed.connect(self.on_single_failed)
        self.single_worker.start()

    def start_batch(self):
        if (self.single_worker and self.single_worker.isRunning()) or (self.batch_worker and self.batch_worker.isRunning()):
            QMessageBox.warning(self, "Busy", "A generation is already running.")
            return

        raw = self.scenes_input.toPlainText()
        prompts = [line.strip() for line in raw.splitlines() if line.strip()]
        if not prompts:
            QMessageBox.warning(self, "No scenes", "Add at least one scene prompt (one per line).")
            return

        do_stitch = self.stitch_checkbox.isChecked()
        if do_stitch and not ffmpeg_exists():
            QMessageBox.critical(self, "ffmpeg missing", "Auto-stitch is enabled, but ffmpeg is not found in PATH.")
            return

        cfg = self.get_cfg()
        out_dir = self.out_dir_input.text().strip()
        out_name = self.out_name_input.text().strip() or "scene"

        self.logs.clear()
        self.progress_bar.setValue(0)
        self.set_status("Starting batch…")
        self.set_busy(True)
        self.last_output_path = None
        self.last_stitched_path = None

        self.batch_worker = BatchLocalWorker(prompts=prompts, out_dir=out_dir, out_name=out_name, cfg=cfg, do_stitch=do_stitch)
        self.batch_worker.log.connect(self.log)
        self.batch_worker.status.connect(self.set_status)
        self.batch_worker.progress.connect(self.progress_bar.setValue)
        self.batch_worker.done.connect(self.on_batch_done)
        self.batch_worker.failed.connect(self.on_batch_failed)
        self.batch_worker.start()

    def cancel_any(self):
        if self.single_worker and self.single_worker.isRunning():
            self.single_worker.cancel()
            self.log("Cancel requested (single)…")
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.cancel()
            self.log("Cancel requested (batch)…")
        self.set_status("Cancelling…")

    def on_single_done(self, out_path: str):
        self.last_output_path = out_path
        self.progress_bar.setValue(100)
        self.set_status("Done ✅")
        self.set_busy(False)

        cfg = self.get_cfg()
        store = self.history_store()
        store.add(HistoryItem(
            created_at=dt.datetime.now().isoformat(timespec="seconds"),
            mode="single",
            status="completed",
            model_id=cfg.model_id,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            seconds=cfg.seconds,
            steps=cfg.steps,
            guidance=cfg.guidance,
            seed=cfg.seed,
            prompt_preview=short(self.prompt_input.toPlainText(), 160),
            output_path=out_path
        ))
        self.load_history_table()

    def on_single_failed(self, msg: str):
        self.progress_bar.setValue(0)
        self.set_status("Failed ❌")
        self.set_busy(False)
        self.log(f"❌ Error: {msg}")

        cfg = self.get_cfg()
        store = self.history_store()
        store.add(HistoryItem(
            created_at=dt.datetime.now().isoformat(timespec="seconds"),
            mode="single",
            status=("cancelled" if msg == "Cancelled" else "failed"),
            model_id=cfg.model_id,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            seconds=cfg.seconds,
            steps=cfg.steps,
            guidance=cfg.guidance,
            seed=cfg.seed,
            prompt_preview=short(self.prompt_input.toPlainText(), 160),
            output_path="",
            error=msg
        ))
        self.load_history_table()
        QMessageBox.critical(self, "Generation failed", msg)

    def on_batch_done(self, stitched_path: str, clip_paths: list):
        self.progress_bar.setValue(100)
        self.set_status("Batch done ✅")
        self.set_busy(False)

        self.last_output_path = clip_paths[-1] if clip_paths else None
        self.last_stitched_path = stitched_path or None
        self.open_last_btn.setEnabled(bool(self.last_output_path))
        self.open_stitched_btn.setEnabled(bool(self.last_stitched_path))

        cfg = self.get_cfg()
        store = self.history_store()
        store.add(HistoryItem(
            created_at=dt.datetime.now().isoformat(timespec="seconds"),
            mode="batch",
            status="completed",
            model_id=cfg.model_id,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            seconds=cfg.seconds,
            steps=cfg.steps,
            guidance=cfg.guidance,
            seed=cfg.seed,
            prompt_preview=short(self.scenes_input.toPlainText(), 160),
            output_path="; ".join(clip_paths[:3]) + ("…" if len(clip_paths) > 3 else ""),
            stitched_path=stitched_path or ""
        ))
        self.load_history_table()

    def on_batch_failed(self, msg: str):
        self.progress_bar.setValue(0)
        self.set_status("Failed ❌")
        self.set_busy(False)
        self.log(f"❌ Error: {msg}")

        cfg = self.get_cfg()
        store = self.history_store()
        store.add(HistoryItem(
            created_at=dt.datetime.now().isoformat(timespec="seconds"),
            mode="batch",
            status=("cancelled" if msg == "Cancelled" else "failed"),
            model_id=cfg.model_id,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            seconds=cfg.seconds,
            steps=cfg.steps,
            guidance=cfg.guidance,
            seed=cfg.seed,
            prompt_preview=short(self.scenes_input.toPlainText(), 160),
            output_path="",
            error=msg
        ))
        self.load_history_table()
        QMessageBox.critical(self, "Batch failed", msg)

    def open_output_folder(self):
        folder = self.out_dir_input.text().strip()
        if folder:
            open_path(folder)

    def open_last_clip(self):
        if self.last_output_path:
            open_path(self.last_output_path)

    def open_last_stitched(self):
        if self.last_stitched_path:
            open_path(self.last_stitched_path)

    def load_history_table(self):
        store = self.history_store()
        store.load()
        items = store.items

        self.history_table.setRowCount(len(items))
        for r, it in enumerate(items):
            vals = [
                it.created_at,
                it.mode,
                it.status,
                it.model_id,
                f"{it.width}x{it.height}",
                str(it.fps),
                str(it.seconds),
                str(it.seed),
                it.prompt_preview,
                it.stitched_path if it.stitched_path else it.output_path
            ]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v or "")
                if c in (0, 1, 2, 5, 6, 7):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.history_table.setItem(r, c, item)

    def selected_history_item(self) -> Optional[HistoryItem]:
        row = self.history_table.currentRow()
        if row < 0:
            return None
        store = self.history_store()
        store.load()
        if row >= len(store.items):
            return None
        return store.items[row]

    def open_selected(self):
        it = self.selected_history_item()
        if not it:
            QMessageBox.information(self, "No selection", "Select a row in History first.")
            return

        if it.stitched_path and pathlib.Path(it.stitched_path).exists():
            open_path(it.stitched_path)
            return

        if it.output_path and pathlib.Path(it.output_path).exists():
            open_path(it.output_path)
            return

        folder = self.out_dir_input.text().strip()
        if folder:
            open_path(folder)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()