import os
import sys
import json
import time
import shutil
import pathlib
import datetime as dt
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional, List

import requests

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QTextEdit,
    QPushButton, QComboBox, QProgressBar, QGroupBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter
)

# -----------------------------
# OpenAI Sora Video API (REST)
# -----------------------------
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Sora API allows seconds: "4","8","12" and sizes depending on model.
# Source: OpenAI docs (video generation guide + Sora prompting guide). :contentReference[oaicite:1]{index=1}


def now_slug() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def short(text: str, n: int = 60) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else (t[:n - 1] + "…")


def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def open_path(path: str):
    # Cross-platform open
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        os.system(f'open "{path}"')
    else:
        os.system(f'xdg-open "{path}"')


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
        "Avoid: text, watermark, logo, glitch, flicker\n"
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
    "Product Ad (Studio)": (
        "High-end product commercial.\n"
        "Product: {subject}\n"
        "Studio setup: {setting}\n"
        "Lighting: softbox highlights, glossy reflections\n"
        "Camera: smooth slider move, macro details\n"
        "Action: slow rotation / reveal, premium feel\n"
        "Avoid: text, watermark, distortion\n"
    ),
    "Action Scene (Grounded)": (
        "Grounded action scene with realistic physics.\n"
        "Location: {setting}\n"
        "Characters: {subject}\n"
        "Action beats: {action}\n"
        "Camera: {camera}\n"
        "Lighting: {lighting}\n"
        "Mood: {mood}\n"
        "Avoid: gore, text, watermark, glitch\n"
    ),
}

TEMPLATE_PLACEHOLDER_HINT = (
    "Tip: You can replace placeholders like {subject}, {setting}, {camera} with your own words."
)


# -----------------------------
# History model
# -----------------------------
@dataclass
class HistoryItem:
    created_at: str
    mode: str                 # single | batch
    status: str               # completed | failed | cancelled | in_progress
    video_id: str
    model: str
    size: str
    seconds: str
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
# Workers
# -----------------------------
class SingleSoraWorker(QThread):
    log = pyqtSignal(str)
    status = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal(str, str)      # output filepath, video_id
    failed = pyqtSignal(str)

    def __init__(
        self,
        api_key: str,
        prompt: str,
        model: str,
        size: str,
        seconds: str,
        out_dir: str,
        out_name: str,
        reference_image_path: Optional[str],
        verify_tls: bool,
        poll_interval_sec: int
    ):
        super().__init__()
        self.api_key = api_key.strip()
        self.prompt = prompt
        self.model = model
        self.size = size
        self.seconds = seconds
        self.out_dir = out_dir
        self.out_name = out_name
        self.reference_image_path = reference_image_path
        self.verify_tls = verify_tls
        self.poll_interval_sec = poll_interval_sec
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def run(self):
        try:
            if not self.api_key:
                raise ValueError("Missing API key. Set OPENAI_API_KEY or paste it into the app.")
            if not self.prompt.strip():
                raise ValueError("Prompt is empty.")

            out_dir = pathlib.Path(self.out_dir).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{self.out_name or 'clip'}_{now_slug()}.mp4"
            out_path = out_dir / filename

            self.progress.emit(0)
            self.status.emit("Creating render job…")
            self.log.emit("POST /v1/videos (multipart/form-data)")

            files = {
                "prompt": (None, self.prompt),
                "model": (None, self.model),
                "size": (None, self.size),
                "seconds": (None, str(self.seconds)),
            }

            # input_reference is supported by the video-generation guide. :contentReference[oaicite:2]{index=2}
            img_handle = None
            if self.reference_image_path:
                p = pathlib.Path(self.reference_image_path)
                if not p.exists():
                    raise ValueError(f"Reference image not found: {p}")
                img_handle = open(p, "rb")
                files["input_reference"] = (p.name, img_handle, "application/octet-stream")

            try:
                resp = requests.post(
                    f"{OPENAI_BASE_URL}/videos",
                    headers=self._headers(),
                    files=files,
                    timeout=120,
                    verify=self.verify_tls
                )
            finally:
                if img_handle:
                    img_handle.close()

            if resp.status_code >= 400:
                raise RuntimeError(f"Create failed ({resp.status_code}): {resp.text}")

            job = resp.json()
            video_id = job.get("id")
            if not video_id:
                raise RuntimeError(f"Unexpected response: {json.dumps(job, indent=2)}")

            self.log.emit(f"Job created: {video_id}")
            self.status.emit("Queued…")

            while True:
                if self._cancel:
                    self.status.emit("Cancelled.")
                    self.progress.emit(0)
                    return

                r = requests.get(
                    f"{OPENAI_BASE_URL}/videos/{video_id}",
                    headers=self._headers(),
                    timeout=60,
                    verify=self.verify_tls
                )
                if r.status_code >= 400:
                    raise RuntimeError(f"Retrieve failed ({r.status_code}): {r.text}")
                info = r.json()

                st = info.get("status", "unknown")
                prog = info.get("progress", 0)
                self.status.emit(f"{st}")
                try:
                    self.progress.emit(int(prog))
                except Exception:
                    self.progress.emit(0)

                self.log.emit(f"GET /v1/videos/{video_id} → status={st}, progress={prog}")

                if st == "completed":
                    break
                if st == "failed":
                    err = info.get("error", {})
                    raise RuntimeError(f"Generation failed: {err.get('message') or json.dumps(err)}")

                time.sleep(self.poll_interval_sec)

            self.status.emit("Downloading MP4…")
            self.log.emit(f"GET /v1/videos/{video_id}/content")

            with requests.get(
                f"{OPENAI_BASE_URL}/videos/{video_id}/content",
                headers=self._headers(),
                stream=True,
                timeout=300,
                verify=self.verify_tls
            ) as dl:
                if dl.status_code >= 400:
                    raise RuntimeError(f"Download failed ({dl.status_code}): {dl.text}")
                with open(out_path, "wb") as f:
                    for chunk in dl.iter_content(chunk_size=1024 * 1024):
                        if self._cancel:
                            self.status.emit("Cancelled during download.")
                            self.progress.emit(0)
                            return
                        if chunk:
                            f.write(chunk)

            self.progress.emit(100)
            self.status.emit("Done ✅")
            self.log.emit(f"Saved: {out_path}")
            self.done.emit(str(out_path), video_id)

        except Exception as e:
            self.failed.emit(str(e))


class BatchSoraWorker(QThread):
    log = pyqtSignal(str)
    status = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal(str, list)     # stitched_path, list_of_clip_paths
    failed = pyqtSignal(str)

    def __init__(
        self,
        api_key: str,
        prompts: List[str],
        model: str,
        size: str,
        seconds: str,
        out_dir: str,
        out_name: str,
        reference_image_path: Optional[str],
        verify_tls: bool,
        poll_interval_sec: int,
        do_stitch: bool
    ):
        super().__init__()
        self.api_key = api_key.strip()
        self.prompts = [p.strip() for p in prompts if p.strip()]
        self.model = model
        self.size = size
        self.seconds = seconds
        self.out_dir = out_dir
        self.out_name = out_name
        self.reference_image_path = reference_image_path
        self.verify_tls = verify_tls
        self.poll_interval_sec = poll_interval_sec
        self.do_stitch = do_stitch
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            if not self.api_key:
                raise ValueError("Missing API key.")
            if not self.prompts:
                raise ValueError("No scene prompts found. Add at least 1 scene.")

            out_dir = pathlib.Path(self.out_dir).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            if self.do_stitch and not ffmpeg_exists():
                raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg or disable Auto-stitch.")

            clip_paths: List[str] = []
            total = len(self.prompts)
            stitched_path = ""

            for i, prompt in enumerate(self.prompts, start=1):
                if self._cancel:
                    self.status.emit("Cancelled.")
                    self.progress.emit(0)
                    return

                self.status.emit(f"Scene {i}/{total}: creating job…")
                self.log.emit(f"--- Scene {i}/{total} ---")
                self.log.emit(short(prompt, 140))

                # Reuse single logic but inline for fewer moving parts
                files = {
                    "prompt": (None, prompt),
                    "model": (None, self.model),
                    "size": (None, self.size),
                    "seconds": (None, str(self.seconds)),
                }

                img_handle = None
                if self.reference_image_path:
                    p = pathlib.Path(self.reference_image_path)
                    img_handle = open(p, "rb")
                    files["input_reference"] = (p.name, img_handle, "application/octet-stream")

                try:
                    resp = requests.post(
                        f"{OPENAI_BASE_URL}/videos",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        files=files,
                        timeout=120,
                        verify=self.verify_tls
                    )
                finally:
                    if img_handle:
                        img_handle.close()

                if resp.status_code >= 400:
                    raise RuntimeError(f"Create failed ({resp.status_code}): {resp.text}")

                job = resp.json()
                video_id = job.get("id")
                if not video_id:
                    raise RuntimeError(f"Unexpected response: {json.dumps(job, indent=2)}")

                # Poll
                while True:
                    if self._cancel:
                        self.status.emit("Cancelled.")
                        self.progress.emit(0)
                        return

                    r = requests.get(
                        f"{OPENAI_BASE_URL}/videos/{video_id}",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=60,
                        verify=self.verify_tls
                    )
                    if r.status_code >= 400:
                        raise RuntimeError(f"Retrieve failed ({r.status_code}): {r.text}")
                    info = r.json()
                    st = info.get("status", "unknown")
                    prog = info.get("progress", 0)
                    self.status.emit(f"Scene {i}/{total}: {st} ({prog}%)")
                    self.log.emit(f"status={st}, progress={prog}")

                    # overall progress: distribute scenes evenly
                    base = int(((i - 1) / total) * 100)
                    span = int((1 / total) * 100)
                    try:
                        local = int(prog)
                    except Exception:
                        local = 0
                    self.progress.emit(min(99, base + int((local / 100) * span)))

                    if st == "completed":
                        break
                    if st == "failed":
                        err = info.get("error", {})
                        raise RuntimeError(f"Scene {i} failed: {err.get('message') or json.dumps(err)}")

                    time.sleep(self.poll_interval_sec)

                # Download
                clip_name = f"{self.out_name or 'scene'}_{i:02d}_{now_slug()}.mp4"
                clip_path = out_dir / clip_name
                self.status.emit(f"Scene {i}/{total}: downloading…")

                with requests.get(
                    f"{OPENAI_BASE_URL}/videos/{video_id}/content",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    stream=True,
                    timeout=300,
                    verify=self.verify_tls
                ) as dl:
                    if dl.status_code >= 400:
                        raise RuntimeError(f"Download failed ({dl.status_code}): {dl.text}")
                    with open(clip_path, "wb") as f:
                        for chunk in dl.iter_content(chunk_size=1024 * 1024):
                            if self._cancel:
                                self.status.emit("Cancelled.")
                                self.progress.emit(0)
                                return
                            if chunk:
                                f.write(chunk)

                clip_paths.append(str(clip_path))
                self.log.emit(f"Saved clip: {clip_path}")

            # Stitch
            if self.do_stitch and clip_paths:
                self.status.emit("Stitching clips (ffmpeg)…")

                concat_file = out_dir / f"concat_{now_slug()}.txt"
                lines = []
                for p in clip_paths:
                    # ffmpeg concat demuxer expects: file 'path'
                    safe = p.replace("'", r"'\''")
                    lines.append(f"file '{safe}'")
                concat_file.write_text("\n".join(lines), encoding="utf-8")

                stitched_name = f"{self.out_name or 'episode'}_STITCHED_{now_slug()}.mp4"
                stitched = out_dir / stitched_name

                cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(concat_file),
                    "-c", "copy",
                    str(stitched)
                ]

                self.log.emit("Running: " + " ".join(cmd))
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(
                        "ffmpeg stitch failed.\n"
                        f"STDERR:\n{proc.stderr[-4000:]}"
                    )

                stitched_path = str(stitched)
                self.log.emit(f"Stitched: {stitched_path}")

            self.progress.emit(100)
            self.status.emit("Batch done ✅")
            self.done.emit(stitched_path, clip_paths)

        except Exception as e:
            self.failed.emit(str(e))


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sora Studio — Text → HQ Video (PyQt)")
        self.setMinimumSize(1200, 720)

        self.single_worker: Optional[SingleSoraWorker] = None
        self.batch_worker: Optional[BatchSoraWorker] = None

        self.reference_image_path: Optional[str] = None
        self.last_output_path: Optional[str] = None
        self.last_stitched_path: Optional[str] = None

        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(14, 14, 14, 14)
        main.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(splitter, 1)

        # Left panel
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(12)
        splitter.addWidget(left)

        # Right panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(12)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

        # ---------------- Settings
        settings_box = QGroupBox("Settings")
        g = QGridLayout(settings_box)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(10)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Leave empty to use OPENAI_API_KEY (recommended)")
        if os.environ.get("OPENAI_API_KEY"):
            self.api_key_input.setText("")  # do not autofill secrets

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "sora-2",
            "sora-2-pro",
            "sora-2-2025-10-06",
            "sora-2-pro-2025-10-06",
            "sora-2-2025-12-08",
        ])

        self.size_combo = QComboBox()
        self.size_combo.addItems([
            "1280x720",
            "1792x1024",
            "720x1280",
            "1024x1792",
        ])

        self.seconds_combo = QComboBox()
        self.seconds_combo.addItems(["4", "8", "12"])

        self.poll_combo = QComboBox()
        self.poll_combo.addItems(["3", "5", "8", "10"])

        self.out_dir_input = QLineEdit(str(pathlib.Path.home() / "sora_outputs"))
        self.out_name_input = QLineEdit("scene")

        self.tls_checkbox = QCheckBox("Verify HTTPS (recommended)")
        self.tls_checkbox.setChecked(True)

        self.stitch_checkbox = QCheckBox("Auto-stitch batch scenes (requires ffmpeg)")
        self.stitch_checkbox.setChecked(True)

        self.browse_out_btn = QPushButton("Browse…")

        row = 0
        g.addWidget(QLabel("API Key"), row, 0)
        g.addWidget(self.api_key_input, row, 1, 1, 3)

        row += 1
        g.addWidget(QLabel("Model"), row, 0)
        g.addWidget(self.model_combo, row, 1)
        g.addWidget(QLabel("Size"), row, 2)
        g.addWidget(self.size_combo, row, 3)

        row += 1
        g.addWidget(QLabel("Seconds"), row, 0)
        g.addWidget(self.seconds_combo, row, 1)
        g.addWidget(QLabel("Poll (sec)"), row, 2)
        g.addWidget(self.poll_combo, row, 3)

        row += 1
        g.addWidget(QLabel("Output folder"), row, 0)
        g.addWidget(self.out_dir_input, row, 1, 1, 2)
        g.addWidget(self.browse_out_btn, row, 3)

        row += 1
        g.addWidget(QLabel("Filename base"), row, 0)
        g.addWidget(self.out_name_input, row, 1, 1, 3)

        row += 1
        self.ref_label = QLabel("Reference image: (none)")
        self.ref_btn = QPushButton("Pick image…")
        self.ref_clear_btn = QPushButton("Clear")
        g.addWidget(self.ref_label, row, 0, 1, 2)
        g.addWidget(self.ref_btn, row, 2)
        g.addWidget(self.ref_clear_btn, row, 3)

        row += 1
        g.addWidget(self.tls_checkbox, row, 0, 1, 2)
        g.addWidget(self.stitch_checkbox, row, 2, 1, 2)

        left_layout.addWidget(settings_box)

        # ---------------- Prompt + templates
        prompt_box = QGroupBox("Single Prompt (Generate one clip)")
        pv = QVBoxLayout(prompt_box)

        template_row = QHBoxLayout()
        self.template_combo = QComboBox()
        self.template_combo.addItems(["(No template)"] + list(TEMPLATES.keys()))
        self.apply_template_btn = QPushButton("Apply Template")
        self.template_hint = QLabel(TEMPLATE_PLACEHOLDER_HINT)
        self.template_hint.setWordWrap(True)

        template_row.addWidget(QLabel("Template:"))
        template_row.addWidget(self.template_combo, 1)
        template_row.addWidget(self.apply_template_btn)

        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(
            "Write your prompt here.\n\n"
            "Example:\n"
            "Wide documentary shot inside an Arctic underground lab, cold blue lighting, subtle haze, "
            "scientists in insulated coats, slow push-in, realistic faces, no text, no watermark."
        )
        self.prompt_input.setMinimumHeight(170)

        pv.addLayout(template_row)
        pv.addWidget(self.template_hint)
        pv.addWidget(self.prompt_input)

        btn_row = QHBoxLayout()
        self.generate_single_btn = QPushButton("Generate Clip")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        btn_row.addWidget(self.generate_single_btn)
        btn_row.addWidget(self.cancel_btn)
        pv.addLayout(btn_row)

        left_layout.addWidget(prompt_box)

        # ---------------- Batch scenes
        batch_box = QGroupBox("Batch Scenes (One prompt per line) — optional")
        bv = QVBoxLayout(batch_box)
        self.scenes_input = QTextEdit()
        self.scenes_input.setPlaceholderText(
            "Put ONE scene prompt per line.\n"
            "Example:\n"
            "Scene 1: Arctic lab exterior, drone shot...\n"
            "Scene 2: Inside lab corridor tracking shot...\n"
            "Scene 3: Close-up of scientist holding vial...\n\n"
            "Tip: Keep scenes 4–8 seconds each for best consistency."
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
        self.logs.setMinimumHeight(200)
        lv.addWidget(self.logs)
        right_layout.addWidget(logs_box)

        history_box = QGroupBox("History")
        hv = QVBoxLayout(history_box)
        self.history_table = QTableWidget(0, 9)
        self.history_table.setHorizontalHeaderLabels([
            "Time", "Mode", "Status", "Model", "Size", "Sec", "Prompt", "Clip", "Stitched"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        history_btn_row = QHBoxLayout()
        self.refresh_history_btn = QPushButton("Refresh")
        self.open_selected_btn = QPushButton("Open Selected Clip")
        self.open_selected_stitched_btn = QPushButton("Open Selected Stitched")
        history_btn_row.addWidget(self.refresh_history_btn)
        history_btn_row.addWidget(self.open_selected_btn)
        history_btn_row.addWidget(self.open_selected_stitched_btn)

        hv.addWidget(self.history_table)
        hv.addLayout(history_btn_row)
        right_layout.addWidget(history_box, 1)

        # ---------------- Wiring
        self.browse_out_btn.clicked.connect(self.pick_out_dir)
        self.ref_btn.clicked.connect(self.pick_reference_image)
        self.ref_clear_btn.clicked.connect(self.clear_reference_image)
        self.apply_template_btn.clicked.connect(self.apply_template)
        self.generate_single_btn.clicked.connect(self.start_single)
        self.generate_batch_btn.clicked.connect(self.start_batch)
        self.cancel_btn.clicked.connect(self.cancel_any)
        self.open_out_folder_btn.clicked.connect(self.open_output_folder)
        self.open_last_btn.clicked.connect(self.open_last_clip)
        self.open_stitched_btn.clicked.connect(self.open_last_stitched)
        self.refresh_history_btn.clicked.connect(self.load_history_table)
        self.open_selected_btn.clicked.connect(self.open_selected_clip)
        self.open_selected_stitched_btn.clicked.connect(self.open_selected_stitched)

        self.apply_theme()
        self.load_history_table()

        # Warn if ffmpeg missing but stitch enabled
        if self.stitch_checkbox.isChecked() and not ffmpeg_exists():
            self.log("⚠️ ffmpeg not found in PATH. Auto-stitch will fail until ffmpeg is installed.")

    # ---------------- UI helpers
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
                color: #bcd1ff; font-weight: 700;
            }
            QLineEdit, QTextEdit, QComboBox {
                background: #0f1b30; border: 1px solid #253046;
                border-radius: 10px; padding: 8px;
                selection-background-color: #2f5fff;
            }
            QPushButton {
                background: #2f5fff; border: none; border-radius: 12px;
                padding: 10px 12px; font-weight: 700;
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
                font-weight: 700;
            }
        """)
        self.setFont(QFont("Segoe UI", 10))

    def log(self, text: str):
        self.logs.append(text)

    def set_status(self, text: str):
        self.status_label.setText(text)

    def get_api_key(self) -> str:
        return self.api_key_input.text().strip() or os.environ.get("OPENAI_API_KEY", "").strip()

    def get_settings(self):
        return (
            self.model_combo.currentText(),
            self.size_combo.currentText(),
            self.seconds_combo.currentText(),
            int(self.poll_combo.currentText()),
            self.out_dir_input.text().strip(),
            (self.out_name_input.text().strip() or "scene"),
            self.tls_checkbox.isChecked(),
        )

    def history_store(self) -> HistoryStore:
        return HistoryStore(self.out_dir_input.text().strip())

    # ---------------- Actions
    def pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder", self.out_dir_input.text())
        if d:
            self.out_dir_input.setText(d)
            self.load_history_table()

    def pick_reference_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select reference image", "",
            "Images (*.png *.jpg *.jpeg *.webp);;All files (*.*)"
        )
        if path:
            self.reference_image_path = path
            self.ref_label.setText(f"Reference image: {pathlib.Path(path).name}")

    def clear_reference_image(self):
        self.reference_image_path = None
        self.ref_label.setText("Reference image: (none)")

    def apply_template(self):
        name = self.template_combo.currentText()
        if name == "(No template)":
            return
        tpl = TEMPLATES.get(name, "")
        # Put template in prompt box
        self.prompt_input.setPlainText(tpl)
        self.log(f"Template applied: {name}")

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

        api_key = self.get_api_key()
        prompt = self.prompt_input.toPlainText().strip()
        model, size, seconds, poll, out_dir, out_name, verify_tls = self.get_settings()

        if not prompt:
            QMessageBox.warning(self, "Prompt missing", "Please enter a prompt.")
            return

        self.logs.clear()
        self.progress_bar.setValue(0)
        self.set_status("Starting…")
        self.set_busy(True)
        self.last_output_path = None
        self.last_stitched_path = None

        self.single_worker = SingleSoraWorker(
            api_key=api_key,
            prompt=prompt,
            model=model,
            size=size,
            seconds=seconds,
            out_dir=out_dir,
            out_name=out_name,
            reference_image_path=self.reference_image_path,
            verify_tls=verify_tls,
            poll_interval_sec=poll
        )
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

        api_key = self.get_api_key()
        raw = self.scenes_input.toPlainText()
        prompts = [line.strip() for line in raw.splitlines() if line.strip()]

        model, size, seconds, poll, out_dir, out_name, verify_tls = self.get_settings()
        do_stitch = self.stitch_checkbox.isChecked()

        if not prompts:
            QMessageBox.warning(self, "No scenes", "Add at least one scene prompt (one per line).")
            return

        if do_stitch and not ffmpeg_exists():
            QMessageBox.critical(
                self,
                "ffmpeg missing",
                "Auto-stitch is enabled, but ffmpeg is not found in PATH.\n\n"
                "Install ffmpeg, restart PyCharm, then try again.\n"
                "Or untick Auto-stitch."
            )
            return

        self.logs.clear()
        self.progress_bar.setValue(0)
        self.set_status("Starting batch…")
        self.set_busy(True)
        self.last_output_path = None
        self.last_stitched_path = None

        self.batch_worker = BatchSoraWorker(
            api_key=api_key,
            prompts=prompts,
            model=model,
            size=size,
            seconds=seconds,
            out_dir=out_dir,
            out_name=out_name,
            reference_image_path=self.reference_image_path,
            verify_tls=verify_tls,
            poll_interval_sec=poll,
            do_stitch=do_stitch
        )
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

    # ---------------- Callbacks
    def on_single_done(self, out_path: str, video_id: str):
        self.last_output_path = out_path
        self.progress_bar.setValue(100)
        self.set_status("Done ✅")
        self.set_busy(False)

        # Save history
        model, size, seconds, *_ = self.get_settings()
        store = self.history_store()
        store.add(HistoryItem(
            created_at=dt.datetime.now().isoformat(timespec="seconds"),
            mode="single",
            status="completed",
            video_id=video_id,
            model=model,
            size=size,
            seconds=seconds,
            prompt_preview=short(self.prompt_input.toPlainText(), 140),
            output_path=out_path
        ))
        self.load_history_table()

    def on_single_failed(self, msg: str):
        self.progress_bar.setValue(0)
        self.set_status("Failed ❌")
        self.set_busy(False)
        self.log(f"❌ Error: {msg}")

        # Save history as failed
        model, size, seconds, *_ = self.get_settings()
        store = self.history_store()
        store.add(HistoryItem(
            created_at=dt.datetime.now().isoformat(timespec="seconds"),
            mode="single",
            status="failed",
            video_id="",
            model=model,
            size=size,
            seconds=seconds,
            prompt_preview=short(self.prompt_input.toPlainText(), 140),
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

        model, size, seconds, *_ = self.get_settings()
        store = self.history_store()
        store.add(HistoryItem(
            created_at=dt.datetime.now().isoformat(timespec="seconds"),
            mode="batch",
            status="completed",
            video_id="(multiple)",
            model=model,
            size=size,
            seconds=seconds,
            prompt_preview=short(self.scenes_input.toPlainText(), 140),
            output_path="; ".join(clip_paths[:3]) + ("…" if len(clip_paths) > 3 else ""),
            stitched_path=stitched_path or ""
        ))
        self.load_history_table()

    def on_batch_failed(self, msg: str):
        self.progress_bar.setValue(0)
        self.set_status("Failed ❌")
        self.set_busy(False)
        self.log(f"❌ Error: {msg}")

        model, size, seconds, *_ = self.get_settings()
        store = self.history_store()
        store.add(HistoryItem(
            created_at=dt.datetime.now().isoformat(timespec="seconds"),
            mode="batch",
            status="failed",
            video_id="(multiple)",
            model=model,
            size=size,
            seconds=seconds,
            prompt_preview=short(self.scenes_input.toPlainText(), 140),
            output_path="",
            error=msg
        ))
        self.load_history_table()
        QMessageBox.critical(self, "Batch failed", msg)

    # ---------------- Open actions
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

    # ---------------- History table
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
                it.model,
                it.size,
                it.seconds,
                it.prompt_preview,
                it.output_path,
                it.stitched_path,
            ]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v or "")
                if c in (0, 1, 2, 3, 4, 5):
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

    def open_selected_clip(self):
        it = self.selected_history_item()
        if not it:
            QMessageBox.information(self, "No selection", "Select a row in History first.")
            return

        # For batch, output_path may be a summary; open folder instead
        if it.mode == "batch":
            folder = self.out_dir_input.text().strip()
            if folder:
                open_path(folder)
            return

        if it.output_path and pathlib.Path(it.output_path).exists():
            open_path(it.output_path)
        else:
            QMessageBox.warning(self, "Missing file", "The clip file path is not available (or file was moved).")

    def open_selected_stitched(self):
        it = self.selected_history_item()
        if not it:
            QMessageBox.information(self, "No selection", "Select a row in History first.")
            return

        if it.stitched_path and pathlib.Path(it.stitched_path).exists():
            open_path(it.stitched_path)
        else:
            QMessageBox.warning(self, "No stitched file", "No stitched video for this entry (or file was moved).")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()