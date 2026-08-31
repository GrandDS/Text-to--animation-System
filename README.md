# Text-to-Animation System

A Python desktop project for turning text prompts into short video clips through two independent workflows:

1. **Sora Studio** — a PyQt6 desktop client that submits video-generation jobs to OpenAI's video API, monitors job progress, downloads completed clips, stores local history, and supports optional FFmpeg-based post-processing.
2. **Offline Video Studio** — a local generation workflow that creates an image from a text prompt using a Diffusers text-to-image model and converts that image into a short animated clip using pan/zoom motion and FFmpeg encoding.

## Project files

- `sora_studio.py` — API-based desktop video-generation application.
- `offline_video_studio.py` — local/offline text-to-image plus animated-video application.
- `requirements.txt` — Python dependencies.
- `.env.example` — example environment configuration for the API-based application.
- `.gitignore` — excludes virtual environments, IDE metadata, generated video files, histories, caches and secrets.

## Requirements

- Python 3.10+ recommended.
- FFmpeg installed and available on your system `PATH` for video encoding/stitching features.
- A compatible PyTorch environment. GPU acceleration is strongly recommended for the offline Diffusers workflow; CPU execution can be slow and memory intensive.
- An OpenAI API key is required only for `sora_studio.py`.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Sora Studio

Set your API key as an environment variable instead of committing it to source control.

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
python sora_studio.py
```

macOS/Linux:

```bash
export OPENAI_API_KEY="your_api_key_here"
python sora_studio.py
```

The application also provides a field for entering an API key at runtime. Do not save real credentials in this repository.

## Offline Video Studio

Run:

```bash
python offline_video_studio.py
```

On first use, the configured Diffusers model may need to be downloaded. Model weights are intentionally not included in this repository.

The offline workflow generates a keyframe from the prompt and then creates motion using a Ken Burns-style pan/zoom process before encoding the frames into a video.

## Notes

- Generated videos, render directories and local history files are intentionally excluded from Git.
- `.venv`, PyCharm `.idea` files and Python bytecode are not part of the distributable project.
- Availability, model names and parameters for external video APIs can change; verify current provider documentation when running the API-based workflow.

## Status

The repository contains the original application source supplied for this project, cleaned of local IDE/environment files and packaged for version control. Both Python source files have been syntax-checked with `py_compile`.
