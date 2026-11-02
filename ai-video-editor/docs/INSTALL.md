# Installation

Required: **ffmpeg + ffprobe** (full build: libass, sidechaincompress, loudnorm, afftdn), Python 3.9+, `pillow`, `numpy`. Everything else is optional.

## Linux (Debian/Ubuntu)
```bash
sudo apt-get install -y ffmpeg espeak-ng python3-pip
pip install -r requirements.txt            # add --break-system-packages on managed Pythons
python scripts/env_check.py --strict
```
## macOS
```bash
brew install ffmpeg espeak-ng && pip3 install -r requirements.txt
```
## Windows 10/11 (PowerShell)
```powershell
winget install Gyan.FFmpeg          # then reopen PowerShell so PATH updates
python -m pip install -r requirements.txt
python scripts\env_check.py --strict
```
If ffmpeg lives elsewhere set `AIVE_FFMPEG` / `AIVE_FFPROBE` to the full path. Install `espeak-ng` from https://github.com/espeak-ng/espeak-ng/releases (optional, robotic fallback voice).
## Termux (Android)
```bash
pkg install python ffmpeg espeak && pip install pillow numpy      # opencv/onnxruntime wheels may be unavailable: effects and AI models then stay disabled
```
## Optional models (all free, downloaded from GitHub releases)
```bash
python scripts/setup_voice.py                    # natural voice (Kokoro fp32, ~350 MB)
python scripts/setup_models.py --matting         # person cut-out effects (15 MB)
python scripts/setup_models.py --whisper tiny    # offline transcription (~116 MB); or base (~207 MB)
python scripts/setup_models.py --list            # what is installed
```
Models are cached in `~/.cache/ai-video-editor` (`AIVE_MODEL_DIR` to change). Installers: `install.sh [--with-models]`, `install.ps1 -WithModels`.

Check everything: `python scripts/env_check.py` (feature matrix, missing filters, encoder in use, models, provider status without secrets).
GPU: no GPU is required. If NVENC/QSV/VideoToolbox is present it is used only after a real 1-frame trial encode succeeds (`AIVE_GPU=off` to force CPU).
Memory: draft = 720p (~700 MB), hd = 1080p (~1.4 GB); hd is downgraded automatically when little RAM is free.
