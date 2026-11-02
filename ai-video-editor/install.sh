#!/usr/bin/env bash
# Linux / macOS / Termux installer.  Usage:  bash install.sh [--with-models]
set -e
OS="$(uname -s)"
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Installing..."
  if   [ -n "$TERMUX_VERSION" ] || [ -d /data/data/com.termux ]; then pkg install -y ffmpeg python
  elif [ "$OS" = "Darwin" ]; then brew install ffmpeg
  elif command -v apt-get >/dev/null 2>&1; then sudo apt-get update && sudo apt-get install -y ffmpeg
  elif command -v dnf >/dev/null 2>&1; then sudo dnf install -y ffmpeg
  else echo "Please install ffmpeg manually: https://ffmpeg.org/download.html"; exit 1; fi
fi
python3 -m pip install -r "$(dirname "$0")/requirements.txt" || python3 -m pip install --break-system-packages -r "$(dirname "$0")/requirements.txt"
if command -v apt-get >/dev/null 2>&1 && ! command -v espeak-ng >/dev/null 2>&1; then sudo apt-get install -y espeak-ng || true; fi
python3 "$(dirname "$0")/scripts/env_check.py" --strict >/dev/null && echo "Core install OK."
if [ "$1" = "--with-models" ]; then
  python3 "$(dirname "$0")/scripts/setup_voice.py" || true
  python3 "$(dirname "$0")/scripts/setup_models.py" --matting --whisper tiny || true
fi
python3 "$(dirname "$0")/scripts/env_check.py" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['next']); [print('  optional:',x) for x in d['optional_next_steps']]"
