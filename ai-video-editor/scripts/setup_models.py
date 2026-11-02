#!/usr/bin/env python3
"""One place to install optional AI models. Everything is cached in model_dir() and skipped if present.

  python scripts/setup_models.py --list
  python scripts/setup_models.py --voice            # natural TTS (Kokoro, ~120 MB)
  python scripts/setup_models.py --matting          # person segmentation (Robust Video Matting, 15 MB)
  python scripts/setup_models.py --whisper tiny     # offline transcription via sherpa-onnx (tiny ~116 MB, base ~207 MB)
  python scripts/setup_models.py --all

Models are downloaded from GitHub releases (works in the claude.ai sandbox). If a download fails the script says so and exits 1.
It never pretends: features stay disabled until the files verify.
"""
import argparse, os, sys, tarfile, urllib.request
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from voice import models_root, model_dir, neural_ready

RVM_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"
RVM_FILE = "rvm_mobilenetv3_fp32.onnx"; RVM_MIN = 14_000_000
WHISPER = {  # name -> (archive, approx size, multilingual)
    "tiny": ("sherpa-onnx-whisper-tiny", 110_000_000, True),
    "tiny.en": ("sherpa-onnx-whisper-tiny.en", 110_000_000, False),
    "base": ("sherpa-onnx-whisper-base", 200_000_000, True),
}
WH_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"

def matting_path():  return models_root() / "matting" / RVM_FILE
def whisper_dir(name="tiny"):  return models_root() / "whisper" / WHISPER[name][0]
def matting_ready():  p = matting_path(); return p.exists() and p.stat().st_size > RVM_MIN
def whisper_ready(name="tiny"):
    d = whisper_dir(name)
    try:
        import sherpa_onnx  # noqa
    except ImportError:
        return False
    return d.exists() and any(d.glob("*encoder*.onnx")) and any(d.glob("*tokens.txt"))

def fetch(url, dest, min_size, tries=3):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_size: return True
    for n in range(tries):
        try:
            print(f"downloading {url.split('/')[-1]} ...", flush=True); tmp = str(dest) + ".part"
            urllib.request.urlretrieve(url, tmp); os.replace(tmp, dest)
            if dest.stat().st_size >= min_size: return True
            print("  file too small, retrying")
        except Exception as e:
            print(f"  attempt {n + 1}/{tries} failed: {type(e).__name__}: {e}")
    return False

def pip_install(*pkgs):
    import subprocess
    for extra in (["--break-system-packages"], []):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs, *extra], capture_output=True, text=True)
        if r.returncode == 0: return True
    print("pip install failed:", r.stderr[-400:]); return False

def safe_extract(tar_path, dest):
    with tarfile.open(tar_path, "r:*") as t:
        base = Path(dest).resolve()
        for m in t.getmembers():
            if not str((base / m.name).resolve()).startswith(str(base)):
                raise RuntimeError(f"unsafe path in archive: {m.name}")
        t.extractall(dest)

def setup_matting():
    try:
        import onnxruntime  # noqa
    except ImportError:
        if not pip_install("onnxruntime"): return False
    return fetch(RVM_URL, matting_path(), RVM_MIN) and matting_ready()

def setup_whisper(name="tiny"):
    if name not in WHISPER: print("unknown whisper model; choose:", ", ".join(WHISPER)); return False
    try:
        import sherpa_onnx  # noqa
    except ImportError:
        if not pip_install("sherpa-onnx"): return False
    folder, size, _ = WHISPER[name]; arc = models_root() / "whisper" / (folder + ".tar.bz2")
    if not whisper_ready(name):
        if not fetch(WH_BASE + folder + ".tar.bz2", arc, size * 0.8): return False
        safe_extract(arc, models_root() / "whisper"); arc.unlink(missing_ok=True)
    return whisper_ready(name)

def setup_voice():
    from setup_voice import main as sv
    try: sv(); return neural_ready()
    except SystemExit as e: return e.code in (0, None)

def status():
    return {"model_root": str(models_root()), "voice_neural": neural_ready(), "matting_rvm": matting_ready(),
            **{f"whisper_{n}": whisper_ready(n) for n in WHISPER}}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", action="store_true"); ap.add_argument("--matting", action="store_true")
    ap.add_argument("--whisper", choices=list(WHISPER)); ap.add_argument("--all", action="store_true"); ap.add_argument("--list", action="store_true")
    a = ap.parse_args(); ok = True
    if a.list or not (a.voice or a.matting or a.whisper or a.all):
        import json; print(json.dumps(status(), indent=2)); return
    if a.voice or a.all: ok = setup_voice() and ok
    if a.matting or a.all: ok = setup_matting() and ok
    if a.whisper or a.all: ok = setup_whisper(a.whisper or "tiny") and ok
    import json; print(json.dumps(status(), indent=2))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
