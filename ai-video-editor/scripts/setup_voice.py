#!/usr/bin/env python3
"""One-time setup for the natural (neural) voice: Kokoro-82M via kokoro-onnx, CPU only.

  python scripts/setup_voice.py          # pip install + download ~120 MB of model files, then self-test

Works on claude.ai web (pip + GitHub downloads are allowed there) and on any local machine.
Models (~350 MB, fp32) are cached in ~/.cache/ai-video-editor/kokoro (override with AIVE_MODEL_DIR).
Safe to re-run: it skips what is already there.
"""
import os, subprocess, sys, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from voice import model_dir, MODEL_FILE, VOICES_FILE, neural_ready

BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
FILES = {MODEL_FILE: ("kokoro-v1.0.onnx", 300_000_000), VOICES_FILE: ("voices-v1.0.bin", 20_000_000)}   # fp32 (326 MB): the int8 build produces NaN audio

def pip_install():
    try:
        import kokoro_onnx  # noqa
        return True
    except ImportError:
        pass
    r = subprocess.run([sys.executable, "-m", "pip", "install", "kokoro-onnx", "soundfile", "--break-system-packages", "-q"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "kokoro-onnx", "soundfile", "-q"], capture_output=True, text=True)
    if r.returncode != 0:
        print("pip install failed:\n" + r.stderr[-800:]); return False
    return True

def download(name, remote, min_size, d):
    dest = d / name
    if dest.exists() and dest.stat().st_size > min_size:
        return True
    for attempt in range(3):
        try:
            print(f"downloading {remote} ..."); tmp = str(dest) + ".part"
            urllib.request.urlretrieve(BASE + remote, tmp); os.replace(tmp, dest)
            if dest.stat().st_size > min_size: return True
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
    return False

def main():
    d = model_dir(); d.mkdir(parents=True, exist_ok=True)
    ok = pip_install()
    for name, (remote, mn) in FILES.items():
        ok = download(name, remote, mn, d) and ok
    if not ok or not neural_ready():
        print("NEURAL VOICE NOT AVAILABLE. Fall back to espeak (robotic) or ask the user for their own recording.")
        print("If the network blocks github.com, the user can update network settings or upload a voice file instead.")
        sys.exit(1)
    from voice import synth_neural
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "t.wav"); synth_neural("Hello, this is a test.", p, "en-us", "af_heart", 1.0)
    print(f"OK neural voice ready ({os.path.getsize(p)} bytes test audio). models in {d}")

if __name__ == "__main__":
    main()
