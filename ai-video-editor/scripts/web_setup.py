#!/usr/bin/env python3
"""One-shot bootstrap for claude.ai web (or any fresh container). Run ONCE per conversation, before the first render.

  python scripts/web_setup.py                      # core only: pip packages for effects/QA (fast, ~30-60 s)
  python scripts/web_setup.py --voice              # + natural voice (Kokoro fp32, ~350 MB, parallel download)
  python scripts/web_setup.py --voice --matting    # + person cut-out effects (15 MB)
  python scripts/web_setup.py --all                # + offline Whisper tiny (~116 MB)
  python scripts/web_setup.py --whisper base       # a bigger Whisper (~207 MB)

claude.ai containers start empty each conversation, so models are re-downloaded each time (from GitHub; fast there). Only request what the job needs.
Prints one compact JSON status at the end. Exit code 1 only if a REQUIRED tool is missing (ffmpeg/Pillow/numpy).
"""
import argparse, importlib.util, json, os, subprocess, sys, tarfile, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from common import find_binary

def has(mod): return importlib.util.find_spec(mod) is not None

def pip_install(pkgs):
    if not pkgs: return "nothing to install"
    for extra in (["--break-system-packages"], []):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *pkgs, *extra], capture_output=True, text=True)
        if r.returncode == 0: return "ok: " + " ".join(pkgs)
    return "FAILED: " + (r.stderr or "")[-200:].strip()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", action="store_true"); ap.add_argument("--matting", action="store_true"); ap.add_argument("--whisper", nargs="?", const="tiny", choices=["tiny", "tiny.en", "base"])
    ap.add_argument("--all", action="store_true"); ap.add_argument("--espeak", action="store_true", help="also install the robotic fallback voice (needs apt + root)")
    a = ap.parse_args(); t0 = time.time(); log = {}
    if a.all: a.voice = a.matting = True; a.whisper = a.whisper or "tiny"
    if not find_binary("ffmpeg"): print(json.dumps({"ready": False, "error": "ffmpeg not found; cannot render on this machine", "hint": "apt-get install -y ffmpeg"})); sys.exit(1)
    need = []
    for mod, pkg in (("PIL", "pillow"), ("numpy", "numpy"), ("cv2", "opencv-python-headless"), ("scipy", "scipy"), ("psutil", "psutil")):
        if not has(mod): need.append(pkg)
    if (a.voice or a.matting) and not has("onnxruntime"): need.append("onnxruntime")
    if a.voice and not has("kokoro_onnx"): need += ["kokoro-onnx", "soundfile"]
    if a.whisper and not has("sherpa_onnx"): need.append("sherpa-onnx")
    log["pip"] = pip_install(need)
    if a.espeak and not find_binary("espeak-ng"):
        r = subprocess.run(["apt-get", "install", "-y", "espeak-ng"], capture_output=True, text=True); log["espeak"] = "ok" if r.returncode == 0 else "failed (needs root); neural voice or your own audio recommended"
    from voice import models_root, model_dir, MODEL_FILE, VOICES_FILE, neural_ready
    import setup_models as SM, setup_voice as SV
    jobs = []
    if a.voice and not neural_ready():
        jobs += [("voice-model", SM.fetch, (SV.BASE + "kokoro-v1.0.onnx", model_dir() / MODEL_FILE, 300_000_000)), ("voice-files", SM.fetch, (SV.BASE + "voices-v1.0.bin", model_dir() / VOICES_FILE, 20_000_000))]
    if a.matting and not SM.matting_ready(): jobs.append(("matting", SM.fetch, (SM.RVM_URL, SM.matting_path(), SM.RVM_MIN)))
    if a.whisper and not SM.whisper_ready(a.whisper):
        folder, size, _ = SM.WHISPER[a.whisper]; arc = models_root() / "whisper" / (folder + ".tar.bz2"); jobs.append(("whisper", SM.fetch, (SM.WH_BASE + folder + ".tar.bz2", arc, int(size * 0.8))))
    with ThreadPoolExecutor(max_workers=4) as ex:                       # downloads are network-bound: run them in parallel
        futs = {n: ex.submit(f, *args) for n, f, args in jobs}; res = {n: f.result() for n, f in futs.items()}
    if a.whisper and "whisper" in res and res["whisper"]:
        folder = SM.WHISPER[a.whisper][0]; arc = models_root() / "whisper" / (folder + ".tar.bz2")
        SM.safe_extract(arc, models_root() / "whisper"); arc.unlink(missing_ok=True)
    log["downloads"] = {n: ("ok" if ok else "FAILED") for n, ok in res.items()} or "nothing to download"
    status = {"ffmpeg": True, "pillow": has("PIL"), "numpy": has("numpy"), "opencv": has("cv2"), "neural_voice": neural_ready(), "matting": SM.matting_ready(), "whisper": SM.whisper_ready(a.whisper or "tiny")}
    required_ok = status["pillow"] and status["numpy"]
    wanted_ok = (not a.voice or status["neural_voice"]) and (not a.matting or status["matting"]) and (not a.whisper or SM.whisper_ready(a.whisper))
    print(json.dumps({"ready": required_ok, "requested_ok": wanted_ok, "seconds": round(time.time() - t0, 1), "status": status, "log": log,
                      "note": None if wanted_ok else "some requested models failed to install: fall back to espeak/your own audio, or skip those effects"}, indent=1))
    sys.exit(0 if required_ok else 1)

if __name__ == "__main__":
    main()
