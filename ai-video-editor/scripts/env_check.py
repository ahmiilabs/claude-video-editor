#!/usr/bin/env python3
"""Detect what this machine can do. Run FIRST. Prints a JSON capability report (secrets are never printed).

  python scripts/env_check.py            # report
  python scripts/env_check.py --install  # also try to install missing basics (pip packages; espeak-ng via apt where possible)
  python scripts/env_check.py --strict   # exit code 1 if a REQUIRED dependency is missing (use in CI / installers)

REQUIRED for any rendering: ffmpeg+ffprobe, Pillow, numpy.   Everything else is optional and only enables extra features.
"""
import importlib, importlib.util, json, os, platform, shutil, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import FONT_DIR, find_binary, os_family, INSTALL_HINTS, pick_encoder
import config

PIP = {"PIL": "pillow", "numpy": "numpy", "cv2": "opencv-python-headless", "onnxruntime": "onnxruntime", "psutil": "psutil", "scipy": "scipy",
       "kokoro_onnx": "kokoro-onnx soundfile", "sherpa_onnx": "sherpa-onnx"}

def can(mod):
    try: return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError): return False

def pip_install(pkgs):
    for extra in (["--break-system-packages"], []):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs.split(), *extra], capture_output=True, text=True)
        if r.returncode == 0: return "ok"
    return "failed: " + (r.stderr or "")[-160:].strip()

def install_missing():
    notes = []
    for mod in ("PIL", "numpy", "cv2", "onnxruntime", "psutil", "scipy"):
        if not can(mod): notes.append(f"pip {PIP[mod]}: {pip_install(PIP[mod])}")
    if not find_binary("espeak-ng") and os_family() == "linux" and shutil.which("apt-get"):
        r = subprocess.run(["apt-get", "install", "-y", "espeak-ng"], capture_output=True, text=True); notes.append("apt espeak-ng: " + ("ok" if r.returncode == 0 else "failed (needs root)"))
    return notes

HOSTS = {"github.com": "model downloads", "pypi.org": "pip packages", "api.anthropic.com": "Anthropic API", "api.openai.com": "OpenAI API", "huggingface.co": "Hugging Face models"}

def reachable(host, timeout=2.5):
    import socket
    try:
        with socket.create_connection((host, 443), timeout=timeout): return True
    except OSError: return False

def probe_network():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(HOSTS)) as ex: res = dict(zip(HOSTS, ex.map(reachable, HOSTS)))
    return res

def ffmpeg_info():
    ff = find_binary("ffmpeg")
    if not ff: return {"found": False}
    out = subprocess.run([ff, "-hide_banner", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    filt = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    need = ["ass", "sidechaincompress", "loudnorm", "afftdn", "alimiter", "blackdetect", "freezedetect", "silencedetect", "volumedetect", "aresample"]
    return {"found": True, "version": out, "filters_missing": [f for f in need if f" {f} " not in filt], "libx264": " libx264 " in enc, "alpha_mov": " qtrle " in enc, "alpha_webm": " libvpx-vp9 " in enc,
            "hardware_h264": [e for e in ("h264_nvenc", "h264_qsv", "h264_videotoolbox") if e in enc]}

def main():
    notes = install_missing() if "--install" in sys.argv else []
    from voice import neural_ready, models_root
    from setup_models import matting_ready, whisper_ready
    ff = ffmpeg_info(); mods = {m: can(m) for m in PIP}
    in_web = os.path.exists("/mnt/user-data") or bool(os.environ.get("CLAUDE_AI_WEB"))
    required = {"ffmpeg": ff.get("found", False), "ffprobe": bool(find_binary("ffprobe")), "pillow": mods["PIL"], "numpy": mods["numpy"]}
    espeak = bool(find_binary("espeak-ng"))
    try: enc = pick_encoder()[1]
    except Exception: enc = "libx264"
    rep = {"platform": platform.platform(), "os_family": os_family(), "python": platform.python_version(), "mode": "web" if in_web else "code", "cpus": os.cpu_count(),
           "ffmpeg": ff, "python_modules": mods, "espeak_ng": espeak, "video_encoder_in_use": enc, "fonts_bundled": sorted(p.name for p in FONT_DIR.glob("*.ttf")),
           "models": {"root": str(models_root()), "neural_voice": neural_ready(), "matting_rvm": matting_ready(), "whisper_tiny": whisper_ready("tiny"), "whisper_base": whisper_ready("base")},
           "providers": config.status(), "install_notes": notes}
    try:
        import psutil; rep["ram_free_mb"] = int(psutil.virtual_memory().available / 1e6)
    except ImportError: pass
    f = {"text_to_reel": all(required.values()), "transitions_and_graphics": all(required.values()), "audio_pipeline": ff.get("found", False) and not ff.get("filters_missing"),
         "word_captions": ff.get("found", False) and "ass" not in ff.get("filters_missing", []), "clip_effects_basic": all(required.values()) and mods["cv2"],
         "face_tracking_haar": mods["cv2"], "person_matting_effects": mods["cv2"] and mods["onnxruntime"] and matting_ready(), "voice_natural_neural": neural_ready(),
         "voice_robotic_espeak": espeak, "voice_from_user_audio": ff.get("found", False), "transcription_offline_whisper": whisper_ready("tiny") or whisper_ready("base"),
         "transcription_word_level_faster_whisper": can("faster_whisper"), "ai_script_writer": rep["providers"]["llm"]["configured"], "ai_images": rep["providers"]["image"]["configured"],
         "ai_video_clips": rep["providers"]["video"]["configured"], "transparent_video_export": ff.get("alpha_mov", False), "batch_rendering": all(required.values())}
    net = probe_network() if "--no-network" not in sys.argv else {}; rep["network"] = net; web_notes = []
    from urllib.parse import urlparse
    for kind in ("llm", "image", "video"):
        prov = config.get_provider(kind); host = urlparse(prov.base_url).hostname if prov.base_url else None
        if prov.name != "none" and host and host not in ("localhost", "127.0.0.1", "::1") and net and not reachable(host):
            web_notes.append(f"{kind} provider host {host} is NOT reachable from this machine (the claude.ai sandbox only allows a few hosts): feature disabled")
            f[{"llm": "ai_script_writer", "image": "ai_images", "video": "ai_video_clips"}[kind]] = False
    if in_web:
        web_notes.append("claude.ai web: Claude itself is the script writer (no LLM key needed); AI image/video providers are normally unreachable here: use the user's own images/videos; containers reset every conversation, so run web_setup.py first")
        if net and not net.get("github.com"): web_notes.append("github.com unreachable: models cannot be downloaded; use espeak/user audio and skip matting/whisper")
    rep["web_notes"] = web_notes
    rep["features"] = f; rep["recommended_quality"] = "draft" if in_web or (rep.get("ram_free_mb", 9999) < 2000) else "hd"
    missing = [k for k, v in required.items() if not v]; rep["blocking_missing"] = missing; hints = []
    if not ff.get("found"): hints.append("ffmpeg: " + INSTALL_HINTS["ffmpeg"].get(os_family(), "install ffmpeg"))
    for m in ("PIL", "numpy"):
        if not mods[m]: hints.append(f"python: pip install {PIP[m]}")
    if ff.get("filters_missing"): hints.append("this ffmpeg lacks filters " + ", ".join(ff["filters_missing"]) + ": install a full build (e.g. gyan.dev full build on Windows)")
    rep["install_hints"] = hints
    rep["optional_next_steps"] = [s for s, ok in (("python scripts/setup_voice.py        # natural voice (~350 MB)", neural_ready()), ("python scripts/setup_models.py --matting   # person cut-out effects (15 MB)", matting_ready()),
                                                  ("python scripts/setup_models.py --whisper tiny  # offline transcription (~116 MB)", whisper_ready("tiny") or whisper_ready("base"))) if not ok]
    rep["next"] = ("Fix blocking items: " + "; ".join(hints)) if missing else "Ready. Write the spec, then run scripts/reel.py (or scripts/pipeline.py)."
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if "--strict" in sys.argv and missing: sys.exit(1)

if __name__ == "__main__":
    main()
