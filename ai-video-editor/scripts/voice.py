"""Voiceover for each scene.

spec["voice"]["mode"]:
  auto    (default) -> user file if given, else neural if installed, else espeak, else silent
  neural  -> Kokoro-82M (natural, free, CPU). Run scripts/setup_voice.py once. ~2x realtime on CPU.
  file    -> the user's own recording for the whole reel (scene lengths split by word count)
  espeak  -> robotic offline fallback
  silent  -> no audio, durations estimated from word count

Neural voice notes (important for quality):
  * lang "hi" reads DEVANAGARI. Put the spoken Hindi/Urdu-style line in scene["voice_text"]
    (Devanagari), keep scene["narration"] in Roman for captions. Roman text sounds English-y.
  * lang "en-us"/"en-gb" reads normal English text.
  * Lines are split at sentence ends, with short natural pauses; results are cached by content
    so re-rendering after a visual tweak does not re-synthesize.
"""
import hashlib, os, re, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from common import run, have, duration_of

WPS = 2.6; MIN_SCENE = 1.6
MODEL_FILE = "kokoro-v1.0.onnx"; VOICES_FILE = "voices.bin"   # fp32 on purpose: the int8 build emits NaN/huge samples (audible glitches) for Hindi
DEFAULT_VOICE = {"hi": "hf_alpha", "en-us": "af_heart", "en-gb": "bf_emma"}
VOICE_HELP = {"hi": "hf_alpha/hf_beta (female), hm_omega/hm_psi (male)",
              "en-us": "af_heart, af_bella, af_nicole (female), am_michael, am_adam (male)",
              "en-gb": "bf_emma, bf_isabella (female), bm_george, bm_lewis (male)"}

def est_duration(text):
    n = max(1, len(re.findall(r"\S+", text or "")))
    return max(MIN_SCENE, n / WPS + 0.35)

# ------------------------------------------------------------------ neural
def models_root():
    """Cache root for ALL optional models (voice, matting, whisper). Override with AIVE_MODEL_DIR."""
    base = os.environ.get("AIVE_MODEL_DIR")
    for cand in ([Path(base)] if base else []) + [Path.home() / ".cache" / "ai-video-editor", Path(tempfile.gettempdir()) / "ai-video-editor-models"]:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            if os.access(cand, os.W_OK): return cand
        except Exception:
            continue
    return Path(tempfile.gettempdir())

def model_dir():
    """Kokoro voice files live in <root>/kokoro."""
    d = models_root() / "kokoro"; d.mkdir(parents=True, exist_ok=True); return d

def neural_ready():
    try:
        import kokoro_onnx  # noqa
    except ImportError:
        return False
    d = model_dir()
    return (d / MODEL_FILE).exists() and (d / VOICES_FILE).exists() and (d / MODEL_FILE).stat().st_size > 200_000_000

_engine = None
def _kokoro():
    global _engine
    if _engine is None:
        from kokoro_onnx import Kokoro
        d = model_dir(); _engine = Kokoro(str(d / MODEL_FILE), str(d / VOICES_FILE))
    return _engine

def _write_wav(path, samples, sr):
    import numpy as np
    from scipy.io import wavfile
    wavfile.write(str(path), sr, (np.clip(samples, -1, 1) * 32767).astype("int16"))

def split_sentences(text):
    parts = re.split(r"(?<=[\.\?\!।॥])\s+", text.strip())
    return [p for p in parts if p.strip()]

def _trim_silence(x, sr, thresh_db=-48.0, margin=0.06):
    """Strip leading/trailing silence ourselves (kokoro-onnx's built-in trim can return EMPTY audio on some builds)."""
    import numpy as np
    n = max(1, int(sr * 0.01)); m = len(x) // n
    if m < 3: return x
    db = 20 * np.log10(np.sqrt((x[: m * n].reshape(m, n) ** 2).mean(1)) + 1e-9); idx = np.nonzero(db > thresh_db)[0]
    if len(idx) == 0: return x
    a = max(0, int((idx[0] * n) - margin * sr)); b = min(len(x), int((idx[-1] + 1) * n + margin * sr)); return x[a:b]

def synth_neural(text, out_wav, lang="hi", voice=None, speed=1.0, pause=0.22):
    """Synthesize sentence by sentence and join with short pauses; returns out_wav. Raises if the model returns empty audio."""
    import numpy as np
    k = _kokoro(); voice = voice or DEFAULT_VOICE.get(lang, "af_heart")
    chunks, sr = [], 24000
    for sent in split_sentences(text) or [text]:
        try: s, sr = k.create(sent, voice=voice, speed=float(speed), lang=lang, trim=False)
        except TypeError: s, sr = k.create(sent, voice=voice, speed=float(speed), lang=lang)
        s = np.asarray(s, dtype="float32")
        if not np.isfinite(s).all():
            raise RuntimeError("neural TTS produced NaN/inf samples (corrupt or int8 model). Reinstall: delete the kokoro cache folder and run scripts/setup_voice.py")
        peak = float(np.abs(s).max()) if len(s) else 0.0
        if peak > 0.98: s = s * (0.98 / peak)
        s = _trim_silence(s, sr)
        if len(s) < sr * 0.15 and len(re.sub(r"\W+", "", sent)) > 3:
            raise RuntimeError(f"neural TTS returned empty audio for: {sent[:40]!r} (voice={voice}, lang={lang})")
        chunks.append(s); chunks.append(np.zeros(int(sr * pause), dtype="float32"))
    audio = np.concatenate(chunks[:-1]) if len(chunks) > 1 else chunks[0]
    _write_wav(out_wav, audio, sr)
    return out_wav

def tts_cache_dir():
    d = Path(os.environ.get("AIVE_TTS_CACHE") or (models_root() / "tts_cache")); d.mkdir(parents=True, exist_ok=True); return d

def prune_tts_cache(max_mb=500):
    """Delete the oldest cached clips once the cache exceeds max_mb."""
    files = sorted(tts_cache_dir().glob("cache_*.wav"), key=lambda p: p.stat().st_mtime); total = sum(f.stat().st_size for f in files) / 1e6
    while files and total > max_mb: f = files.pop(0); total -= f.stat().st_size / 1e6; f.unlink(missing_ok=True)

def _cache_key(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]

# ------------------------------------------------------------------ espeak / silence
def _espeak(text, out_wav, lang, speed, pitch):
    run(["espeak-ng", "-v", lang, "-s", str(speed), "-p", str(pitch), "-w", str(out_wav), text])

def _silence(path, secs):
    run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", f"{secs:.3f}", str(path)])

def _pad_to(src, dst, d):
    if src and os.path.exists(src):
        run(["ffmpeg", "-y", "-v", "error", "-i", src, "-af", "apad", "-t", f"{d:.3f}", "-ac", "1", "-ar", "24000", dst])
    else:
        _silence(dst, d)

def _concat(parts, wav, workdir, polish):
    lst = os.path.join(workdir, "voices.txt")
    open(lst, "w").write("".join(f"file '{os.path.abspath(p)}'\n" for p in parts))
    raw = os.path.join(workdir, "voice_raw.wav")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", raw])
    af = "highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11" if polish else "anull"
    run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-af", af, "-ar", "24000", "-ac", "1", wav])

# ------------------------------------------------------------------ main entry
def build_voice(spec, workdir):
    v = spec.get("voice", {}) or {}
    scenes = spec["scenes"]; os.makedirs(workdir, exist_ok=True)
    mode = v.get("mode", "auto")
    if mode == "auto":
        mode = "file" if v.get("file") else "neural" if neural_ready() else "espeak" if have("espeak-ng") else "silent"
    if mode == "neural" and not neural_ready():
        print("[voice] neural voice not installed -> run `python scripts/setup_voice.py`. Falling back to espeak.", file=sys.stderr)
        mode = "espeak" if have("espeak-ng") else "silent"
    if mode == "espeak" and not have("espeak-ng"):
        print("[voice] espeak-ng missing -> silent (run env_check.py --install)", file=sys.stderr); mode = "silent"

    if mode == "file":
        src = v["file"]; total = duration_of(src)
        if total <= 0: raise ValueError(f"voice.file is missing or unreadable (not an audio file?): {src}")
        weights = [max(1, len((s.get("narration") or s.get("text", "")).split())) for s in scenes]
        fixed = sum(s.get("duration", 0) for s in scenes)
        free_w = sum(w for w, s in zip(weights, scenes) if not s.get("duration")) or 1
        left = max(0.5, total - fixed)
        durs = [s["duration"] if s.get("duration") else left * w / free_w for w, s in zip(weights, scenes)]
        wav = os.path.join(workdir, "voice.wav")
        run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1", "-ar", "24000", wav])
        return durs, wav, mode

    if mode in ("neural", "espeak"):
        lang = v.get("lang", "hi" if mode == "neural" else "hi")
        speed = float(v.get("speed", 1.0 if mode == "neural" else 150)); pitch = int(v.get("pitch", 50))
        voice = v.get("voice") or DEFAULT_VOICE.get(lang)
        parts, durs = [], []
        for i, s in enumerate(scenes):
            spoken = s.get("voice_text") or s.get("narration") or s.get("text", "")
            if mode == "neural" and lang == "hi" and spoken and not re.search(r"[\u0900-\u097F]", spoken):
                print(f"[voice] scene {i + 1}: Hindi voice got Roman text; add 'voice_text' in Devanagari for natural sound.", file=sys.stderr)
            p = os.path.join(workdir, f"v{i:02d}.wav")
            if spoken.strip() and s.get("narration", spoken).strip():
                if mode == "neural":
                    key = _cache_key(spoken, lang, voice, speed)
                    cached = str(tts_cache_dir() / f"cache_{key}.wav")
                    if not os.path.exists(cached): synth_neural(spoken, cached, lang, voice, speed); prune_tts_cache()
                    p = cached
                else:
                    _espeak(spoken, p, lang, int(speed), pitch)
                d = duration_of(p) + float(s.get("pad", 0.35))
            else:
                p = None; d = float(s.get("duration", MIN_SCENE))
            d = max(d, float(s.get("duration", 0)))
            padded = os.path.join(workdir, f"v{i:02d}p.wav"); _pad_to(p, padded, d)
            parts.append(padded); durs.append(d)
            print(f"[voice] scene {i + 1}/{len(scenes)} {d:.1f}s", flush=True)
        wav = os.path.join(workdir, "voice.wav"); _concat(parts, wav, workdir, polish=(mode == "neural"))
        return durs, wav, mode

    durs = [float(s.get("duration") or est_duration(s.get("narration") or s.get("text", ""))) for s in scenes]
    return durs, None, "silent"
