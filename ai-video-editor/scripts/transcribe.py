#!/usr/bin/env python3
"""Speech recognition -> timed words -> SRT / ASS captions.

  python scripts/transcribe.py clip.mp4 --lang en --srt out.srt --ass out.ass --style karaoke --json out.json

Works on uploaded audio, generated voiceovers, and the audio track of any video (decoded through ffmpeg).

ENGINES (--engine, or AIVE_TRANSCRIBE_ENGINE; "auto" tries them in this order)
  faster-whisper   TRUE word timestamps. Needs `pip install faster-whisper` and a model (downloaded from Hugging Face on first use).
  openai-whisper   TRUE word timestamps. Needs `pip install openai-whisper` and its model download.
  openai-api       TRUE word timestamps via an OpenAI-compatible /audio/transcriptions endpoint. Needs AIVE_TRANSCRIBE_API_KEY.
  sherpa           Offline Whisper via sherpa-onnx (`setup_models.py --whisper tiny`). Real SENTENCE-level timestamps; words are
                   placed inside each sentence using the audio's voiced regions (granularity = "segment", words flagged approx).

Every transcript states its `granularity` ("word" or "segment") so callers and reports never overstate the accuracy.
"""
import argparse, json, os, re, sys, uuid
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import run, find_binary, get_logger, Workspace
import config

log = get_logger("aive.transcribe")
SR = 16000

class TranscribeError(RuntimeError):
    pass

# ------------------------------------------------------------------ audio helpers
def load_audio(path, sr=SR):
    """Decode ANY audio/video file to mono float32 via ffmpeg."""
    import subprocess
    r = subprocess.run([find_binary("ffmpeg", required=True), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise TranscribeError(f"cannot decode audio from {path}: {r.stderr.decode(errors='replace')[:200] or 'no audio stream'}")
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0

def energy_envelope(a, sr=SR, hop=0.01):
    n = int(sr * hop); m = len(a) // n
    if m == 0: return np.zeros(1), hop
    return np.sqrt((a[: m * n].reshape(m, n) ** 2).mean(1) + 1e-12), hop

def voiced_mask(env, hop=0.01, min_gap=0.12, min_len=0.06):
    """Boolean per-frame speech/non-speech from an adaptive threshold (relative to the clip's own level)."""
    if len(env) < 3: return np.ones(len(env), bool)
    db = 20 * np.log10(env + 1e-9); hi = np.percentile(db, 95); thr = max(hi - 28, np.percentile(db, 15) + 6, -55)
    m = db > thr; g = max(1, int(min_gap / hop)); i = 0; n = len(m)
    while i < n:                                   # close short gaps inside speech
        if not m[i]:
            j = i
            while j < n and not m[j]: j += 1
            if 0 < i and j < n and (j - i) < g: m[i:j] = True
            i = j
        else: i += 1
    ml = max(1, int(min_len / hop)); i = 0
    while i < n:                                   # drop blips
        if m[i]:
            j = i
            while j < n and m[j]: j += 1
            if (j - i) < ml: m[i:j] = False
            i = j
        else: i += 1
    return m

def _weight(word):
    w = re.sub(r"\W+", "", word.lower()); v = len(re.findall(r"[aeiouyअ-औा-ौ]", w))
    return max(1.0, 0.6 * len(w) + 0.9 * v)

def place_words(words_text, t0, t1, mask, hop=0.01, scene=None):
    """Distribute words over [t0, t1] using only the *voiced* time inside it (pauses consume no word weight)."""
    toks = re.findall(r"\S+", words_text)
    if not toks: return []
    a, b = int(t0 / hop), min(len(mask), int(np.ceil(t1 / hop)))
    seg = mask[a:b]; idx = np.nonzero(seg)[0]
    if len(idx) < len(toks) * 2: idx = np.arange(len(seg))        # almost no detectable speech: fall back to even spacing
    if len(idx) == 0: return []
    wts = np.array([_weight(t) for t in toks]); cum = np.concatenate([[0], np.cumsum(wts)]) / wts.sum()
    out = []
    for k, tok in enumerate(toks):
        i0 = idx[min(len(idx) - 1, int(cum[k] * len(idx)))]; i1 = idx[min(len(idx) - 1, max(int(cum[k + 1] * len(idx)) - 1, int(cum[k] * len(idx))))]
        d = {"w": tok, "start": round((a + i0) * hop, 3), "end": round((a + i1 + 1) * hop, 3), "approx": True}
        if scene is not None: d["scene"] = scene
        out.append(d)
    return out

def trim_to_voiced(t0, t1, mask, hop=0.01):
    a, b = int(t0 / hop), min(len(mask), int(np.ceil(t1 / hop))); idx = np.nonzero(mask[a:b])[0]
    return (t0, t1) if len(idx) == 0 else ((a + idx[0]) * hop, (a + idx[-1] + 1) * hop)

# ------------------------------------------------------------------ engines
def _sherpa(audio, lang, model):
    from setup_models import whisper_ready, whisper_dir
    try: import sherpa_onnx
    except ImportError: raise TranscribeError("sherpa-onnx not installed: pip install sherpa-onnx")
    if not whisper_ready(model): raise TranscribeError(f"whisper '{model}' not installed: python scripts/setup_models.py --whisper {model}")
    d = whisper_dir(model); enc = next(d.glob(f"*encoder.int8.onnx"), None) or next(d.glob("*encoder*.onnx")); dec = next(d.glob("*decoder.int8.onnx"), None) or next(d.glob("*decoder*.onnx")); tok = next(d.glob("*tokens.txt"))
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=str(enc), decoder=str(dec), tokens=str(tok), language=lang if lang != "auto" else "", task="transcribe",
                                                      num_threads=max(1, os.cpu_count() or 1), enable_segment_timestamps=True)
    env, hop = energy_envelope(audio); chunk = 28 * SR; pos = 0; segs = []
    while pos < len(audio):
        end = min(len(audio), pos + chunk)
        if end < len(audio):                                    # cut at the quietest point of the last 4 s so words are not split
            lo = max(pos, end - 4 * SR); frames = env[int(lo / SR / hop): int(end / SR / hop)]
            if len(frames): end = int(lo + int(np.argmin(frames)) * hop * SR)
        piece = audio[pos:end]; st = rec.create_stream(); st.accept_waveform(SR, piece); rec.decode_stream(st); r = st.result
        starts = list(r.segment_timestamps) or [0.0]; texts = list(r.segment_texts) or [r.text]
        for i, (s0, tx) in enumerate(zip(starts, texts)):
            e0 = starts[i + 1] if i + 1 < len(starts) else len(piece) / SR
            if tx.strip(): segs.append({"start": pos / SR + s0, "end": pos / SR + e0, "text": tx.strip()})
        pos = end
    return segs, "segment"

def _faster_whisper(path, lang, model):
    try: from faster_whisper import WhisperModel
    except ImportError: raise TranscribeError("faster-whisper not installed: pip install faster-whisper")
    try: m = WhisperModel(model, device="auto", compute_type="auto")
    except Exception as e: raise TranscribeError(f"faster-whisper could not load model '{model}' (offline or blocked download?): {config.redact(e)}")
    it, _ = m.transcribe(str(path), language=None if lang == "auto" else lang, word_timestamps=True, vad_filter=True)
    segs = []
    for s in it:
        segs.append({"start": s.start, "end": s.end, "text": s.text.strip(), "words": [{"w": w.word.strip(), "start": w.start, "end": w.end, "approx": False} for w in (s.words or []) if w.word.strip()]})
    return segs, "word"

def _openai_whisper(path, lang, model):
    try: import whisper
    except ImportError: raise TranscribeError("openai-whisper not installed: pip install openai-whisper")
    try: m = whisper.load_model(model)
    except Exception as e: raise TranscribeError(f"openai-whisper could not load '{model}': {config.redact(e)}")
    r = m.transcribe(str(path), language=None if lang == "auto" else lang, word_timestamps=True); segs = []
    for s in r["segments"]:
        segs.append({"start": s["start"], "end": s["end"], "text": s["text"].strip(), "words": [{"w": w["word"].strip(), "start": w["start"], "end": w["end"], "approx": False} for w in s.get("words", []) if w["word"].strip()]})
    return segs, "word"

def _multipart(fields, files):
    b = "----aive" + uuid.uuid4().hex; body = b""
    for k, v in fields:
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, (fn, data, ct) in files.items():
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{fn}"\r\nContent-Type: {ct}\r\n\r\n'.encode() + data + b"\r\n"
    return body + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"

def _openai_api(path, lang, model):
    import urllib.request, urllib.error
    key = os.environ.get("AIVE_TRANSCRIBE_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    config.load_env(); key = key or os.environ.get("AIVE_TRANSCRIBE_API_KEY", "")
    if not key or config.PLACEHOLDER_RE.match(key): raise TranscribeError("openai-api engine needs AIVE_TRANSCRIBE_API_KEY (or OPENAI_API_KEY)")
    config._register_secret(key); base = (os.environ.get("AIVE_TRANSCRIBE_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    with Workspace("tr_") as ws:                                    # small mono mp3 keeps uploads under API size limits
        mp3 = ws.path / "a.mp3"; run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(mp3)]); data = mp3.read_bytes()
    fields = [("model", model if model not in ("tiny", "base", "small") else "whisper-1"), ("response_format", "verbose_json"), ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment")]
    if lang != "auto": fields.append(("language", lang))
    body, ctype = _multipart(fields, {"file": ("a.mp3", data, "audio/mpeg")})
    req = urllib.request.Request(base + "/audio/transcriptions", data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=float(os.environ.get("AIVE_TIMEOUT", "120"))) as r: j = json.loads(r.read())
    except urllib.error.HTTPError as e: raise TranscribeError(f"transcription API error HTTP {e.code}: {config.redact(e.read().decode(errors='replace')[:200])}")
    except Exception as e: raise TranscribeError(f"transcription API request failed: {config.redact(e)}")
    words = [{"w": w["word"].strip(), "start": w["start"], "end": w["end"], "approx": False} for w in j.get("words", [])]
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in j.get("segments", [])] or [{"start": 0, "end": words[-1]["end"] if words else 0, "text": j.get("text", "")}]
    for s in segs: s["words"] = [w for w in words if s["start"] - 0.05 <= w["start"] < s["end"] + 0.05]
    return segs, "word" if words else "segment"

ENGINES = ["faster-whisper", "openai-api", "sherpa", "openai-whisper"]

def available_engines():
    from setup_models import whisper_ready
    out = {}
    import importlib.util
    out["faster-whisper"] = importlib.util.find_spec("faster_whisper") is not None
    out["openai-whisper"] = importlib.util.find_spec("whisper") is not None
    out["sherpa"] = whisper_ready(os.environ.get("AIVE_WHISPER_MODEL", "tiny"))
    config.load_env(); k = os.environ.get("AIVE_TRANSCRIBE_API_KEY", "")
    out["openai-api"] = bool(k) and not config.PLACEHOLDER_RE.match(k)
    return out

def transcribe(path, lang="auto", engine=None, model=None):
    """-> {"engine","language","granularity","segments":[{start,end,text,words}],"words":[flat]}  (raises TranscribeError, never fakes)."""
    config.load_env(); engine = (engine or os.environ.get("AIVE_TRANSCRIBE_ENGINE", "auto")).lower(); model = model or os.environ.get("AIVE_WHISPER_MODEL", "tiny")
    audio = load_audio(path)
    if len(audio) < SR * 0.3: raise TranscribeError("audio is empty or shorter than 0.3 s")
    order = ENGINES if engine == "auto" else [engine]; errors = []
    for e in order:
        try:
            if e not in ENGINES: raise TranscribeError(f"unknown engine '{e}'. Choose from: auto, {', '.join(ENGINES)}")
            log.info("transcribing with %s ...", e)
            if e == "sherpa": segs, gran = _sherpa(audio, lang, model)
            elif e == "faster-whisper": segs, gran = _faster_whisper(path, lang, model)
            elif e == "openai-whisper": segs, gran = _openai_whisper(path, lang, model)
            else: segs, gran = _openai_api(path, lang, model)
            break
        except TranscribeError as ex:
            errors.append(f"{e}: {ex}")
            if engine != "auto": raise
    else:
        raise TranscribeError("no transcription engine available. Tried:\n  " + "\n  ".join(errors) + "\nInstall one: python scripts/setup_models.py --whisper tiny")
    env, hop = energy_envelope(audio); mask = voiced_mask(env, hop); words = []
    NONSPEECH = re.compile(r"^\s*[\[\(\*♪].*[\]\)\*♪]?\s*$")        # Whisper hallucinates "[MUSIC]" / "(applause)" on silence
    segs = [s for s in segs if s["text"].strip() and not NONSPEECH.match(s["text"])]
    if not segs: raise TranscribeError("no speech detected in the audio")
    for s in segs:
        if not s.get("words"):
            a, b = trim_to_voiced(s["start"], s["end"], mask, hop); s["start"], s["end"] = round(a, 3), round(b, 3)
            s["words"] = place_words(s["text"], s["start"], s["end"], mask, hop)
        words += s["words"]
    return {"engine": e, "language": lang, "granularity": gran, "audio_seconds": round(len(audio) / SR, 2), "segments": segs, "words": words,
            "note": ("word times are interpolated inside real sentence timestamps" if gran == "segment" else "engine-provided word timestamps")}

# ------------------------------------------------------------------ alignment for generated voiceovers (known script)
def align_scenes(wav, scenes, durs, texts):
    """Timing for TTS voiceovers where the script is KNOWN: each scene's window is exact (we synthesised it), and words are placed
    inside the voiced part of that window using the real audio. granularity = 'scene-exact' (not recognition, not word-exact)."""
    audio = load_audio(wav); env, hop = energy_envelope(audio); mask = voiced_mask(env, hop); out, t0 = [], 0.0
    for i, (d, tx) in enumerate(zip(durs, texts)):
        if tx and scenes[i].get("captions", True) is not False:
            out += place_words(tx, t0, t0 + d, mask, hop, scene=i)
        t0 += d
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("media", nargs="?"); ap.add_argument("--lang", default="auto"); ap.add_argument("--engine", default=None); ap.add_argument("--model", default=None)
    ap.add_argument("--srt"); ap.add_argument("--ass"); ap.add_argument("--json"); ap.add_argument("--style", default="karaoke", choices=["highlight", "karaoke", "plain"])
    ap.add_argument("--size", default="720x1280"); ap.add_argument("--list-engines", action="store_true")
    a = ap.parse_args()
    if a.list_engines or not a.media:
        print(json.dumps(available_engines(), indent=2)); return
    try: tr = transcribe(a.media, a.lang, a.engine, a.model)
    except TranscribeError as e:
        print("TRANSCRIPTION FAILED:", e, file=sys.stderr); sys.exit(1)
    import captions
    if a.json: Path(a.json).write_text(json.dumps(tr, indent=2, ensure_ascii=False), encoding="utf-8")
    if a.srt: captions.write_srt(tr["words"], a.srt)
    if a.ass:
        w, h = (int(x) for x in a.size.split("x")); captions.write_ass(tr["words"], w, h, a.ass, a.style)
    print(f"engine={tr['engine']} granularity={tr['granularity']} words={len(tr['words'])}\n{' '.join(s['text'] for s in tr['segments'])}")

if __name__ == "__main__":
    main()
