#!/usr/bin/env python3
"""Professional audio pipeline (FFmpeg filters only, no extra dependencies).

  python scripts/audio.py enhance --in raw.wav --out clean.wav [--no-denoise]
  python scripts/audio.py mix --voice v.wav --music bed.mp3 --out mix.wav [--sfx whoosh.wav@1.5 ...] [--no-duck]
  python scripts/audio.py synth-music --kind calm --seconds 30 --out bed.wav
  python scripts/audio.py synth-sfx --kind whoosh --out whoosh.wav
  python scripts/audio.py analyse mix.wav

Mix chain:  voice -> (gain) -+
            music -> fades -> sidechaincompress(keyed by the raw voice) -+-> amix -> alimiter -> loudnorm -> alimiter
            sfx   -> adelay -> gain ----------------------------------+
Ducking is real sidechain compression: the music drops while narration is present and recovers after.
Output length is forced to the exact timeline length so audio never drifts from the video.
"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from common import run, find_binary, duration_of, ffprobe, get_logger

log = get_logger("aive.audio")
SR = 44100

def _fmt():  return f"aformat=sample_rates={SR}:channel_layouts=stereo"

def enhance_voice(src, dst, denoise=True, highpass=80, lowpass=15000, compress=True, target_lufs=-16.0):
    """Clean a voice recording: high-pass rumble, FFT denoise, gentle compression, loudness normalisation, peak limit."""
    chain = [f"highpass=f={highpass}"]
    if denoise: chain.append("afftdn=nr=14:nf=-38:tn=1")
    if lowpass: chain.append(f"lowpass=f={lowpass}")
    if compress: chain.append("acompressor=threshold=-20dB:ratio=3:attack=8:release=120:makeup=2")
    chain += [f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11", "alimiter=limit=0.89:level=disabled"]
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-af", ",".join(chain), "-ar", str(SR), "-ac", "1", str(dst)])
    return str(dst)

def synth_music(kind, seconds, dst):
    """Royalty-free synthetic background bed (sine pad + slow tremolo). A placeholder for real music; users should supply their own."""
    chords = {"calm": [220.0, 277.18, 329.63, 440.0], "warm": [196.0, 246.94, 293.66, 392.0],
              "tense": [174.61, 207.65, 261.63, 349.23], "upbeat": [261.63, 329.63, 392.0, 523.25]}
    freqs = chords.get(kind, chords["calm"]); trem = 0.22 if kind != "upbeat" else 1.6
    inputs, labels = [], []
    for i, f in enumerate(freqs):
        inputs += ["-f", "lavfi", "-i", f"sine=f={f}:d={seconds}:r={SR}"]; labels.append(f"[{i}:a]")
    flt = "".join(labels) + f"amix=inputs={len(freqs)}:normalize=0,volume=0.35,tremolo=f={trem}:d=0.35,lowpass=f=1800,aformat=sample_rates={SR}:channel_layouts=stereo[o]"
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", flt, "-map", "[o]", "-t", str(seconds), str(dst)])
    return str(dst)

def synth_sfx(kind, dst):
    """Small synthetic effects: whoosh, pop, riser, ding."""
    specs = {
        "whoosh": ("anoisesrc=d=0.7:c=pink:a=0.6:r=44100", "highpass=f=300,lowpass=f=6000,afade=t=in:d=0.25,afade=t=out:st=0.3:d=0.4,tremolo=f=6:d=0.2"),
        "pop": ("sine=f=520:d=0.12:r=44100", "afade=t=out:st=0.02:d=0.1,volume=0.8"),
        "riser": ("anoisesrc=d=1.2:c=white:a=0.5:r=44100", "highpass=f=500,lowpass=f=9000,afade=t=in:d=1.1,volume=0.8"),
        "ding": ("sine=f=1320:d=0.6:r=44100", "afade=t=out:st=0.05:d=0.55,volume=0.7"),
    }
    if kind not in specs: raise ValueError(f"unknown sfx '{kind}'. Choose from {', '.join(specs)}")
    src, flt = specs[kind]
    run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", src, "-af", flt + f",aformat=sample_rates={SR}:channel_layouts=stereo", str(dst)])
    return str(dst)

def parse_sfx_arg(s):
    """'file.wav@1.5' or 'file.wav@1.5:0.6' (file@seconds[:volume])"""
    m = re.match(r"^(.*)@([\d.]+)(?::([\d.]+))?$", s)
    if not m: raise SystemExit(f"bad --sfx '{s}', expected file@seconds[:volume]")
    return {"file": m.group(1), "at": float(m.group(2)), "volume": float(m.group(3) or 0.7)}

def mix(out, total, voice=None, music=None, sfx=None, voice_vol=1.0, music_vol=0.22, duck=True, fade_in=0.4, fade_out=1.2,
        target_lufs=-15.0, duck_threshold=0.03, duck_ratio=8.0):
    """Build the final soundtrack of exactly `total` seconds. Returns a dict describing what was applied."""
    sfx = [s for s in (sfx or []) if s.get("file") and os.path.exists(s["file"])]
    if not (voice or music or sfx):
        raise ValueError("mix(): nothing to mix (no voice, music or sfx)")
    T = f"{total:.3f}"; cmd = ["ffmpeg", "-y", "-v", "error"]; idx = {}; n = 0
    key_gain = 0.0
    if duck and voice and music and os.path.exists(str(voice)):          # key the compressor on a LEVEL-NORMALISED copy of the voice so quiet recordings still duck the music
        lufs = analyse(voice).get("lufs")
        if lufs is not None and lufs > -69: key_gain = max(-6.0, min(24.0, -18.0 - lufs))
    for key, path, loop in (("voice", voice, False), ("music", music, True)):
        if path:
            if not os.path.exists(path): raise FileNotFoundError(f"{key} file not found: {path}")
            if loop: cmd += ["-stream_loop", "-1"]
            cmd += ["-i", str(path)]; idx[key] = n; n += 1
    sfx_idx = []
    for s in sfx:
        cmd += ["-i", str(s["file"])]; sfx_idx.append(n); n += 1
    parts, mixin = [], []
    if "voice" in idx:
        v = idx["voice"]
        if duck and "music" in idx:
            parts.append(f"[{v}:a]{_fmt()},apad=whole_dur={T},atrim=0:{T},asplit=2[vk0][vmain]"); parts.append(f"[vk0]volume={key_gain:.1f}dB[vkey]")
        else:
            parts.append(f"[{v}:a]{_fmt()},apad=whole_dur={T},atrim=0:{T}[vmain]")
        parts.append(f"[vmain]volume={voice_vol}[vo]"); mixin.append("[vo]")
    if "music" in idx:
        m = idx["music"]; fo = max(0.0, min(fade_out, total / 2)); st = max(0.0, total - fo)
        parts.append(f"[{m}:a]{_fmt()},atrim=0:{T},asetpts=PTS-STARTPTS,volume={music_vol},afade=t=in:d={fade_in},afade=t=out:st={st:.3f}:d={fo:.3f}[mus]")
        if duck and "voice" in idx:
            parts.append(f"[mus][vkey]sidechaincompress=threshold={duck_threshold}:ratio={duck_ratio}:attack=20:release=450:makeup=1[mu]")
            mixin.append("[mu]")
        else:
            mixin.append("[mus]")
    for k, (s, i) in enumerate(zip(sfx, sfx_idx)):
        ms = int(max(0.0, float(s.get("at", 0))) * 1000)
        parts.append(f"[{i}:a]{_fmt()},adelay={ms}|{ms},volume={float(s.get('volume', 0.7))}[s{k}]"); mixin.append(f"[s{k}]")
    parts.append("".join(mixin) + f"amix=inputs={len(mixin)}:duration=longest:normalize=0,apad=whole_dur={T},atrim=0:{T},"
                 f"alimiter=limit=0.89:level=disabled,loudnorm=I={target_lufs}:TP=-1.5:LRA=11,alimiter=limit=0.89:level=disabled,"
                 f"aresample={SR},atrim=0:{T}[out]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[out]", "-ar", str(SR), "-ac", "2", str(out)]
    run(cmd)
    validate_audio(out, total)
    return {"voice": bool(voice), "music": bool(music), "ducking": bool(duck and voice and music), "sfx": len(sfx), "duration": round(duration_of(out), 2)}

def validate_audio(path, expect_duration=None, tol=0.25):
    """Raise if the audio stream is missing/invalid, silent, or the wrong length (prevents invalid streams reaching the mux)."""
    info = ffprobe(path); au = [s for s in info["streams"] if s["codec_type"] == "audio"]
    if not au: raise RuntimeError(f"audio validation failed: no audio stream in {path}")
    d = float(info["format"].get("duration") or 0)
    if expect_duration and abs(d - expect_duration) > tol:
        raise RuntimeError(f"audio validation failed: duration {d:.2f}s != timeline {expect_duration:.2f}s")
    return True

def analyse(path, window=None):
    """Return peak/mean level (dB) and integrated loudness. window=(start,end) analyses only that part."""
    args = ["-hide_banner", "-nostats", "-v", "info"]
    if window: args += ["-ss", str(window[0]), "-t", str(window[1] - window[0])]
    r = subprocess.run([find_binary("ffmpeg", required=True), *args, "-i", str(path), "-vn", "-af", "volumedetect,ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True, errors="replace")
    def g(pat, last=False):
        m = re.findall(pat, r.stderr)
        return float(m[-1 if last else 0]) if m else None
    # ebur128 prints running values then a Summary block; the LAST "I:" / "Peak:" are the final numbers
    return {"peak_db": g(r"max_volume: (-?[\d.]+) dB"), "mean_db": g(r"mean_volume: (-?[\d.]+) dB"),
            "lufs": g(r"I:\s+(-?[\d.]+) LUFS", True), "true_peak_dbfs": g(r"Peak:\s+(-?[\d.]+) dBFS", True)}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enhance"); e.add_argument("--in", dest="inp", required=True); e.add_argument("--out", required=True); e.add_argument("--no-denoise", action="store_true")
    m = sub.add_parser("mix"); m.add_argument("--voice"); m.add_argument("--music"); m.add_argument("--sfx", action="append", default=[]); m.add_argument("--out", required=True)
    m.add_argument("--seconds", type=float); m.add_argument("--voice-vol", type=float, default=1.0); m.add_argument("--music-vol", type=float, default=0.22); m.add_argument("--no-duck", action="store_true")
    sm = sub.add_parser("synth-music"); sm.add_argument("--kind", default="calm"); sm.add_argument("--seconds", type=float, default=30); sm.add_argument("--out", required=True)
    ss = sub.add_parser("synth-sfx"); ss.add_argument("--kind", default="whoosh"); ss.add_argument("--out", required=True)
    an = sub.add_parser("analyse"); an.add_argument("file")
    a = ap.parse_args()
    if a.cmd == "enhance": print(enhance_voice(a.inp, a.out, denoise=not a.no_denoise))
    elif a.cmd == "mix":
        total = a.seconds or max(duration_of(x) for x in (a.voice, a.music) if x)
        print(json.dumps(mix(a.out, total, a.voice, a.music, [parse_sfx_arg(s) for s in a.sfx], a.voice_vol, a.music_vol, not a.no_duck)))
    elif a.cmd == "synth-music": print(synth_music(a.kind, a.seconds, a.out))
    elif a.cmd == "synth-sfx": print(synth_sfx(a.kind, a.out))
    else: print(json.dumps(analyse(a.file), indent=2))

if __name__ == "__main__":
    main()
