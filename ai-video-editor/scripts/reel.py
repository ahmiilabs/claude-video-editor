#!/usr/bin/env python3
"""Text -> Reel pipeline. One spec in, one mp4 out.   (CLI unchanged; new capabilities are opt-in via spec fields.)

  python scripts/reel.py spec.json --out reel.mp4
  python scripts/reel.py spec.json --preview             # contact sheet only (fast; check layout before a full render)

Stages: voice -> (optional audio enhance) -> timeline render [styles + media + overlays + transitions] -> captions -> audio mix -> mux -> QA.
Python API:  from reel import render_reel;  result = render_reel(spec_dict, "out.mp4")
Concurrency-safe: every render uses its own temp workspace and writes the output atomically (batch.py relies on this).
"""
import argparse, json, os, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from common import (resolve_spec_paths, size_for, FramePipe, run, FONT_DIR, duration_of, Workspace, get_logger, pick_encoder, ffmpeg_filter_path, find_binary)
from voice import build_voice
from styles import STYLES
import config

log = get_logger("aive.reel")

class SpecError(ValueError):
    pass

def load_spec(path):
    try: spec = json.load(open(path, encoding="utf-8"))
    except FileNotFoundError: raise SpecError(f"spec file not found: {path}")
    except json.JSONDecodeError as e: raise SpecError(f"spec is not valid JSON ({path}): {e}")
    if isinstance(spec, dict): resolve_spec_paths(spec, Path(path).resolve().parent)
    return validate_spec(spec)

def validate_spec(spec):
    if not isinstance(spec, dict): raise SpecError("spec must be a JSON object")
    if not spec.get("scenes") or not isinstance(spec["scenes"], list): raise SpecError("spec needs a non-empty 'scenes' list")
    for i, s in enumerate(spec["scenes"]):
        if not isinstance(s, dict) or not (s.get("text") or s.get("narration")):
            raise SpecError(f"scene {i + 1} has neither 'text' nor 'narration'")
        st = s.get("style") or spec.get("style", "kinetic")
        if st not in STYLES: raise SpecError(f"scene {i + 1}: unknown style '{st}'. Available: {', '.join(STYLES)}")
        for k in ("image", "video"):
            if s.get(k) and not os.path.exists(s[k]): raise SpecError(f"scene {i + 1}: {k} file not found: {s[k]}")
        if s.get("duration") is not None and float(s["duration"]) <= 0: raise SpecError(f"scene {i + 1}: duration must be > 0")
    if spec.get("aspect", "9:16") not in ("9:16", "1:1", "16:9"): raise SpecError("aspect must be 9:16, 1:1 or 16:9")
    return spec

def bgm_path(spec):
    b = spec.get("bgm")
    return b if b and os.path.exists(b) else None

def style_for(spec, scene):
    return STYLES[scene.get("style") or spec.get("style", "kinetic")]

def assets_used(spec):
    out = [s.get(k) for s in spec["scenes"] for k in ("image", "video") if s.get(k)]
    for k in ("bgm",):
        if spec.get(k): out.append(spec[k])
    if (spec.get("voice") or {}).get("file"): out.append(spec["voice"]["file"])
    m = (spec.get("audio") or {}).get("music")
    if m and not str(m).startswith("synth:"): out.append(m)
    return [str(x) for x in out]

# ---------------------------------------------------------------- timeline render
def render_timeline(spec, durs, w, h, fps, out_video, progress=True):
    import transitions as TR, graphics as GR
    from media import SceneMedia
    scenes = spec["scenes"]; n = len(scenes); accent = spec.get("accent", "#FF6B35")
    starts = [sum(durs[:i]) for i in range(n)]; total = sum(durs); N = max(1, int(round(total * fps)))
    tcfg = spec.get("transitions"); legacy_fade = tcfg is None
    trans = []                                    # per boundary: (fn, td, name) or None
    for i in range(n - 1):
        trans.append(TR.resolve(tcfg, scenes[i], i, n, durs[i], durs[i + 1]) if tcfg is not None or scenes[i].get("transition") else None)
    medias, cache = {}, {}
    def media_of(i):
        if i not in medias:
            medias[i] = SceneMedia(scenes[i], i, w, h) if (scenes[i].get("image") or scenes[i].get("video")) else None
        return medias[i]
    def scene_frame(i, t):
        s = scenes[i]; m = media_of(i)
        ctx = {"W": w, "H": h, "index": i, "total": n, "accent": s.get("accent", accent), "cache": cache}
        if m is not None: ctx["media"] = m.frame(max(0.0, t), durs[i])
        img = style_for(spec, s)(max(0.0, t), durs[i], s, ctx)
        if s.get("overlays"): GR.draw_overlays(img, max(0.0, t), s["overlays"], s.get("accent", accent))
        return img
    pipe = FramePipe(out_video, w, h, fps); t0 = time.time(); scene_i = 0; last_report = -1
    try:
        for k in range(N):
            T = k / fps
            while scene_i < n - 1 and T >= starts[scene_i] + durs[scene_i] - 1e-9: scene_i += 1
            frame = None
            for j, tr in enumerate(trans):
                if tr is None: continue
                fn, td, _ = tr; b = starts[j] + durs[j]; a0 = b - td / 2
                if a0 <= T < b + td / 2:
                    frame = fn(scene_frame(j, T - starts[j]), scene_frame(j + 1, T - starts[j + 1]), (T - a0) / td); break
            if frame is None:
                frame = scene_frame(scene_i, T - starts[scene_i])
                if legacy_fade and k > 0 and scene_i > 0 and (T - starts[scene_i]) < 4 / fps:
                    from PIL import Image
                    frame = Image.blend(Image.new("RGB", frame.size, (0, 0, 0)), frame, 0.5 + 0.125 * int((T - starts[scene_i]) * fps))
            pb = spec.get("progress_bar")
            if pb and pb.get("enabled", True): GR.progress_bar(frame, T, total, pb)
            pipe.write(frame)
            if progress and scene_i != last_report:
                last_report = scene_i; print(f"[render] scene {scene_i + 1}/{n}  ({k}/{N} frames, {time.time() - t0:.0f}s)", flush=True)
        pipe.close()
    except BaseException:
        try: pipe.p.kill()
        except (OSError, ProcessLookupError): pass            # encoder already exited
        raise
    finally:
        for m in medias.values():
            if m is not None: m.close()
    return {"starts": starts, "total": total, "frames": N, "transitions": [(t[2], round(t[1], 2)) if t else None for t in trans]}

def verify_captions(plain_video, final_video, windows, zone=(0.55, 0.88), max_samples=8):
    """Evidence that captions are really burned in: compare the caption-less render with the final file inside the caption zone at caption times.
    Returns the fraction of sampled caption windows where the zone differs visibly (None if it could not be measured)."""
    try:
        import cv2
    except ImportError:
        return None
    a, b = cv2.VideoCapture(str(plain_video)), cv2.VideoCapture(str(final_video))
    if not a.isOpened() or not b.isOpened(): return None
    picks = windows[:: max(1, len(windows) // max_samples)][:max_samples]; hits = n = 0
    for w0, w1 in picks:
        t = (w0 + w1) / 2; frames = []
        for cap in (a, b):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, fr = cap.read(); frames.append(fr if ok else None)
        if frames[0] is None or frames[1] is None: continue
        H = frames[0].shape[0]; z0, z1 = int(H * zone[0]), int(H * zone[1])
        d = cv2.absdiff(cv2.resize(frames[0], (frames[1].shape[1], frames[1].shape[0]))[z0:z1], frames[1][z0:z1]).max(axis=2); n += 1
        if (d > 60).mean() > 0.004: hits += 1
    a.release(); b.release()
    return round(hits / n, 2) if n else None

def preview(spec, w, h, out_png):
    from PIL import Image
    from media import SceneMedia
    scenes = spec["scenes"]; accent = spec.get("accent", "#FF6B35"); cache = {}; tiles = []
    for i, s in enumerate(scenes):
        d = float(s.get("duration", 3.0)); ctx = {"W": w, "H": h, "index": i, "total": len(scenes), "accent": s.get("accent", accent), "cache": cache}
        if s.get("image") or s.get("video"):
            m = SceneMedia(s, i, w, h); ctx["media"] = m.frame(d * 0.5, d); m.close()
        img = style_for(spec, s)(d * 0.85, d, s, ctx)
        if s.get("overlays"):
            import graphics; graphics.draw_overlays(img, d * 0.85, s["overlays"], s.get("accent", accent))
        tiles.append(img.resize((w // 2, h // 2)))
    cols = min(4, len(tiles)); rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tiles[0].width, rows * tiles[0].height), (30, 30, 30))
    for k, tl in enumerate(tiles): sheet.paste(tl, ((k % cols) * tl.width, (k // cols) * tl.height))
    sheet.save(out_png); print(out_png); return str(out_png)

# ---------------------------------------------------------------- captions
def caption_words(spec, durs, wav, vmode, scenes):
    """Word list with timing. Mode (spec.captions.timing): auto | tts | whisper | estimate.  Returns (words, mode_used, note)."""
    import captions as CP
    cfg = spec.get("captions", {}) or {}; want = cfg.get("timing", "auto")
    if want == "auto": want = "tts" if vmode in ("neural", "espeak") and wav else ("whisper" if vmode == "file" and wav else "estimate")
    if want == "tts" and wav:
        try:
            import transcribe as TX
            words = TX.align_scenes(wav, scenes, durs, [s.get("narration") or "" for s in scenes])
            if words: return words, "tts", "scene windows exact; words placed inside the real voiced audio"
        except Exception as e:
            log.warning("audio-aligned captions failed (%s); falling back to estimated timing", config.redact(e))
    if want == "whisper" and wav:
        try:
            import transcribe as TX
            tr = TX.transcribe(wav, cfg.get("language") or (spec.get("voice") or {}).get("lang", "auto"), cfg.get("engine"), cfg.get("model"))
            for wd in tr["words"]: wd["scene"] = next((i for i in range(len(durs)) if sum(durs[:i]) <= wd["start"] < sum(durs[:i + 1])), len(durs) - 1)
            return tr["words"], f"whisper:{tr['engine']}:{tr['granularity']}", tr["note"]
        except Exception as e:
            log.warning("whisper captions failed (%s); falling back to estimated timing", config.redact(e))
    return CP.estimate_words(scenes, durs), "estimate", "timing estimated from word length (not from the audio)"

def estimate_seconds(spec, quality="draft", fps=24):
    """Rough wall-clock estimate on ONE CPU core (measured: draft ~0.075 s/frame incl. mux, hd ~0.17; neural voice ~0.9 s per spoken second when not cached)."""
    words = sum(len((s.get("narration") or s.get("text") or "").split()) for s in spec["scenes"]); speech = max(words / 2.4, 1.6 * len(spec["scenes"]))
    dur = sum(float(s.get("duration") or 0) or max(1.6, len((s.get("narration") or s.get("text") or "").split()) / 2.4 + 0.35) for s in spec["scenes"])
    voice_mode = (spec.get("voice") or {}).get("mode", "auto"); voice = speech * 0.9 if voice_mode in ("auto", "neural") else 0.0
    render = dur * fps * (0.075 if quality == "draft" else 0.17) * (24 / fps if quality == "draft" else 1)
    return {"video_seconds": round(dur, 1), "voice_s": round(voice), "render_s": round(render), "total_s": round(voice + render + 8)}

# ---------------------------------------------------------------- main API
def render_reel(spec, out, work=None, quality=None, fps=None, captions=True, qa=True, progress=True):
    """Render `spec` to `out`. Returns a result dict (also written to <out>.render.json). Raises on failure (after recording it)."""
    import captions as CP, audio as AU
    spec = validate_spec(spec); out = Path(out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    side_path = Path(str(out) + ".render.json"); t_start = time.time()
    gen = config.general(); quality = quality or spec.get("quality") or gen["quality"] or ("draft" if os.path.exists("/mnt/user-data") else "hd")
    try:
        import psutil
        free = psutil.virtual_memory().available / 1e6
        if quality == "hd" and free < 1500: log.warning("only %.0f MB RAM free: rendering at draft quality", free); quality = "draft"
    except ImportError: pass
    w, h = size_for(spec.get("aspect", "9:16"), quality); fps = fps or int(spec.get("fps", 24 if quality == "draft" else 30)); scenes = spec["scenes"]
    est = estimate_seconds(spec, quality, fps); log.info("estimate: ~%ds total (voice ~%ds + render ~%ds) for ~%.0fs of video on one CPU core", est["total_s"], est["voice_s"], est["render_s"], est["video_seconds"])
    if os.path.exists("/mnt/user-data") and est["total_s"] > 240: log.warning("this may exceed a single claude.ai command's time limit (~%ds estimated). Split it into shorter reels (<=35 s each) or use fewer scenes.", est["total_s"])
    sidecar = {"status": "running", "spec_title": spec.get("title"), "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "aspect": spec.get("aspect", "9:16"), "size": [w, h], "fps": fps}
    ws = Workspace("reel_", keep=bool(work)) if not work else None
    wd = Path(work) if work else None
    try:
        with (ws if ws else _Null()) as _:
            workdir = str(ws.path if ws else wd); os.makedirs(workdir, exist_ok=True)
            durs, wav, vmode = build_voice(spec, workdir)
            log.info("voice=%s total=%.1fs scenes=%d size=%dx%d@%d", vmode, sum(durs), len(durs), w, h, fps)
            if sum(durs) > 75 and quality == "draft": log.warning(">75s at draft quality may hit sandbox time limits; consider fewer scenes")
            acfg = spec.get("audio", {}) or {}
            if wav and acfg.get("enhance", "auto" if vmode == "file" else False) in (True, "auto") and vmode == "file":
                wav = AU.enhance_voice(wav, os.path.join(workdir, "voice_clean.wav"), denoise=acfg.get("denoise", True))
            silent_vid = os.path.join(workdir, "frames.mp4")
            tl = render_timeline(spec, durs, w, h, fps, silent_vid, progress)
            total = tl["total"]; starts = tl["starts"]
            # captions
            ccfg = spec.get("captions", {}) or {}; use_caps = captions and ccfg.get("enabled", True) and any(s.get("narration") for s in scenes)
            vf = None; cap_info = {"expected": False}
            if use_caps:
                words, tmode, tnote = caption_words(spec, durs, wav, vmode, scenes)
                if words:
                    ass = os.path.join(workdir, "caps.ass")
                    r = CP.write_ass(words, w, h, ass, style=ccfg.get("style", "highlight"), position=ccfg.get("position", "lower"), words_per_chunk=int(ccfg.get("words_per_chunk", 3)))
                    if ccfg.get("srt"): CP.write_srt(words, str(out.with_suffix(".srt")))
                    if ccfg.get("export_ass"): Path(str(out.with_suffix(".ass"))).write_text(Path(ass).read_text(encoding="utf-8"), encoding="utf-8")
                    vf = f"ass={ffmpeg_filter_path(ass)}:fontsdir={ffmpeg_filter_path(FONT_DIR)}"
                    cap_info = {"expected": True, "windows": r["windows"], "dialogues": r["dialogues"], "ass": ass, "timing": tmode, "timing_note": tnote, "zone": [0.55, 0.88]}
            # soundtrack
            audio_file = None; audio_desc = {"voice": bool(wav)}
            music = acfg.get("music") or bgm_path(spec); sfx = []
            if music and str(music).startswith("synth:"): music = AU.synth_music(str(music).split(":", 1)[1] or "calm", total + 1, os.path.join(workdir, "bed.wav"))
            for i, s in enumerate(scenes):
                a = s.get("audio")
                if a and a.get("file"): sfx.append({"file": a["file"], "at": starts[i] + float(a.get("at", 0)), "volume": float(a.get("volume", 0.8))})
            tkind = acfg.get("sfx_on_transitions")
            if tkind:
                sf = AU.synth_sfx(tkind, os.path.join(workdir, f"sfx_{tkind}.wav"))
                for j, t in enumerate(tl["transitions"]):
                    if t: sfx.append({"file": sf, "at": max(0, starts[j] + durs[j] - t[1] / 2), "volume": float(acfg.get("sfx_volume", 0.5))})
            if music or sfx:
                audio_file = os.path.join(workdir, "mix.wav")
                audio_desc = AU.mix(audio_file, total, voice=wav, music=music if music and os.path.exists(str(music)) else None, sfx=sfx, voice_vol=float(acfg.get("voice_volume", 1.0)),
                                    music_vol=float(acfg.get("music_volume", 0.2)), duck=bool(acfg.get("duck", True)), fade_in=float(acfg.get("fade_in", 0.4)),
                                    fade_out=float(acfg.get("fade_out", 1.2)), target_lufs=float(acfg.get("target_lufs", -15)))
            elif wav: audio_file = wav
            # mux (atomic: write to .part then rename)
            part = str(out) + ".part.mp4"; cmd = ["ffmpeg", "-y", "-v", "error", "-i", silent_vid]
            cmd += ["-i", audio_file] if audio_file else ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
            if vf: cmd += ["-vf", vf]
            cmd += ["-map", "0:v", "-map", "1:a", *pick_encoder(), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", part]
            run(cmd); os.replace(part, out)
            if cap_info.get("expected"): cap_info["verified"] = verify_captions(silent_vid, out, cap_info["windows"], tuple(cap_info["zone"]))
            vw = [(starts[i], starts[i] + max(0.5, durs[i] - 0.35)) for i, s in enumerate(scenes) if (s.get("narration") or "").strip()] if wav else []
            sidecar.update({"status": "ok", "duration": round(total, 2), "boundaries": [round(starts[i], 2) for i in range(1, len(starts))], "voice_windows": vw if vmode != "silent" else [],
                            "captions": cap_info, "assets": assets_used(spec), "expect_audio": True, "audio_signal_expected": bool(wav or music or sfx), "voice_mode": vmode,
                            "transitions": tl["transitions"], "audio": audio_desc, "elapsed_s": round(time.time() - t_start, 1)})
            side_path.write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    except BaseException as e:
        sidecar.update({"status": "failed", "error": config.redact(f"{type(e).__name__}: {e}")})
        try: side_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        except OSError: pass
        for junk in (str(out) + ".part.mp4",):
            try: os.remove(junk)
            except OSError: pass
        raise
    log.info("done: %s  %.1fs  %dx%d@%d  (%.0fs)", out, duration_of(out), w, h, fps, time.time() - t_start)
    result = {"out": str(out), "sidecar": str(side_path), "duration": sidecar["duration"], "size": [w, h], "fps": fps, "voice_mode": vmode,
              "captions_timing": cap_info.get("timing"), "audio": audio_desc}
    if qa:
        from qa_check import run_qa
        rep = run_qa(out, report_path=str(out) + ".qa.json"); result["qa_passed"] = rep["passed"]; result["qa_failed"] = rep["failed_checks"]
    return result

class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): return False

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec"); ap.add_argument("--out", default="reel.mp4"); ap.add_argument("--quality", choices=["draft", "hd"], default=None)
    ap.add_argument("--fps", type=int, default=None); ap.add_argument("--work", default=None, help="keep intermediate files in this folder (default: private temp dir, auto-deleted)")
    ap.add_argument("--preview", action="store_true", help="only write a contact-sheet PNG"); ap.add_argument("--no-captions", action="store_true"); ap.add_argument("--no-qa", action="store_true")
    a = ap.parse_args()
    try: spec = load_spec(a.spec)
    except SpecError as e: print(f"SPEC ERROR: {e}", file=sys.stderr); sys.exit(2)
    if a.preview:
        quality = a.quality or spec.get("quality") or ("draft" if os.path.exists("/mnt/user-data") else "hd"); w, h = size_for(spec.get("aspect", "9:16"), quality)
        wd = Path(a.work or Path(tempfile.gettempdir()) / "reel_work"); wd.mkdir(parents=True, exist_ok=True)
        return preview(spec, w, h, wd / "preview.png")
    try: res = render_reel(spec, a.out, a.work, a.quality, a.fps, not a.no_captions, not a.no_qa)
    except KeyboardInterrupt: print("interrupted; temp files cleaned", file=sys.stderr); sys.exit(130)
    except Exception as e: print(f"RENDER FAILED: {config.redact(e)}", file=sys.stderr); sys.exit(1)
    print(json.dumps(res, indent=2))
    if not a.no_qa and not res.get("qa_passed", True): sys.exit(1)

if __name__ == "__main__":
    main()
