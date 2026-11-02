#!/usr/bin/env python3
"""Unified production pipeline: TOPIC -> script -> assets (images/videos) -> voice -> transcription/captions -> render (transitions, audio, mux) -> QA -> post effects.

  python scripts/pipeline.py --topic "Why is chai expensive on trains" --language roman-urdu --duration 30 --out reel.mp4 --workdir runs/chai
  python scripts/pipeline.py --spec my_spec.json --out reel.mp4                      # skip the writer: manual spec
  python scripts/pipeline.py --spec my_spec.json --skip images,videos --post-effects "color-grade --look warm"
  python scripts/pipeline.py --spec my_spec.json --only voice,render

STAGES (in order)  script | images | videos | voice | transcribe | render | qa | effects
Every stage has a contract (input -> output), its own timing/log entry in <workdir>/run.json and never raises past the runner:
  - optional stages (images, videos, transcribe, effects) that fail are logged and the pipeline continues with the documented fallback
    (style background / Ken-Burns / estimated or audio-aligned captions / un-effected video). --strict makes them fatal.
  - mandatory stages (script when no --spec, voice, render) abort the run. QA failure => exit code 1 (the file is still kept).
Manual alternatives: your own script (--spec), your own images/videos (scene.image / scene.video), your own voice (voice.file).
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import config
from common import get_logger, find_binary, resolve_spec_paths

log = get_logger("aive.pipeline")
STAGES = ["script", "images", "videos", "voice", "transcribe", "render", "qa", "effects"]
OPTIONAL = {"images", "videos", "transcribe", "effects"}
HERE = Path(__file__).resolve().parent

class Run:
    def __init__(self, workdir):
        self.dir = Path(workdir).resolve(); self.dir.mkdir(parents=True, exist_ok=True); self.file = self.dir / "run.json"
        self.data = json.loads(self.file.read_text(encoding="utf-8")) if self.file.exists() else {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "stages": {}}
        self.logf = open(self.dir / "run.log", "a", encoding="utf-8")
    def note(self, msg):
        line = config.redact(f"[{time.strftime('%H:%M:%S')}] {msg}"); self.logf.write(line + "\n"); self.logf.flush(); log.info(config.redact(msg))
    def stage(self, name, status, seconds=0, outputs=None, error=None, detail=None):
        self.data["stages"][name] = {"status": status, "seconds": round(seconds, 1), "outputs": outputs or [], "error": config.redact(error) if error else None, "detail": detail}
        self.file.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")

def stage_script(ctx):
    """in: topic/brief or spec path -> out: spec dict (+ spec.json)."""
    if ctx["spec_path"]:
        spec = json.load(open(ctx["spec_path"], encoding="utf-8")); resolve_spec_paths(spec, Path(ctx["spec_path"]).resolve().parent); return spec, "manual spec loaded", [str(ctx["spec_path"])]
    import scriptwriter as SW
    spec = SW.generate(ctx["topic"], ctx["language"], ctx["duration"], ctx["platform"], ctx["tone"], ctx["category"], ctx["audience"])
    p = ctx["run"].dir / "spec.json"; p.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"); return spec, f"script generated ({len(spec['scenes'])} scenes)", [str(p)]

def stage_images(ctx):
    import imagegen as IG
    spec = ctx["spec"]
    if not any(s.get("image_prompt") or s.get("image") for s in spec["scenes"]): return "no image prompts or images in the spec: nothing to do", []
    rep = IG.generate_scene_images(spec, ctx["run"].dir / "assets")
    if rep["failed"] and not (rep["generated"] or rep["user"] or rep["reused"]): raise RuntimeError("; ".join(f"scene {f['scene']}: {f['error']}" for f in rep["failed"][:3]))
    return f"generated={rep['generated']} user={rep['user']} reused={rep['reused']} failed={[f['scene'] for f in rep['failed']]}", [str(ctx["run"].dir / "assets" / "manifest.json")]

def stage_videos(ctx):
    import videogen as VG
    spec = ctx["spec"]
    if not any(s.get("video_prompt") and not s.get("video") for s in spec["scenes"]): return "no video prompts: nothing to do", []
    rep = VG.generate_scene_videos(spec, ctx["run"].dir / "assets", quality=ctx["quality"] or "draft")
    if rep["failed"]: raise RuntimeError(f"video generation failed for scenes {[f['scene'] for f in rep['failed']]}: {rep['failed'][0]['error']}; Ken-Burns fallback scenes: {rep['fallback_kenburns']}")
    return f"generated={rep['generated']}", [str(ctx["run"].dir / "assets" / "jobs.json")]

def stage_voice(ctx):
    """Pre-synthesises the voiceover so the render reuses it (cached). Informational for file/silent voice modes."""
    from voice import build_voice
    spec = ctx["spec"]; wd = ctx["run"].dir / "voice"; durs, wav, mode = build_voice(spec, str(wd))
    ctx["durs"] = durs; return f"voice mode={mode} total={sum(durs):.1f}s", [wav] if wav else []

def stage_transcribe(ctx):
    """Decides caption timing from real audio when possible and stores the transcript. TTS voices keep exact scene alignment (known script)."""
    spec = ctx["spec"]; v = spec.get("voice", {}) or {}; cap = spec.setdefault("captions", {})
    if v.get("file") and cap.get("timing", "auto") in ("auto", "whisper"):
        import transcribe as TX
        tr = TX.transcribe(v["file"], cap.get("language") or v.get("lang", "auto"), cap.get("engine"), cap.get("model"))
        p = ctx["run"].dir / "transcript.json"; p.write_text(json.dumps(tr, indent=2, ensure_ascii=False), encoding="utf-8"); cap["timing"] = "whisper"
        import captions as CP; CP.write_srt(tr["words"], str(ctx["run"].dir / "transcript.srt"))
        return f"transcribed with {tr['engine']} ({tr['granularity']}-level timestamps)", [str(p)]
    return "voice is generated from a known script: captions use audio-aligned scene timing (no transcription needed)", []

def stage_render(ctx):
    import reel
    res = reel.render_reel(ctx["spec"], ctx["out"], work=None, quality=ctx["quality"], qa=False); ctx["render"] = res
    return f"rendered {res['duration']}s {res['size'][0]}x{res['size'][1]}@{res['fps']} (voice={res['voice_mode']}, captions={res['captions_timing']})", [res["out"]]

def stage_qa(ctx):
    from qa_check import run_qa
    rep = run_qa(ctx["out"], report_path=str(ctx["out"]) + ".qa.json", quiet=True); ctx["qa"] = rep
    if not rep["passed"]: raise RuntimeError("QA failed: " + ", ".join(rep["failed_checks"]))
    return f"QA passed ({len(rep['checks'])} checks, warnings: {rep['warnings'] or 'none'})", [str(ctx["out"]) + ".qa.json"]

def stage_effects(ctx):
    if not ctx["post_effects"]: return "no post effects requested", []
    out = Path(ctx["out"]); fx = out.with_name(out.stem + "_fx" + out.suffix)
    cmd = [sys.executable, str(HERE / "effects.py"), str(out), *ctx["post_effects"].split(), "--out", str(fx)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError("effects failed: " + (r.stderr or r.stdout)[-300:])
    ctx["final"] = str(fx); return f"effects applied -> {fx.name}", [str(fx)]

RUNNERS = {"script": stage_script, "images": stage_images, "videos": stage_videos, "voice": stage_voice, "transcribe": stage_transcribe, "render": stage_render, "qa": stage_qa, "effects": stage_effects}

def run_pipeline(ctx, skip=(), only=None, strict=False):
    run = ctx["run"]; run.note(f"pipeline start; stages={'only ' + ','.join(only) if only else 'all'} skip={list(skip)}"); ok = True; failed_qa = False
    for name in STAGES:
        if (only and name not in only) or name in skip: run.stage(name, "skipped"); continue
        if name == "script" and not ctx["spec_path"] and not ctx["topic"] and ctx.get("spec") is None: raise SystemExit("give --topic or --spec")
        t0 = time.time()
        try:
            if name == "script":
                spec, msg, outs = stage_script(ctx); ctx["spec"] = spec
            else:
                if ctx.get("spec") is None:
                    sp = ctx["spec_path"] or (run.dir / "spec.json")
                    if not Path(sp).exists(): raise RuntimeError("no spec available: run the script stage or pass --spec")
                    ctx["spec"] = resolve_spec_paths(json.load(open(sp, encoding="utf-8")), Path(sp).resolve().parent)
                msg, outs = RUNNERS[name](ctx)
            run.stage(name, "ok", time.time() - t0, outs, detail=msg); run.note(f"{name}: {msg}")
            if name in ("script", "images", "videos", "transcribe"): (run.dir / "spec.resolved.json").write_text(json.dumps(ctx["spec"], ensure_ascii=False, indent=2), encoding="utf-8")
        except (KeyboardInterrupt, SystemExit): raise
        except Exception as e:
            msg = config.redact(f"{type(e).__name__}: {e}"); run.stage(name, "failed", time.time() - t0, error=msg); run.note(f"{name} FAILED: {msg}")
            if name == "qa": failed_qa = True; ok = False; continue
            if name in OPTIONAL and not strict: run.note(f"{name}: optional stage failed, continuing with its fallback"); continue
            ok = False; break
    for name in STAGES:                                   # stages after an abort were never reached: say so explicitly
        if name not in run.data["stages"]: run.stage(name, "not_run", detail="not reached because an earlier stage failed")
    run.data["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S"); run.data["ok"] = ok; run.stage("_summary", "ok" if ok else "failed"); return ok

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic"); ap.add_argument("--spec"); ap.add_argument("--out", default="reel.mp4"); ap.add_argument("--workdir", default="pipeline_run")
    ap.add_argument("--language", default="en"); ap.add_argument("--duration", type=int, default=30); ap.add_argument("--platform", default="reels"); ap.add_argument("--tone", default="energetic")
    ap.add_argument("--category", default="education"); ap.add_argument("--audience", default="general audience"); ap.add_argument("--quality", choices=["draft", "hd"])
    ap.add_argument("--skip", default=""); ap.add_argument("--only", default=""); ap.add_argument("--strict", action="store_true"); ap.add_argument("--post-effects", default="", help='effects.py arguments, e.g. "--effect color-grade --look warm"')
    a = ap.parse_args()
    for lst in (a.skip, a.only):
        bad = [s for s in lst.split(",") if s and s not in STAGES]
        if bad: ap.error(f"unknown stage(s) {bad}; choose from {STAGES}")
    run = Run(a.workdir)
    ctx = {"run": run, "spec_path": a.spec, "topic": a.topic, "language": a.language, "duration": a.duration, "platform": a.platform, "tone": a.tone, "category": a.category,
           "audience": a.audience, "quality": a.quality, "out": str(Path(a.out).resolve()), "post_effects": a.post_effects, "spec": None}
    try: ok = run_pipeline(ctx, [s for s in a.skip.split(",") if s], {s for s in a.only.split(",") if s} or None, a.strict)
    except KeyboardInterrupt: print("interrupted", file=sys.stderr); sys.exit(130)
    except SystemExit as e: print(e, file=sys.stderr); sys.exit(2)
    for n, st in run.data["stages"].items():
        if n != "_summary": print(f"{st['status']:8s} {n:11s} {st['seconds']:6.1f}s  {st.get('detail') or st.get('error') or ''}")
    print(("\nOK -> " if ok else "\nFAILED. ") + (ctx.get("final") or ctx["out"]) + f"   (run log: {run.dir / 'run.log'})"); sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
