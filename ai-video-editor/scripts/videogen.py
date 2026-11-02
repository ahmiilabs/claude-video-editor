#!/usr/bin/env python3
"""AI video-clip generation (text-to-video / image-to-video) with asynchronous job tracking, plus a real Ken-Burns fallback.

Providers differ wildly, so the HTTP adapter is DESCRIBED BY CONFIG instead of hard-coded. AIVE_VIDEO_CONFIG = path to a JSON file (or inline JSON):
{
  "submit":   {"method": "POST", "path": "/generate", "body": {"model": "{model}", "prompt": "{prompt}", "duration": "{duration}", "aspect_ratio": "{aspect}", "image": "{image_b64}"},
               "id_path": "id"},
  "status":   {"path": "/jobs/{id}", "state_path": "status", "done": ["completed", "succeeded"], "failed": ["failed", "error"], "result_url_path": "output.url"},
  "limits":   {"durations": [4, 8], "aspects": ["16:9", "9:16"], "needs_image": false},
  "poll_seconds": 5, "timeout_seconds": 600
}
`limits` are enforced BEFORE submitting (duration snapped to the nearest supported value, unsupported aspect -> clear error) because providers do not share capabilities.
Jobs are stored in <assets>/jobs.json (submitted -> polling -> completed|failed -> downloaded|invalid) so an interrupted run can resume.
  python scripts/videogen.py spec.json --assets assets_dir          # fill scene['video'] from scene['video_prompt']
  python scripts/videogen.py --kenburns photo.jpg --seconds 5 --aspect 9:16 --out clip.mp4
"""
import argparse, base64, json, os, sys, time, uuid
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import config
from net import http_json, download, dig, ProviderError
from assets import AssetManager, AssetError, validate_video
from common import get_logger, run, size_for, FramePipe

log = get_logger("aive.video")

class VideoGenError(RuntimeError):
    pass

class JobTracker:
    def __init__(self, root):
        self.path = Path(root) / "jobs.json"; self.path.parent.mkdir(parents=True, exist_ok=True)
        self.jobs = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
    def _save(self):
        tmp = str(self.path) + ".tmp"; Path(tmp).write_text(json.dumps(self.jobs, indent=2), encoding="utf-8"); os.replace(tmp, self.path)
    def create(self, scene, **kw):
        jid = "vj_" + uuid.uuid4().hex[:10]; self.jobs[jid] = {"scene": scene, "status": "queued", "created": time.time(), "updated": time.time(), "attempts": 0, "error": None, **kw}; self._save(); return jid
    def update(self, jid, **kw):
        self.jobs[jid].update(kw, updated=time.time()); self._save()
    def pending(self, scene):
        for jid, j in self.jobs.items():
            if j["scene"] == scene and j["status"] in ("submitted", "polling") and j.get("remote_id"): return jid, j
        return None, None

def _fill(obj, vals):
    if isinstance(obj, str):
        for k, v in vals.items():
            if obj == "{" + k + "}": return v                       # exact placeholder keeps the native type (number, null)
            obj = obj.replace("{" + k + "}", str(v))
        return obj
    if isinstance(obj, dict): return {k: _fill(v, vals) for k, v in obj.items() if _fill(v, vals) is not None}
    if isinstance(obj, list): return [_fill(v, vals) for v in obj]
    return obj

class HTTPAsyncVideo:
    name = "http"
    def __init__(self, p, cfg):
        self.p, self.cfg = p, cfg
        for k in ("submit", "status"):
            if k not in cfg: raise VideoGenError(f"AIVE_VIDEO_CONFIG is missing '{k}'")
        self.lim = cfg.get("limits", {})
    def headers(self): return {"Authorization": f"Bearer {self.p.api_key}"} if self.p.api_key else {}
    def plan(self, duration, aspect, has_image):
        lim = self.lim
        if lim.get("aspects") and aspect not in lim["aspects"]: raise VideoGenError(f"provider does not support aspect {aspect} (supports {lim['aspects']})")
        if lim.get("needs_image") and not has_image: raise VideoGenError("this provider only does image-to-video: the scene needs an image first")
        if lim.get("durations"): duration = min(lim["durations"], key=lambda d: abs(d - duration))
        return duration
    def submit(self, prompt, duration, aspect, image_path=None):
        sc = self.cfg["submit"]; vals = {"model": self.p.model, "prompt": prompt, "duration": duration, "aspect": aspect, "image_b64": base64.b64encode(Path(image_path).read_bytes()).decode() if image_path else None}
        j = http_json(sc.get("method", "POST"), self.p.base_url + sc["path"], self.headers(), _fill(sc.get("body", {}), vals), self.p.timeout, self.p.retries)
        rid = dig(j, sc.get("id_path", "id"))
        if not rid: raise ProviderError("submit response had no job id")
        return str(rid)
    def poll(self, rid):
        st = self.cfg["status"]; j = http_json("GET", self.p.base_url + st["path"].replace("{id}", rid), self.headers(), None, self.p.timeout, self.p.retries)
        state = str(dig(j, st.get("state_path", "status"), "")).lower()
        if state in [x.lower() for x in st.get("failed", ["failed"])]: return "failed", dig(j, st.get("error_path", "error"), "provider reported failure")
        if state in [x.lower() for x in st.get("done", ["completed"])]:
            url = dig(j, st.get("result_url_path", "output.url"))
            return ("done", url) if url else ("failed", "completed but no result URL")
        return "running", None

def get_video_provider():
    p = config.get_provider("video")
    if p.name in ("none", ""): raise VideoGenError("no video provider configured (AIVE_VIDEO_PROVIDER=http + AIVE_VIDEO_CONFIG). Using Ken-Burns animation of images instead.")
    try: p.validate()
    except config.ConfigError as e: raise VideoGenError(str(e))
    raw = os.environ.get("AIVE_VIDEO_CONFIG", "").strip()
    if not raw: raise VideoGenError("AIVE_VIDEO_CONFIG (path or inline JSON describing the provider's endpoints) is not set")
    try: cfg = json.loads(Path(raw).read_text(encoding="utf-8")) if Path(raw).exists() else json.loads(raw)
    except (ValueError, OSError) as e: raise VideoGenError(f"AIVE_VIDEO_CONFIG is not valid JSON: {e}")
    cfg.setdefault("poll_seconds", 5); cfg.setdefault("timeout_seconds", 600)
    return HTTPAsyncVideo(p, cfg)

def fit_to_canvas(src, dst, aspect, seconds, fps=24, quality="draft"):
    """Scale/crop a downloaded clip to the reel canvas, force constant fps, trim to `seconds`, drop audio."""
    w, h = size_for(aspect, quality)
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", f"{seconds:.2f}", "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps}", "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(dst)])
    return dst

def kenburns_clip(image, out, seconds=5.0, aspect="9:16", fps=24, quality="draft", mode="in"):
    """Real fallback: animate a still (slow push/pan) into an mp4. Uses the same engine as scene images in reels."""
    from media import SceneMedia
    w, h = size_for(aspect, quality); m = SceneMedia({"image": str(image), "kenburns": mode, "media_scrim": 0.0}, 0, w, h)
    pipe = FramePipe(str(out), w, h, fps)
    try:
        for k in range(int(seconds * fps)): pipe.write(m.frame(k / fps, seconds))
        pipe.close()
    finally: m.close()
    return str(out)

def generate_scene_videos(spec, assets_dir, provider=None, only=None, quality="draft", poll_override=None):
    """For scenes with video_prompt (and no video): submit all jobs, poll together, download, validate. Failures fall back to the scene image
    (Ken-Burns, handled by the renderer) when one exists. Returns a report; never marks a failed job as success."""
    am = AssetManager(assets_dir); jt = JobTracker(assets_dir); aspect = spec.get("aspect", "9:16"); rep = {"generated": [], "fallback_kenburns": [], "failed": [], "jobs": {}}
    try: prov = provider or get_video_provider(); err = None
    except VideoGenError as e: prov, err = None, str(e)
    active = {}
    for i, s in enumerate(spec["scenes"], 1):
        if only and i not in only: continue
        if s.get("video") or not s.get("video_prompt"): continue
        if prov is None:
            rep["failed"].append({"scene": i, "error": err}); (rep["fallback_kenburns"].append(i) if s.get("image") else None); am.record(i, "video", None, "provider", "failed", error=err); continue
        dur = float(s.get("duration") or 4.0); img = s.get("image")
        try: dur = prov.plan(dur, aspect, bool(img))
        except VideoGenError as e:
            rep["failed"].append({"scene": i, "error": str(e)}); (rep["fallback_kenburns"].append(i) if img else None); am.record(i, "video", None, "provider", "failed", error=str(e)); continue
        jid, job = jt.pending(i)
        try:
            if jid is None:
                jid = jt.create(i, provider=prov.p.name, prompt=s["video_prompt"], duration=dur); rid = prov.submit(s["video_prompt"], dur, aspect, img)
                jt.update(jid, remote_id=rid, status="submitted", attempts=jt.jobs[jid]["attempts"] + 1)
            active[jid] = {"scene": i, "dur": dur}
        except ProviderError as e:
            msg = config.redact(e); jt.update(jid, status="failed", error=msg); rep["failed"].append({"scene": i, "error": msg}); (rep["fallback_kenburns"].append(i) if img else None); am.record(i, "video", None, "provider", "failed", error=msg)
    deadline = time.time() + (prov.cfg["timeout_seconds"] if prov else 0); interval = poll_override if poll_override is not None else (prov.cfg["poll_seconds"] if prov else 0)
    while active and time.time() < deadline:
        for jid in list(active):
            sc = active[jid]; i = sc["scene"]; s = spec["scenes"][i - 1]
            try: state, info = prov.poll(jt.jobs[jid]["remote_id"]); jt.update(jid, status="polling")
            except ProviderError as e: state, info = ("failed", config.redact(e)) if not e.retryable else ("running", None)
            if state == "running": continue
            del active[jid]
            if state == "failed":
                jt.update(jid, status="failed", error=str(info)); rep["failed"].append({"scene": i, "error": str(info)}); (rep["fallback_kenburns"].append(i) if s.get("image") else None); am.record(i, "video", None, "provider", "failed", error=str(info)); continue
            try:
                raw = Path(assets_dir) / f"_dl_{jid}.mp4"; download(info, raw, prov.headers() if info.startswith(prov.p.base_url) else None)
                dst = am.unique_path(i, "video", ".mp4", jid); fit_to_canvas(raw, dst, aspect, sc["dur"], quality=quality); raw.unlink(missing_ok=True)
                meta = validate_video(dst); jt.update(jid, status="downloaded"); am.record(i, "video", dst, "provider", provider=prov.p.name, model=prov.p.model, **meta)
                s["video"] = str(dst); rep["generated"].append(i)
            except (ProviderError, AssetError, RuntimeError) as e:
                msg = config.redact(e); jt.update(jid, status="invalid", error=msg); rep["failed"].append({"scene": i, "error": msg}); (rep["fallback_kenburns"].append(i) if s.get("image") else None); am.record(i, "video", None, "provider", "failed", error=msg)
        if active: time.sleep(interval)
    for jid, sc in active.items():
        msg = "timed out waiting for the provider"; jt.update(jid, status="failed", error=msg); rep["failed"].append({"scene": sc["scene"], "error": msg}); am.record(sc["scene"], "video", None, "provider", "failed", error=msg)
    rep["jobs"] = {j: v["status"] for j, v in jt.jobs.items()}; return rep

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", nargs="?"); ap.add_argument("--assets", default="assets"); ap.add_argument("--only"); ap.add_argument("--out")
    ap.add_argument("--kenburns"); ap.add_argument("--seconds", type=float, default=5); ap.add_argument("--aspect", default="9:16"); ap.add_argument("--mode", default="in")
    a = ap.parse_args()
    if a.kenburns: print(kenburns_clip(a.kenburns, a.out or "kenburns.mp4", a.seconds, a.aspect, mode=a.mode)); return
    if not a.spec: ap.error("spec is required")
    spec = json.load(open(a.spec, encoding="utf-8")); rep = generate_scene_videos(spec, a.assets, only={int(x) for x in a.only.split(",")} if a.only else None)
    Path(a.out or a.spec).write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(rep, indent=2)); sys.exit(1 if rep["failed"] else 0)

if __name__ == "__main__":
    main()
