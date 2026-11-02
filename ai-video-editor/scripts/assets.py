"""Asset manager: unique filenames, scene association, validation, manifest.

<assets_dir>/manifest.json records every image/video: scene, kind, path, source (provider | user | fallback), status (ok | failed), size,
sha256, dimensions, prompt hash, error. A failed generation is recorded as failed and NEVER pointed at a file that does not exist.
"""
import hashlib, json, os, shutil, time
from pathlib import Path
from common import ffprobe, get_logger

log = get_logger("aive.assets")

class AssetError(RuntimeError):
    pass

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def validate_image(path, min_side=256, aspect=None, tol=0.12):
    """-> (width, height). Raises AssetError for missing/corrupt/too-small/wrong-aspect files."""
    from PIL import Image
    p = Path(path)
    if not p.exists() or p.stat().st_size < 2048: raise AssetError(f"image missing or too small: {path}")
    try:
        with Image.open(p) as im: im.verify()
        with Image.open(p) as im: w, h = im.size; im.load()
    except Exception as e: raise AssetError(f"image is corrupt: {path} ({type(e).__name__})")
    if min(w, h) < min_side: raise AssetError(f"image too small: {w}x{h}")
    if aspect:
        aw, ah = (int(x) for x in aspect.split(":"))
        if abs(w / h - aw / ah) / (aw / ah) > tol: raise AssetError(f"image aspect {w}x{h} does not match {aspect}")
    return w, h

def validate_video(path, min_seconds=0.5):
    """-> dict(width, height, duration, codec). Raises AssetError if not a decodable video."""
    p = Path(path)
    if not p.exists() or p.stat().st_size < 4096: raise AssetError(f"video missing or too small: {path}")
    try: info = ffprobe(p)
    except Exception: raise AssetError(f"video is not readable by ffprobe: {path}")
    vs = [s for s in info["streams"] if s["codec_type"] == "video"]
    if not vs: raise AssetError(f"no video stream in {path}")
    dur = float(info["format"].get("duration") or 0)
    if dur < min_seconds: raise AssetError(f"video too short ({dur:.2f}s): {path}")
    from common import run
    r = run(["ffmpeg", "-v", "error", "-i", str(p), "-t", "2", "-f", "null", "-"], check=False)
    if r.returncode != 0 or (r.stderr or "").strip(): raise AssetError(f"video does not decode cleanly: {(r.stderr or '')[:120]}")
    return {"width": int(vs[0]["width"]), "height": int(vs[0]["height"]), "duration": round(dur, 2), "codec": vs[0].get("codec_name")}

class AssetManager:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True); self.mpath = self.root / "manifest.json"
        self.items = json.loads(self.mpath.read_text(encoding="utf-8")) if self.mpath.exists() else []

    def _save(self):
        tmp = str(self.mpath) + ".tmp"; Path(tmp).write_text(json.dumps(self.items, indent=2), encoding="utf-8"); os.replace(tmp, self.mpath)

    def unique_path(self, scene, kind, ext, key=""):
        h = hashlib.sha1((f"{scene}|{kind}|{key}|{time.time_ns()}").encode()).hexdigest()[:8]
        return self.root / f"s{int(scene):02d}_{kind}_{h}{ext}"

    def find(self, scene, kind, prompt_hash=None):
        for it in reversed(self.items):
            if it["scene"] == scene and it["kind"] == kind and it["status"] == "ok" and os.path.exists(it["path"]) and (prompt_hash is None or it.get("prompt_hash") == prompt_hash): return it
        return None

    def record(self, scene, kind, path=None, source="provider", status="ok", prompt_hash=None, error=None, **extra):
        it = {"scene": scene, "kind": kind, "path": str(path) if path else None, "source": source, "status": status, "prompt_hash": prompt_hash,
              "error": error, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        if path and status == "ok" and os.path.exists(path): it["bytes"] = os.path.getsize(path); it["sha256"] = sha256_file(path)
        self.items.append(it); self._save(); return it

    def register_user_asset(self, scene, kind, src, aspect=None):
        """Validate and copy a user-supplied file into the asset dir. Raises AssetError if unusable (nothing is recorded as ok)."""
        src = Path(src)
        if not src.exists(): raise AssetError(f"user asset not found: {src}")
        if kind == "image": w, h = validate_image(src, aspect=None); meta = {"width": w, "height": h}
        else: meta = validate_video(src)
        dst = self.unique_path(scene, kind, src.suffix.lower(), key=str(src)); shutil.copy2(src, dst)
        return self.record(scene, kind, dst, source="user", **meta)

    def summary(self):
        out = {"ok": 0, "failed": 0, "fallback": 0, "user": 0}
        for it in self.items:
            if it["status"] == "failed": out["failed"] += 1
            elif it["source"] == "user": out["user"] += 1
            elif it["source"] == "fallback": out["fallback"] += 1
            else: out["ok"] += 1
        return out
