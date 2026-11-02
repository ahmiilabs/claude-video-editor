#!/usr/bin/env python3
"""AI image generation for scenes. Provider-independent; ships with an OpenAI-compatible adapter (POST /images/generations, optional /images/edits).

  python scripts/imagegen.py spec.json --assets assets_dir [--only 2,3] [--force]

For every scene that has an `image_prompt` and no `image` file yet: generate -> validate (corrupt / tiny / wrong aspect) -> retry -> save with a
unique name -> set scene["image"]. If the provider fails the scene keeps NO image (the renderer then uses its normal style background); the failure
is recorded in manifest.json and reported. Nothing is ever faked.
Consistency: spec.script_meta.visual_style_anchor (or spec.visual_style) is prepended to every prompt; optional seed; optional reference image.
User-supplied images: just set scene["image"] to your file; they are validated and registered, never regenerated.
"""
import argparse, base64, hashlib, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import config
from net import http_json, download, multipart, dig, ProviderError
from assets import AssetManager, AssetError, validate_image
from common import get_logger

log = get_logger("aive.image")
SIZES = {"9:16": "1024x1792", "16:9": "1792x1024", "1:1": "1024x1024"}   # override: AIVE_IMAGE_SIZES='{"9:16":"768x1344"}'

class ImageError(RuntimeError):
    pass

class OpenAICompatImages:
    name = "openai"
    def __init__(self, p):
        self.p = p; extra = os.environ.get("AIVE_IMAGE_EXTRA", ""); self.extra = json.loads(extra) if extra.strip() else {}
        self.sizes = {**SIZES, **(json.loads(os.environ["AIVE_IMAGE_SIZES"]) if os.environ.get("AIVE_IMAGE_SIZES") else {})}
        self.mode = os.environ.get("AIVE_IMAGE_REFERENCE_MODE", "prompt")      # prompt | edits
    def _h(self): return {"Authorization": f"Bearer {self.p.api_key}"} if self.p.api_key else {}
    def generate(self, prompt, aspect, seed=None, reference=None):
        size = self.sizes.get(aspect, "1024x1024")
        if reference and self.mode == "edits":
            body, ct = multipart([("model", self.p.model), ("prompt", prompt), ("size", size), ("n", "1")], {"image": (Path(reference).name, Path(reference).read_bytes(), "image/png")})
            j = http_json("POST", self.p.base_url + "/images/edits", self._h(), None, self.p.timeout * 2, self.p.retries, raw_body=body, content_type=ct)
        else:
            payload = {"model": self.p.model, "prompt": prompt, "size": size, "n": 1, **self.extra}
            if seed is not None: payload["seed"] = int(seed)
            j = http_json("POST", self.p.base_url + "/images/generations", self._h(), payload, self.p.timeout * 2, self.p.retries)
        item = dig(j, "data.0")
        if not item: raise ProviderError("image response had no data[0]")
        if item.get("b64_json"): return base64.b64decode(item["b64_json"]), ".png"
        if item.get("url"):
            tmp = Path(os.environ.get("AIVE_TMP") or "/tmp") / f"img_dl_{time.time_ns()}"
            try: download(item["url"], tmp, timeout=self.p.timeout * 2); data = tmp.read_bytes()
            finally: tmp.unlink(missing_ok=True)
            return data, ".png"
        raise ProviderError("image response had neither b64_json nor url")

def get_image_provider():
    p = config.get_provider("image")
    if p.name in ("none", ""): raise ImageError("no image provider configured (AIVE_IMAGE_PROVIDER). Supply your own images with scene['image'] instead.")
    try: p.validate()
    except config.ConfigError as e: raise ImageError(str(e))
    if p.name != "openai": raise ImageError(f"unknown image provider '{p.name}' (supported: openai = any OpenAI-compatible images API)")
    return OpenAICompatImages(p)

def build_prompt(spec, scene):
    anchor = (spec.get("visual_style") or (spec.get("script_meta") or {}).get("visual_style_anchor") or "").strip()
    return (f"{anchor}. " if anchor else "") + scene["image_prompt"].strip() + " No text, letters, logos or watermarks in the image."

def generate_scene_images(spec, assets_dir, provider=None, only=None, force=False, sleep=0.0):
    """Fills scene['image'] where possible. Returns {'generated': [...], 'reused': [...], 'user': [...], 'failed': [{'scene','error'}]}."""
    am = AssetManager(assets_dir); aspect = spec.get("aspect", "9:16"); rep = {"generated": [], "reused": [], "user": [], "failed": []}
    seed = int(hashlib.sha1((spec.get("title") or "x").encode()).hexdigest()[:6], 16); first_ok = None
    for i, s in enumerate(spec["scenes"], 1):
        if only and i not in only: continue
        if s.get("image"):
            try: it = am.register_user_asset(i, "image", s["image"]); s["image"] = it["path"]; rep["user"].append(i)
            except AssetError as e: rep["failed"].append({"scene": i, "error": str(e)}); am.record(i, "image", None, "user", "failed", error=str(e)); s.pop("image", None)
            continue
        if not s.get("image_prompt"): continue
        prompt = build_prompt(spec, s); ph = hashlib.sha1(prompt.encode()).hexdigest()[:10]
        hit = None if force else am.find(i, "image", ph)
        if hit: s["image"] = hit["path"]; rep["reused"].append(i); first_ok = first_ok or hit["path"]; continue
        try:
            prov = provider or get_image_provider(); last = None
            for attempt in range(prov.p.retries + 1):
                try:
                    data, ext = prov.generate(prompt, aspect, seed, first_ok if prov.mode == "edits" else None)
                    path = am.unique_path(i, "image", ext, ph); path.write_bytes(data)
                    try: w, h = validate_image(path, aspect=None)
                    except AssetError as ve: path.unlink(missing_ok=True); raise ProviderError(f"provider returned an invalid image: {ve}", None, True)
                    am.record(i, "image", path, "provider", prompt_hash=ph, width=w, height=h, provider=prov.p.name, model=prov.p.model)
                    s["image"] = str(path); rep["generated"].append(i); first_ok = first_ok or str(path); last = None; break
                except ProviderError as e:
                    last = e; log.warning("scene %d image attempt %d failed: %s", i, attempt + 1, e)
                    if not e.retryable: break
                    time.sleep(min(2 ** attempt, 8) if sleep is None else sleep)
            if last: raise last
        except (ProviderError, ImageError) as e:
            msg = config.redact(e); rep["failed"].append({"scene": i, "error": msg}); am.record(i, "image", None, "provider", "failed", ph, msg)
    return rep

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec"); ap.add_argument("--assets", default="assets"); ap.add_argument("--only"); ap.add_argument("--force", action="store_true"); ap.add_argument("--out")
    a = ap.parse_args(); spec = json.load(open(a.spec, encoding="utf-8"))
    rep = generate_scene_images(spec, a.assets, only={int(x) for x in a.only.split(",")} if a.only else None, force=a.force)
    Path(a.out or a.spec).write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2)); sys.exit(1 if rep["failed"] else 0)

if __name__ == "__main__":
    main()
