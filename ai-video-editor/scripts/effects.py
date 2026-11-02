#!/usr/bin/env python3
"""Apply effects to a user's clip. Frame-by-frame (OpenCV + numpy) so it runs on claude.ai web too.

  python scripts/effects.py --list
  python scripts/effects.py clip.mp4 --effect spotlight,color-grade --look warm --out out.mp4
  python scripts/effects.py clip.mp4 --effect glass-widgets --widgets "36|Followers;197K|Reach;4K|Quality"
  python scripts/effects.py clip.mp4 --effect split-chart --data "Jan:20,Feb:45,Mar:80" --title "Growth"
  python scripts/effects.py clip.mp4 --effect punch-zoom,captions --text "Spoken words of the clip"
  python scripts/effects.py clip.mp4 --effect captions --captions-from-audio --caption-style karaoke --caption-follow subject
  python scripts/effects.py clip.mp4 --effect text-behind --title-text "FOCUS" --title-color "#FFD740"
  python scripts/effects.py clip.mp4 --effect bg-replace --bg "gradient:#ff6b35:#1a1a2e"      (also: blur:25, color:#hex, image:p.jpg, video:p.mp4)
  python scripts/effects.py clip.mp4 --effect bg-remove --alpha-format mov --out person.mov    (transparent video)
  python scripts/effects.py clip.mp4 --effect face-crop --aspect 9:16      (must be FIRST in a chain)

Effects chain left-to-right in one pass (captions, if listed, is always applied last via ffmpeg).
Needs: ffmpeg, opencv, numpy, pillow. Matte effects (bg-*, text-behind, smart-spotlight) also need: python scripts/setup_models.py --matting
"""
import argparse, json, math, os, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
import numpy as np
from common import FramePipe, run, font, hex2rgb, ease_out, ease_in_out, clamp, FONT_DIR, duration_of, Workspace, get_logger, find_binary, pick_encoder
import vision as V

CATALOG = {
    "bg-blur":         "[matte] blur the background, keep the person sharp (--bg-sigma 25)",
    "bg-replace":      "[matte] virtual background: --bg blur:N | color:#hex | gradient:#hex:#hex | image:path | video:path",
    "bg-remove":       "[matte] transparent video of the person only (--alpha-format mov|webm|png); must be the only effect",
    "text-behind":     "[matte] big title placed BEHIND the person (--title-text, --title-color, --title-size, --title-y)",
    "smart-spotlight": "[matte] dim only the background using the person matte (--strength)",
    "cinematic-zoom":  "slow eased push-in that follows the subject (--push 1.18)",
    "face-crop":       "subject-following crop to another aspect, e.g. landscape -> 9:16 (--aspect); must be FIRST in a chain",
    "spotlight":     "dim everything except a soft circle that follows the speaker's face",
    "punch-zoom":    "quick zoom pulses on the face every few seconds (--every, --zoom)",
    "color-grade":   "look presets via --look: warm | cool | cinematic | bw | vivid",
    "glass-widgets": "frosted-glass stat cards sliding in (--widgets 'value|label;value|label')",
    "split-chart":   "animated bar chart on top, clip below (--data 'A:10,B:30' --title 'x')",
    "captions":      "word-highlight captions burned in (--text 'narration'); applied last",
}

class FaceTracker:
    """Backward-compatible shim: older code imported FaceTracker().update(frame) -> (cx, cy, w) | None. Now backed by the stable SubjectTracker."""
    def __init__(self): self.t = V.SubjectTracker()
    def update(self, frame_rgb):
        self.t.update(frame_rgb); p = self.t.primary(); return (p["cx"], p["cy"], p["w"]) if p else None

def _cover(img, W, H):
    import cv2
    h, w = img.shape[:2]; s = max(W / w, H / h)
    r = cv2.resize(img, (int(w * s) + 1, int(h * s) + 1)); y0 = (r.shape[0] - H) // 2; x0 = (r.shape[1] - W) // 2
    return r[y0:y0 + H, x0:x0 + W]

# ------------------------------------------------------------- effects: f(frame, t, ctx) -> frame
def fx_spotlight(fr, t, c):
    H, W = fr.shape[:2]; p = c["face"] or (W / 2, H * 0.4, W * 0.3)
    r = max(W, H) * (0.30 + 0.015 * math.sin(t * 2)) 
    yy, xx = np.ogrid[:H, :W]; d = np.sqrt((xx - p[0]) ** 2 + (yy - p[1] * 1.15) ** 2)
    m = np.clip((d - r * 0.55) / (r * 0.75), 0, 1) ** 1.2
    k = 1 - m * c["args"].strength
    return (fr * k[..., None]).astype(np.uint8)

def fx_punch_zoom(fr, t, c):
    import cv2
    H, W = fr.shape[:2]; ev = c["args"].every; ph = t % ev
    z = 1.0 + (c["args"].zoom - 1.0) * (math.sin(math.pi * ph / 0.45) if ph < 0.45 else 0)
    if z <= 1.001: return fr
    p = c["face"] or (W / 2, H / 2, 0); cx, cy = p[0], p[1]
    M = cv2.getRotationMatrix2D((cx, cy), 0, z)
    return cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def fx_color(fr, t, c):
    look = c["args"].look; f = fr.astype(np.float32) / 255
    if look == "warm":   f = f * np.array([1.08, 1.0, 0.88]) + np.array([0.02, 0.0, -0.02])
    elif look == "cool": f = f * np.array([0.9, 1.0, 1.1])
    elif look == "cinematic":
        f = (f - 0.5) * 1.18 + 0.5; f = f * np.array([1.05, 1.0, 0.95]); f[..., 2] += 0.03; f[..., 0] -= 0.01
        g = f.mean(-1, keepdims=True); f = g + (f - g) * 0.85
    elif look == "bw":
        g = f @ np.array([0.299, 0.587, 0.114]); f = np.repeat(g[..., None], 3, -1); f = (f - 0.5) * 1.15 + 0.5
    elif look == "vivid":
        g = f.mean(-1, keepdims=True); f = g + (f - g) * 1.4; f = (f - 0.5) * 1.08 + 0.5
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)

def _round_mask(w, h, r):
    from PIL import Image, ImageDraw
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255); return np.asarray(m) / 255.0

def fx_glass(fr, t, c):
    import cv2
    from PIL import Image, ImageDraw
    H, W = fr.shape[:2]; items = c["widgets"]
    if not items: return fr
    out = fr.copy(); pw, ph = int(W * 0.30), int(H * 0.085); gap = int(H * 0.018)
    for i, (val, lab) in enumerate(items[:6]):
        side_left = (i % 2 == 0); row = i // 2
        k = ease_out((t - 0.4 - i * 0.35) / 0.5)
        if k <= 0: continue
        x_final = int(W * 0.04) if side_left else W - pw - int(W * 0.04)
        x = int(x_final + (-1 if side_left else 1) * (1 - k) * (pw + W * 0.05)); y = int(H * 0.10 + row * (ph + gap) * 1.5 + (0 if side_left else ph * 0.5))
        x0, y0, x1, y1 = max(0, x), y, min(W, x + pw), min(H, y + ph)
        if x1 - x0 < 8 or y1 - y0 < 8: continue
        reg = out[y0:y1, x0:x1]; blur = cv2.GaussianBlur(reg, (0, 0), 14)
        tint = (blur * 0.72 + 255 * 0.28)
        m = _round_mask(x1 - x0, y1 - y0, int(ph * 0.28))[..., None]
        out[y0:y1, x0:x1] = (reg * (1 - m * 0.95) + tint * m * 0.95).astype(np.uint8)
        ov = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        d.rounded_rectangle([0, 0, ov.width - 1, ov.height - 1], radius=int(ph * 0.28), outline=(255, 255, 255, 200), width=2)
        d.text((ph * 0.28, ph * 0.08), str(val), font=font("bold", int(ph * 0.40)), fill=(255, 255, 255, 255))
        d.text((ph * 0.28, ph * 0.64), str(lab), font=font("medium", int(ph * 0.22)), fill=(255, 255, 255, 230))
        base = Image.fromarray(out[y0:y1, x0:x1]).convert("RGBA"); base.alpha_composite(ov); out[y0:y1, x0:x1] = np.asarray(base.convert("RGB"))
    return out

def fx_split(fr, t, c):
    from PIL import Image, ImageDraw
    H, W = fr.shape[:2]; top = int(H * 0.38); acc = hex2rgb(c["args"].accent)
    canvas = Image.new("RGB", (W, H), (14, 14, 18)); d = ImageDraw.Draw(canvas)
    d.text((W * 0.06, H * 0.02), c["args"].title or "", font=font("bold", int(W * 0.06)), fill=(255, 255, 255))
    data = c["data"]; n = max(1, len(data)); mx = max(v for _, v in data) or 1
    area = (W * 0.06, H * 0.075, W * 0.94, top - H * 0.04); bw = (area[2] - area[0]) / n
    for i, (lab, v) in enumerate(data):
        k = ease_out((t - 0.3 - i * 0.2) / 0.7); hh = (area[3] - area[1]) * (v / mx) * k
        x0 = area[0] + i * bw + bw * 0.15; x1 = area[0] + (i + 1) * bw - bw * 0.15
        d.rounded_rectangle([x0, area[3] - hh, x1, area[3]], radius=8, fill=acc)
        f = font("medium", int(W * 0.035)); tw = d.textlength(lab, font=f); d.text(((x0 + x1) / 2 - tw / 2, area[3] + 6), lab, font=f, fill=(190, 190, 200))
        if k > .9:
            vs = f"{v:g}"; f2 = font("bold", int(W * 0.04)); tw = d.textlength(vs, font=f2); d.text(((x0 + x1) / 2 - tw / 2, area[3] - hh - W * 0.055), vs, font=f2, fill=(255, 255, 255))
    clip = _cover(fr, W, H - top); canvas.paste(Image.fromarray(clip), (0, top))
    return np.asarray(canvas)


# ------------------------------------------------------------- matte-based + camera effects (new)
def _alpha(c):
    """Soft person matte for the ORIGINAL frame of this iteration (computed once, shared by every effect in the chain)."""
    if c.get("_alpha") is None:
        if c.get("matter") is None: c["matter"] = V.Matter()
        c["_alpha"] = V.refine_alpha(c["matter"].matte(c["orig"]), feather=c["args"].feather)
    return c["_alpha"]

def fx_bg_blur(fr, t, c):
    a = _alpha(c); bg = V.Background(f"blur:{c['args'].bg_sigma}", fr.shape[1], fr.shape[0]).frame(fr) if c.get("bgobj") is None else c["bgobj"].frame(fr)
    return V.composite(fr, a, bg)

def fx_bg_replace(fr, t, c):
    if c.get("bgobj") is None: c["bgobj"] = V.Background(c["args"].bg, fr.shape[1], fr.shape[0])
    return V.composite(fr, _alpha(c), c["bgobj"].frame(fr))

def fx_text_behind(fr, t, c):
    a = c["args"]; H, W = fr.shape[:2]; al = _alpha(c)
    tm = V.text_mask(W, H, a.title_text or "TEXT", t, size=a.title_size, y=a.title_y)
    vis = ((1 - al) * tm)[..., None]; col = np.array(hex2rgb(a.title_color), np.float32)
    shade = 1.0 - 0.15 * vis                                  # tiny darkening under the letters keeps them readable on bright rooms
    return (fr.astype(np.float32) * shade * (1 - vis) + col * vis).clip(0, 255).astype(np.uint8)

def fx_smart_spotlight(fr, t, c):
    a = _alpha(c); s = c["args"].strength
    dim = (fr.astype(np.float32) * (1 - s)).astype(np.uint8)
    out = V.composite(fr, a, dim).astype(np.float32); out = out * (1 + 0.05 * a[..., None])
    return out.clip(0, 255).astype(np.uint8)

def fx_cinematic_zoom(fr, t, c):
    import cv2
    H, W = fr.shape[:2]; dur = max(0.5, c["duration"]); z = 1.0 + (c["args"].push - 1.0) * ease_in_out(clamp(t / dur))
    p = c.get("subject"); target = (p["cx"], p["cy"]) if p else (W / 2, H / 2)
    cx, cy = c["cam"](target, scale=W)
    M = cv2.getRotationMatrix2D((cx, cy), 0, z)
    M[0, 2] += (W / 2 - cx) * (z - 1) / z * 0.0                # keep zoom centred on the subject
    return cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def fx_face_crop(fr, t, c):
    import cv2
    H, W = fr.shape[:2]; aw, ah = (int(x) for x in c["args"].aspect.split(":")); target = aw / ah
    if abs(W / H - target) < 0.02: return fr
    p = c.get("subject"); cx0 = p["cx"] if p else W / 2; cy0 = p["cy"] if p else H / 2
    cx, cy = c["cam"]((cx0, cy0), scale=W)
    if W / H > target: cw, ch = int(round(H * target)) // 2 * 2, H
    else: cw, ch = W, int(round(W / target)) // 2 * 2
    x0 = int(clamp(cx - cw / 2, 0, W - cw)); y0 = int(clamp(cy - ch * 0.4, 0, H - ch))      # keep the face in the upper third
    return np.ascontiguousarray(fr[y0:y0 + ch, x0:x0 + cw])

EFFECTS = {"spotlight": fx_spotlight, "punch-zoom": fx_punch_zoom, "color-grade": fx_color, "glass-widgets": fx_glass, "split-chart": fx_split,
           "bg-blur": fx_bg_blur, "bg-replace": fx_bg_replace, "text-behind": fx_text_behind, "smart-spotlight": fx_smart_spotlight,
           "cinematic-zoom": fx_cinematic_zoom, "face-crop": fx_face_crop}
MATTE_EFFECTS = {"bg-blur", "bg-replace", "bg-remove", "text-behind", "smart-spotlight"}
NEEDS_SUBJECT = {"spotlight", "punch-zoom", "cinematic-zoom", "face-crop"}

def parse_widgets(s):
    return [tuple((x.split("|") + [""])[:2]) for x in (s or "36|Followers;197K|Reach;4K|Quality").split(";") if x.strip()]

def parse_data(s):
    out = []
    for kv in (s or "Jan:20,Feb:45,Mar:80,Apr:120").split(","):
        if ":" in kv:
            k, v = kv.rsplit(":", 1)
            try: out.append((k.strip(), float(v)))
            except ValueError: raise SystemExit(f"--data: '{kv}' is not label:number")
    if not out: raise SystemExit("--data needs at least one label:number pair, e.g. Jan:20,Feb:45")
    return out

class AlphaPipe:
    """Write RGBA frames: mov (qtrle, widely editable), webm (VP9 + alpha, web), png (numbered PNGs in a folder)."""
    def __init__(self, out, w, h, fps, fmt):
        import subprocess
        self.fmt = fmt; self.n = 0; self.dir = None
        if fmt == "png":
            self.dir = Path(out); self.dir.mkdir(parents=True, exist_ok=True); self.p = None; return
        codec = ["-c:v", "qtrle", "-pix_fmt", "argb"] if fmt == "mov" else ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "2M", "-auto-alt-ref", "0"]
        self.p = subprocess.Popen([find_binary("ffmpeg", required=True), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", *codec, str(out)], stdin=subprocess.PIPE)
    def write(self, rgb, alpha):
        rgba = np.dstack([rgb, (np.clip(alpha, 0, 1) * 255).astype(np.uint8)])
        if self.dir is not None:
            from PIL import Image; Image.fromarray(rgba, "RGBA").save(self.dir / f"frame_{self.n:05d}.png")
        else: self.p.stdin.write(rgba.tobytes())
        self.n += 1
    def close(self):
        if self.p is not None:
            self.p.stdin.close(); self.p.wait()
            if self.p.returncode != 0: raise RuntimeError("alpha export failed (ffmpeg)")

def track_subject_timeline(clip, step=3, use_matte=False):
    """Pre-pass for subject-following captions: [(t, cx, bottom_y, face_h)] smoothed. Uses faces (and the matte if requested)."""
    import cv2
    cap = cv2.VideoCapture(clip); fps = cap.get(cv2.CAP_PROP_FPS) or 30; trk = V.SubjectTracker(fps=fps, every=1); matter = V.Matter() if use_matte else None
    sm = V.Smoother(0.25, 0.04); out = []; i = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        if i % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); H, W = rgb.shape[:2]; al = V.refine_alpha(matter.matte(rgb)) if matter else None; trk.update(rgb, al); p = trk.primary()
            if p:
                cx, cy = sm((p["cx"], p["cy"] + p["h"] * 0.9), scale=W); out.append((i / fps, cx, cy, p["h"], W, H))
        i += 1
    cap.release(); return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", nargs="?"); ap.add_argument("--effect", default="color-grade"); ap.add_argument("--out", default="effects_out.mp4")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--look", default="cinematic"); ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--every", type=float, default=2.5); ap.add_argument("--zoom", type=float, default=1.14)
    ap.add_argument("--widgets"); ap.add_argument("--data"); ap.add_argument("--title", default="")
    ap.add_argument("--text", default=""); ap.add_argument("--accent", default="#FF6B35")
    ap.add_argument("--max-width", type=int, default=720, help="downscale wide clips for speed (0 = keep)")
    ap.add_argument("--bg", default="blur:25"); ap.add_argument("--bg-sigma", type=float, default=25); ap.add_argument("--feather", type=float, default=1.6)
    ap.add_argument("--title-text", default=""); ap.add_argument("--title-color", default="#FFFFFF"); ap.add_argument("--title-size", type=float, default=0.26); ap.add_argument("--title-y", type=float, default=0.38)
    ap.add_argument("--push", type=float, default=1.18); ap.add_argument("--aspect", default="9:16"); ap.add_argument("--alpha-format", default="mov", choices=["mov", "webm", "png"])
    ap.add_argument("--captions-from-audio", action="store_true", help="use real speech recognition for caption timing (needs a transcription engine)")
    ap.add_argument("--caption-style", default="highlight", choices=["highlight", "karaoke", "plain"]); ap.add_argument("--caption-follow", choices=["none", "subject"], default="none")
    ap.add_argument("--lang", default="auto"); ap.add_argument("--srt", help="also write an .srt next to the output")
    ap.add_argument("--start", type=float, default=0.0, help="process only from this second (keeps matte effects fast on claude.ai)"); ap.add_argument("--duration", type=float, default=0.0, help="process only this many seconds")
    a = ap.parse_args()
    if a.list or not a.clip:
        for k, v in CATALOG.items(): print(f"{k:16s} {v}")
        return
    import cv2
    if (a.start or a.duration) and os.path.exists(a.clip):                  # trim first (re-encode so cuts are frame-accurate), then work on the trimmed copy
        trim_ws = Workspace("fxtrim_"); trim_ws.__enter__(); trimmed = os.path.join(str(trim_ws.path), "trim.mp4")
        run(["ffmpeg", "-y", "-v", "error", "-ss", str(a.start), *(["-t", str(a.duration)] if a.duration else []), "-i", a.clip, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "17", "-c:a", "aac", trimmed])
        a.clip = trimmed; import atexit; atexit.register(trim_ws.cleanup)
    names = [x.strip() for x in a.effect.split(",") if x.strip()]
    bad = [n for n in names if n not in CATALOG]
    if bad: raise SystemExit(f"unknown effect(s): {bad}. Use --list")
    if "face-crop" in names and names[0] != "face-crop": raise SystemExit("face-crop must be the first effect in the chain")
    if "bg-remove" in names and len(names) > 1: raise SystemExit("bg-remove must be used alone (it writes an RGBA file)")
    if not os.path.exists(a.clip): raise SystemExit(f"clip not found: {a.clip}")
    cap = cv2.VideoCapture(a.clip)
    if not cap.isOpened(): raise SystemExit("cannot open clip (corrupt or unsupported)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30; W0, H0 = int(cap.get(3)), int(cap.get(4)); nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sc = 1.0 if (a.max_width == 0 or W0 <= a.max_width) else a.max_width / W0
    W, H = int(W0 * sc) // 2 * 2, int(H0 * sc) // 2 * 2
    uses_matte = any(n in MATTE_EFFECTS for n in names)
    if uses_matte:
        from setup_models import matting_ready
        if not matting_ready(): raise SystemExit("this effect needs the segmentation model: python scripts/setup_models.py --matting")
        est = nframes * (0.12 + 0.08 * (W * H) / (720 * 1280)) * 1.5
        if est > 150: print(f"[note] matte effects take roughly {est:.0f}s on this machine ({nframes} frames). Trim the clip or use --max-width 480 to go faster.", file=sys.stderr)
    cap_pos = None
    if "captions" in names and a.caption_follow == "subject":
        cap_pos = track_subject_timeline(a.clip, use_matte=False)
        if not cap_pos: print("[note] no subject found for caption following; using default caption position", file=sys.stderr)
    with Workspace("fx_") as ws:
        tmp = str(ws.path); vid = os.path.join(tmp, "fx.mp4")
        tracker = V.SubjectTracker(fps=fps) if any(n in NEEDS_SUBJECT for n in names) else None
        ctx = {"args": a, "face": None, "subject": None, "widgets": parse_widgets(a.widgets), "data": parse_data(a.data), "duration": nframes / fps, "cam": V.Smoother(0.10, 0.025), "matter": None}
        pipe = None; apipe = None; t_idx = 0; out_wh = (W, H)
        try:
            while True:
                ok, bgr = cap.read()
                if not ok: break
                if sc != 1.0: bgr = cv2.resize(bgr, (W, H))
                fr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); t = t_idx / fps; ctx["orig"] = fr; ctx["_alpha"] = None
                if tracker:
                    tracker.update(fr); p = tracker.primary(); ctx["subject"] = p; ctx["face"] = (p["cx"], p["cy"], p["w"]) if p else None
                if "bg-remove" in names:
                    al = _alpha(ctx)
                    if apipe is None: apipe = AlphaPipe(a.out, W, H, fps, a.alpha_format)
                    apipe.write(fr, al); t_idx += 1; continue
                for n in names:
                    if n in EFFECTS:
                        fr = EFFECTS[n](fr, t, ctx)
                        if n == "face-crop": ctx["orig"] = fr; ctx["_alpha"] = None
                if pipe is None:
                    out_wh = (fr.shape[1] // 2 * 2, fr.shape[0] // 2 * 2); pipe = FramePipe(vid, out_wh[0], out_wh[1], fps, crf=19)
                from PIL import Image
                pipe.write(Image.fromarray(np.ascontiguousarray(fr[: out_wh[1], : out_wh[0]]))); t_idx += 1
            cap.release()
            if t_idx == 0: raise SystemExit("clip has no readable frames")
            if apipe: apipe.close(); print(f"[done] {a.out}  frames={t_idx}  effects={names}  alpha={a.alpha_format}"); return
            pipe.close(); src = vid
            if "captions" in names:
                import captions as CP
                words, gran = [], "estimated"
                if a.captions_from_audio:
                    import transcribe as TX
                    try: tr = TX.transcribe(a.clip, a.lang); words, gran = tr["words"], tr["granularity"]
                    except TX.TranscribeError as e: raise SystemExit(f"--captions-from-audio failed: {e}")
                elif a.text:
                    words = CP.estimate_words([{"narration": a.text}], [t_idx / fps])
                else: raise SystemExit("captions need --text or --captions-from-audio")
                posf = None
                if cap_pos:
                    ts = np.array([p[0] for p in cap_pos]); xs = np.array([p[1] for p in cap_pos]); ys = np.array([p[2] for p in cap_pos]); sw = out_wh[0] / cap_pos[0][4]
                    posf = lambda tt: (float(np.interp(tt, ts, xs)) * sw, float(np.clip(np.interp(tt, ts, ys) * sw + out_wh[1] * 0.06, out_wh[1] * 0.2, out_wh[1] * 0.92)))
                ass = os.path.join(tmp, "c.ass"); CP.write_ass(words, out_wh[0], out_wh[1], ass, style=a.caption_style, pos_fn=posf)
                if a.srt: CP.write_srt(words, a.srt)
                capped = os.path.join(tmp, "cap.mp4")
                from common import ffmpeg_filter_path
                run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-vf", f"ass={ffmpeg_filter_path(ass)}:fontsdir={ffmpeg_filter_path(FONT_DIR)}", *pick_encoder(), "-pix_fmt", "yuv420p", capped])
                src = capped; print(f"[captions] timing={gran} words={len(words)}")
            has_audio = any(s["codec_type"] == "audio" for s in __import__("common").ffprobe(a.clip)["streams"])
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", src] + (["-i", a.clip, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"] if has_audio else []) + ["-c:v", "copy", "-movflags", "+faststart", a.out]
            run(cmd)
        finally:
            cap.release()
    print(f"[done] {a.out}  frames={t_idx}  effects={names}")

if __name__ == "__main__":
    main()
