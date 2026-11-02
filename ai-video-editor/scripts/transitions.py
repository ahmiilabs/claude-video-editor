"""Transition engine. A transition is f(A, B, p) -> frame, where A = outgoing frame, B = incoming frame, p in [0, 1].

All functions work on PIL RGB images of identical size and return the same size (no black bars, no resolution drift).
Presets bundle a cycle of transitions + a duration. Use names in the spec:  "transitions": {"preset": "dynamic", "duration": 0.5}
and per scene: "transition": "slide_left" | {"type": "zoom", "duration": 0.6}   (applies between this scene and the next).
"""
import math, random
import numpy as np
from PIL import Image, ImageFilter
from common import ease_in_out, clamp

def _a(img): return np.asarray(img, dtype=np.float32)
def _img(arr): return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def fade(A, B, p):
    return Image.blend(A, B, ease_in_out(p))

def zoom(A, B, p):
    """A pushes in while fading out; B settles from slightly zoomed-in."""
    q = ease_in_out(p); W, H = A.size
    def scaled(im, s):
        w2, h2 = int(W * s), int(H * s); r = im.resize((w2, h2), Image.BILINEAR)
        return r.crop(((w2 - W) // 2, (h2 - H) // 2, (w2 - W) // 2 + W, (h2 - H) // 2 + H))
    return Image.blend(scaled(A, 1 + 0.35 * q), scaled(B, 1.25 - 0.25 * q), q)

def _slide(A, B, p, dx, dy):
    q = ease_in_out(p); W, H = A.size; out = Image.new("RGB", (W, H))
    ox, oy = int(-dx * W * q), int(-dy * H * q)
    out.paste(A, (ox, oy)); out.paste(B, (ox + dx * W, oy + dy * H)); return out
def slide_left(A, B, p):  return _slide(A, B, p, 1, 0)
def slide_right(A, B, p): return _slide(A, B, p, -1, 0)
def slide_up(A, B, p):    return _slide(A, B, p, 0, 1)
def slide_down(A, B, p):  return _slide(A, B, p, 0, -1)

def _wipe(A, B, p, direction, soft=0.12):
    W, H = A.size; q = ease_in_out(p); a, b = _a(A), _a(B)
    if direction in ("left", "right"):
        x = np.linspace(0, 1, W)[None, :]
        pos = x if direction == "right" else 1 - x
        m = np.clip((q * (1 + soft) - pos) / soft, 0, 1) * np.ones((H, 1))
    else:
        y = np.linspace(0, 1, H)[:, None]
        pos = y if direction == "down" else 1 - y
        m = np.clip((q * (1 + soft) - pos) / soft, 0, 1) * np.ones((1, W))
    return _img(a * (1 - m[..., None]) + b * m[..., None])
def wipe_left(A, B, p):  return _wipe(A, B, p, "left")
def wipe_right(A, B, p): return _wipe(A, B, p, "right")
def wipe_up(A, B, p):    return _wipe(A, B, p, "up")
def wipe_down(A, B, p):  return _wipe(A, B, p, "down")

def blur(A, B, p):
    """Both frames blur toward the middle, cross-dissolve at peak blur."""
    q = ease_in_out(p); r = math.sin(math.pi * q) * 22
    base = Image.blend(A, B, q)
    return base.filter(ImageFilter.GaussianBlur(max(0.1, r))) if r > 0.5 else base

def iris(A, B, p):
    """Circular mask reveal from the centre (a mask transition)."""
    W, H = A.size; q = ease_in_out(p); R = math.hypot(W, H) / 2 * q * 1.05
    yy, xx = np.ogrid[:H, :W]; d = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
    m = np.clip((R - d) / 14, 0, 1)[..., None]
    return _img(_a(A) * (1 - m) + _a(B) * m)

def diagonal(A, B, p):
    W, H = A.size; q = ease_in_out(p); yy, xx = np.mgrid[:H, :W]
    t = (xx / W + yy / H) / 2; m = np.clip((q * 1.2 - t) / 0.2, 0, 1)[..., None]
    return _img(_a(A) * (1 - m) + _a(B) * m)

def light_leak(A, B, p):
    """Cross-dissolve with a warm light flare sweeping across at mid-transition."""
    W, H = A.size; q = ease_in_out(p); base = _a(Image.blend(A, B, q))
    strength = math.sin(math.pi * p) ** 1.5
    xs = np.linspace(-0.3, 1.3, W)[None, :]; c = -0.2 + 1.4 * p
    glow = np.exp(-((xs - c) ** 2) / 0.05) * np.ones((H, 1)); glow = glow[..., None]
    warm = np.array([255, 170, 70], dtype=np.float32)[None, None, :]
    out = base + glow * strength * warm * 0.85 + strength * 28
    return _img(out)

def whip(A, B, p):
    """Fast horizontal push with directional motion blur (motion-based)."""
    import cv2
    q = ease_in_out(p); W, H = A.size
    pushed = _a(_slide(A, B, p, 1, 0)); k = int(2 + 70 * math.sin(math.pi * q)) | 1
    if k > 3:
        kern = np.zeros((k, k), np.float32); kern[k // 2, :] = 1.0 / k
        pushed = cv2.filter2D(pushed, -1, kern)
    return _img(pushed)

def glitch(A, B, p):
    """RGB-split + slice jitter around mid-transition."""
    W, H = A.size; base = _a(A if p < 0.5 else B); rnd = random.Random(int(p * 40)); out = base.copy()
    amp = math.sin(math.pi * p)
    for _ in range(int(6 * amp) + 1):
        y0 = rnd.randrange(H); h = rnd.randrange(6, max(8, H // 14)); sh = int(rnd.uniform(-1, 1) * W * 0.08 * amp)
        out[y0:y0 + h] = np.roll(out[y0:y0 + h], sh, axis=1)
    s = int(10 * amp); out[..., 0] = np.roll(out[..., 0], s, axis=1); out[..., 2] = np.roll(out[..., 2], -s, axis=1)
    return _img(out)

TRANSITIONS = {"fade": fade, "crossfade": fade, "zoom": zoom, "slide_left": slide_left, "slide_right": slide_right, "slide_up": slide_up,
               "slide_down": slide_down, "wipe_left": wipe_left, "wipe_right": wipe_right, "wipe_up": wipe_up, "wipe_down": wipe_down,
               "blur": blur, "iris": iris, "diagonal": diagonal, "light_leak": light_leak, "whip": whip, "glitch": glitch}

PRESETS = {  # name -> (duration seconds, cycle of transitions)
    "none":      (0.0,  []),
    "smooth":    (0.45, ["fade"]),
    "clean":     (0.45, ["wipe_left", "wipe_right"]),
    "dynamic":   (0.40, ["whip", "zoom", "slide_left", "iris", "wipe_up"]),
    "cinematic": (0.70, ["light_leak", "blur", "zoom"]),
    "social":    (0.35, ["whip", "glitch", "zoom", "slide_up"]),
}

def resolve(spec_transitions, scene, index, total, dur_a, dur_b):
    """-> (callable, duration) for the boundary after scene `index`, or None for a hard cut. Duration is clamped so it never
    exceeds 40% of either neighbouring scene (keeps scene boundaries and audio sync intact)."""
    cfg = spec_transitions or {}
    if index >= total - 1: return None
    t = scene.get("transition"); name = None; dur = None
    if isinstance(t, dict): name, dur = t.get("type"), t.get("duration")
    elif isinstance(t, str): name = t
    pd, cycle = PRESETS.get(cfg.get("preset", "smooth"), PRESETS["smooth"])
    if name is None:
        if not cycle and not cfg.get("type"): return None
        name = cfg.get("type") or cycle[index % len(cycle)]
    if name in ("none", "cut"): return None
    if name not in TRANSITIONS: raise ValueError(f"unknown transition '{name}'. Available: {', '.join(sorted(TRANSITIONS))}")
    dur = float(dur if dur is not None else cfg.get("duration", pd or 0.45))
    dur = max(0.0, min(dur, 0.4 * min(dur_a, dur_b)))
    return (TRANSITIONS[name], dur, name) if dur >= 0.08 else None
