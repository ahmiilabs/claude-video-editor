"""Computer vision for video effects.

SEGMENTATION  Robust Video Matting (RVM, MobileNetV3, ONNX) -> a soft alpha matte of the person, stable across frames
              (recurrent model). Needs `python scripts/setup_models.py --matting` (15 MB). No GPU required (~0.1-0.3 s/frame on CPU).
              If the model is missing, functions raise VisionError with the install command. Nothing is faked.
TRACKING      Classic OpenCV Haar face detector (NOT deep-learning), made stable with association + smoothing + dead-zone + hold.
              When the matte is available, the subject box is also derived from the matte (works for side views / no clear face).
"""
import math, os
from pathlib import Path
import numpy as np
from common import font, hex2rgb, ease_out, clamp

class VisionError(RuntimeError):
    pass

def _cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise VisionError("OpenCV is required: pip install opencv-python-headless")

# ------------------------------------------------------------------ matting
class Matter:
    """Per-video person matting. Create one instance per clip (it carries recurrent state between frames)."""
    def __init__(self, max_side=640, target=384):
        from setup_models import matting_path, matting_ready
        if not matting_ready():
            raise VisionError("person-segmentation model not installed. Run: python scripts/setup_models.py --matting")
        try:
            import onnxruntime as ort
        except ImportError:
            raise VisionError("onnxruntime is required: pip install onnxruntime")
        so = ort.SessionOptions(); so.intra_op_num_threads = max(1, os.cpu_count() or 1)
        self.sess = ort.InferenceSession(str(matting_path()), so, providers=["CPUExecutionProvider"])
        self.max_side, self.target = max_side, target; self.ratio = None; self.shape = None; self.prev_small = None; self.reset()

    def reset(self):
        self.rec = [np.zeros((1, 1, 1, 1), np.float32) for _ in range(4)]

    def matte(self, rgb):
        """rgb uint8 HxWx3 -> alpha float32 HxW in [0, 1] at the ORIGINAL resolution."""
        cv2 = _cv2(); H, W = rgb.shape[:2]; s = min(1.0, self.max_side / max(H, W))
        w, h = max(32, int(round(W * s / 16)) * 16), max(32, int(round(H * s / 16)) * 16)
        small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        if self.shape != (h, w):                      # new geometry -> new recurrent state & fixed downsample ratio
            self.shape = (h, w); self.ratio = float(min(1.0, self.target / max(h, w))); self.reset(); self.prev_small = None
        g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if self.prev_small is not None and float(np.abs(g - self.prev_small).mean()) > 45: self.reset()   # hard scene cut
        self.prev_small = g
        src = (small.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {"src": src, "r1i": self.rec[0], "r2i": self.rec[1], "r3i": self.rec[2], "r4i": self.rec[3], "downsample_ratio": np.array([self.ratio], np.float32)})
        self.rec = list(out[2:]); a = out[1][0, 0]
        return cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR).clip(0, 1).astype(np.float32)

def refine_alpha(a, erode=1, feather=1.6, contrast=1.15):
    """Smooth mask edges: tiny erosion removes halo, gaussian feather softens, a mild S-curve keeps the body solid."""
    cv2 = _cv2(); out = a
    if erode > 0: out = cv2.erode(out, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    if feather > 0: out = cv2.GaussianBlur(out, (0, 0), feather)
    if contrast != 1: out = np.clip((out - 0.5) * contrast + 0.5, 0, 1)
    return out.astype(np.float32)

def composite(fg_rgb, alpha, bg_rgb):
    a = alpha[..., None]
    return (fg_rgb.astype(np.float32) * a + bg_rgb.astype(np.float32) * (1 - a)).clip(0, 255).astype(np.uint8)

# ------------------------------------------------------------------ backgrounds
class Background:
    """spec: blur[:sigma] | color:#RRGGBB | gradient:#top:#bottom | image:path | video:path"""
    def __init__(self, spec, W, H):
        cv2 = _cv2(); self.W, self.H = W, H; self.kind, _, arg = (spec or "blur:25").partition(":"); self.arg = arg; self.cap = None
        if self.kind == "color": self.static = np.full((H, W, 3), hex2rgb(arg or "#101018"), np.uint8)
        elif self.kind == "gradient":
            c1, c2 = (arg.split(":") + ["#000000"])[:2]; a, b = np.array(hex2rgb(c1), np.float32), np.array(hex2rgb(c2), np.float32)
            t = np.linspace(0, 1, H)[:, None, None]; self.static = (a * (1 - t) + b * t).astype(np.uint8) * np.ones((1, W, 1), np.uint8)
        elif self.kind == "image":
            if not os.path.exists(arg): raise VisionError(f"background image not found: {arg}")
            im = cv2.cvtColor(cv2.imread(arg), cv2.COLOR_BGR2RGB); s = max(W / im.shape[1], H / im.shape[0]); im = cv2.resize(im, (int(im.shape[1] * s) + 1, int(im.shape[0] * s) + 1))
            y0, x0 = (im.shape[0] - H) // 2, (im.shape[1] - W) // 2; self.static = im[y0:y0 + H, x0:x0 + W]
        elif self.kind == "video":
            if not os.path.exists(arg): raise VisionError(f"background video not found: {arg}")
            self.cap = cv2.VideoCapture(arg); self.static = None
        elif self.kind == "blur": self.static = None; self.sigma = float(arg or 25)
        else: raise VisionError(f"unknown background '{spec}'. Use blur[:n] | color:#hex | gradient:#hex:#hex | image:path | video:path")

    def frame(self, fr):
        cv2 = _cv2()
        if self.kind == "blur": k = int(self.sigma) * 2 + 1; return cv2.GaussianBlur(fr, (k, k), self.sigma)
        if self.cap is not None:
            ok, v = self.cap.read()
            if not ok: self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, v = self.cap.read()
            v = cv2.cvtColor(v, cv2.COLOR_BGR2RGB); s = max(self.W / v.shape[1], self.H / v.shape[0]); v = cv2.resize(v, (int(v.shape[1] * s) + 1, int(v.shape[0] * s) + 1))
            y0, x0 = (v.shape[0] - self.H) // 2, (v.shape[1] - self.W) // 2; return v[y0:y0 + self.H, x0:x0 + self.W]
        return self.static

# ------------------------------------------------------------------ tracking
def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b; ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by)); i = ix * iy
    return i / (aw * ah + bw * bh - i + 1e-9)

class SubjectTracker:
    """Stable face boxes over time (multi-face). Haar cascade detector + IoU association + EMA + dead-zone + hold when lost."""
    def __init__(self, every=3, max_faces=4, smooth=0.35, deadzone=0.025, hold=0.8, fps=30):
        cv2 = _cv2(); self.c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.c.empty(): raise VisionError("OpenCV face cascade could not be loaded")
        self.every, self.max_faces, self.smooth, self.dz, self.hold_frames = every, max_faces, smooth, deadzone, int(hold * fps); self.n = 0; self.tracks = []; self.next_id = 1; self.mask_box = None

    def _detect(self, rgb):
        cv2 = _cv2(); g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY); sc = 360 / g.shape[1] if g.shape[1] > 360 else 1.0
        g = cv2.resize(g, None, fx=sc, fy=sc); g = cv2.equalizeHist(g)
        f = self.c.detectMultiScale(g, 1.15, 5, minSize=(int(24), int(24)))
        f = sorted(f, key=lambda r: -r[2] * r[3])[: self.max_faces]
        return [tuple(v / sc for v in r) for r in f]

    def update(self, rgb, alpha=None):
        if alpha is not None: self.mask_box = self.box_from_alpha(alpha)
        if self.n % self.every == 0:
            dets = self._detect(rgb); used = set()
            for tr in self.tracks:
                best, bi = 0.15, None
                for i, d in enumerate(dets):
                    if i in used: continue
                    v = _iou(tr["box"], d)
                    if v > best: best, bi = v, i
                if bi is None: tr["missed"] += self.every
                else:
                    used.add(bi); d = dets[bi]; b = tr["box"]; tr["missed"] = 0
                    if abs((d[0] + d[2] / 2) - (b[0] + b[2] / 2)) > self.dz * rgb.shape[1] or abs((d[1] + d[3] / 2) - (b[1] + b[3] / 2)) > self.dz * rgb.shape[1] or abs(d[2] - b[2]) > 0.08 * b[2]:
                        k = self.smooth; tr["box"] = tuple(b[j] * (1 - k) + d[j] * k for j in range(4))   # only move when the change exceeds the dead-zone
            for i, d in enumerate(dets):
                if i not in used and len(self.tracks) < self.max_faces: self.tracks.append({"id": self.next_id, "box": d, "missed": 0}); self.next_id += 1
            self.tracks = [t for t in self.tracks if t["missed"] <= self.hold_frames]
        self.n += 1; return self.faces()

    def faces(self):
        return [{"id": t["id"], "x": t["box"][0], "y": t["box"][1], "w": t["box"][2], "h": t["box"][3], "cx": t["box"][0] + t["box"][2] / 2, "cy": t["box"][1] + t["box"][3] / 2, "held": t["missed"] > 0} for t in self.tracks]

    @staticmethod
    def box_from_alpha(a, thr=0.5):
        rows = np.nonzero((a > thr).sum(1) > a.shape[1] * 0.01)[0]
        if len(rows) < 8: return None
        top, bot = int(rows[0]), int(rows[-1]); head = a[top: top + max(4, int((bot - top) * 0.3))] > thr; cols = np.nonzero(head.sum(0) > 0)[0]
        if len(cols) == 0: return None
        return {"top": top, "bottom": bot, "cx": float(cols.mean()), "head_h": float(head.shape[0])}

    def primary(self):
        """Best point to centre on: largest face, else matte-derived head, else None."""
        f = self.faces()
        if f: p = max(f, key=lambda t: t["w"]); return {"cx": p["cx"], "cy": p["cy"], "w": p["w"], "h": p["h"], "source": "face"}
        if self.mask_box: m = self.mask_box; return {"cx": m["cx"], "cy": m["top"] + m["head_h"] * 0.55, "w": m["head_h"], "h": m["head_h"], "source": "matte"}
        return None

class Smoother:
    """Critically-damped-ish follower for crop/zoom centres: EMA with a velocity cap, so the camera never jerks."""
    def __init__(self, k=0.12, max_step=0.02): self.k, self.ms, self.v = k, max_step, None
    def __call__(self, target, scale=1.0):
        if self.v is None: self.v = list(target); return tuple(self.v)
        for i in range(len(target)):
            step = (target[i] - self.v[i]) * self.k; cap = self.ms * scale; self.v[i] += max(-cap, min(cap, step))
        return tuple(self.v)

# ------------------------------------------------------------------ text-behind-person
def text_mask(W, H, text, t, size=0.22, y=0.38, fontname="display", reveal=0.7, drift=0.012, tracking=0.0):
    """Anti-aliased text mask (float 0..1, HxW) with a fade/scale-in and a very slow drift. Returns (mask, bbox)."""
    from PIL import Image, ImageDraw
    p = ease_out(clamp(t / max(0.05, reveal))); fs = int(W * size * (0.9 + 0.1 * p)); f = font(fontname, fs)
    lines = text.upper().split("\\n") if "\\n" in text else [text.upper()]
    im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im); lh = fs * 1.02; total_h = lh * len(lines); y0 = H * y - total_h / 2 + math.sin(t * 0.8) * H * drift
    for i, ln in enumerate(lines):
        tw = d.textlength(ln, font=f)
        if tw > W * 0.96:                                   # auto-shrink a line that would overflow
            f2 = font(fontname, int(fs * W * 0.96 / tw)); tw = d.textlength(ln, font=f2); d.text(((W - tw) / 2, y0 + i * lh), ln, font=f2, fill=255)
        else: d.text(((W - tw) / 2, y0 + i * lh), ln, font=f, fill=255)
    return (np.asarray(im, dtype=np.float32) / 255.0) * p
