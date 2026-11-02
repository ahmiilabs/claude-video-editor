"""Scene background media: still image (Ken Burns motion) or video clip, cover-fitted to the canvas and scrimmed for legible text.

scene["image"] / scene["video"] = path. Optional: "media_scrim" (0-0.9, default 0.45), "kenburns": "in"|"out"|"left"|"right"|"up"|"down".
"""
import math, os
import numpy as np
from PIL import Image
from common import clamp, ease_in_out

def _cover(arr, W, H):
    import cv2
    h, w = arr.shape[:2]; s = max(W / w, H / h)
    r = cv2.resize(arr, (max(W, int(w * s + 0.5)), max(H, int(h * s + 0.5))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y0 = (r.shape[0] - H) // 2; x0 = (r.shape[1] - W) // 2
    return r[y0:y0 + H, x0:x0 + W]

class SceneMedia:
    MODES = ["in", "out", "left", "right", "up", "down"]
    def __init__(self, scene, index, W, H):
        import cv2
        self.W, self.H = W, H; self.scrim = float(scene.get("media_scrim", 0.45)); self.kind = None
        self.mode = scene.get("kenburns") or self.MODES[index % len(self.MODES)]
        if scene.get("video"):
            path = scene["video"]
            if not os.path.exists(path): raise FileNotFoundError(f"scene {index + 1}: video not found: {path}")
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened(): raise ValueError(f"scene {index + 1}: cannot open video {path}")
            self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1; self.vfps = self.cap.get(cv2.CAP_PROP_FPS) or 24; self.kind = "video"; self._last = -1; self._frame = None
        elif scene.get("image"):
            path = scene["image"]
            if not os.path.exists(path): raise FileNotFoundError(f"scene {index + 1}: image not found: {path}")
            im = Image.open(path).convert("RGB"); big = (int(W * 1.25), int(H * 1.25))
            self.big = _cover(np.asarray(im), *big); self.kind = "image"
        else:
            raise ValueError("scene has neither image nor video")

    def _scrim(self, arr):
        return (arr.astype(np.float32) * (1 - self.scrim)).astype(np.uint8) if self.scrim > 0 else arr

    def frame(self, t, dur):
        import cv2
        W, H = self.W, self.H
        if self.kind == "video":
            idx = int((t * self.vfps) % self.n)
            if idx != self._last:
                if idx != self._last + 1: self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, fr = self.cap.read()
                if not ok:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, fr = self.cap.read()
                if ok: self._frame = _cover(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), W, H)
                self._last = idx
            return Image.fromarray(self._scrim(self._frame))
        p = ease_in_out(clamp(t / max(0.1, dur)))
        bw, bh = self.big.shape[1], self.big.shape[0]
        z0, z1 = {"in": (1.0, 1.2), "out": (1.2, 1.0)}.get(self.mode, (1.12, 1.12))
        z = z0 + (z1 - z0) * p; cw, ch = bw / (1.25) / z * 1.0, bh / 1.25 / z * 1.0
        cx, cy = bw / 2, bh / 2; span_x, span_y = (bw - cw) / 2, (bh - ch) / 2
        if self.mode == "left":  cx += span_x * (1 - 2 * p)
        if self.mode == "right": cx -= span_x * (1 - 2 * p)
        if self.mode == "up":    cy += span_y * (1 - 2 * p)
        if self.mode == "down":  cy -= span_y * (1 - 2 * p)
        x0 = int(clamp(cx - cw / 2, 0, bw - cw)); y0 = int(clamp(cy - ch / 2, 0, bh - ch))
        crop = self.big[y0:y0 + int(ch), x0:x0 + int(cw)]
        return Image.fromarray(self._scrim(cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)))

    def close(self):
        if self.kind == "video": self.cap.release()
