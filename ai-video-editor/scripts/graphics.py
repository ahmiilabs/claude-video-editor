"""Motion-graphics overlays drawn on top of a scene frame (or the whole timeline for the progress bar).

scene["overlays"] = [
  {"type": "lower_third", "text": "Name", "subtext": "Title", "at": 0.5, "duration": 3, "y": 0.72},     # every overlay accepts "y" (0-1 of height)
  {"type": "scene_title", "text": "Chapter 1", "at": 0, "duration": 2},
  {"type": "stat", "value": 2400000, "label": "Views", "prefix": "", "suffix": "", "at": 0.3, "duration": 3},
  {"type": "chart", "kind": "bar", "data": {"Jan": 20, "Feb": 45}, "title": "Growth", "at": 0.3, "duration": 4},
  {"type": "glass", "text": "Key point", "position": "bottom", "at": 0.5, "duration": 3}
]
spec["progress_bar"] = {"enabled": true, "position": "top", "color": "#FF6B35"}
"""
import math
from PIL import Image, ImageDraw, ImageFilter
from common import font, hex2rgb, ease_out, ease_in_out, clamp, wrap_text

def _window(t, o, default_dur=3.0):
    """-> (progress_in, progress_out, visible) for an overlay's lifetime."""
    at = float(o.get("at", 0.3)); dur = float(o.get("duration", default_dur)); lt = t - at
    if lt < 0 or lt > dur: return 0, 0, False
    return ease_out(lt / 0.4), ease_in_out(clamp((lt - (dur - 0.35)) / 0.35)), True

def _fmt_num(v):
    v = float(v)
    if abs(v) >= 1e9: return f"{v / 1e9:.1f}B"
    if abs(v) >= 1e6: return f"{v / 1e6:.1f}M"
    if abs(v) >= 1e4: return f"{v / 1e3:.0f}K"
    return f"{v:,.0f}" if v == int(v) else f"{v:.1f}"

def _glass(img, box, radius, tint=(255, 255, 255), alpha=70, blur=14):
    x0, y0, x1, y1 = [int(v) for v in box]; x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(img.width, x1), min(img.height, y1)
    if x1 - x0 < 6 or y1 - y0 < 6: return
    reg = img.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(blur))
    reg = Image.blend(reg, Image.new("RGB", reg.size, tint), alpha / 255)
    mask = Image.new("L", reg.size, 0); ImageDraw.Draw(mask).rounded_rectangle([0, 0, reg.width - 1, reg.height - 1], radius=radius, fill=255)
    img.paste(reg, (x0, y0), mask)
    ImageDraw.Draw(img).rounded_rectangle([x0, y0, x1 - 1, y1 - 1], radius=radius, outline=(255, 255, 255), width=2)

def lower_third(img, t, o, accent):
    pin, pout, vis = _window(t, o)
    if not vis: return
    W, H = img.size; d = ImageDraw.Draw(img); k = pin * (1 - pout)
    bh = int(H * 0.075); x = int(W * 0.06 - (1 - k) * W * 0.5); y = int(H * float(o.get("y", 0.72)))
    f1 = font("bold", int(bh * 0.5)); f2 = font("medium", int(bh * 0.3))
    w = int(max(d.textlength(o.get("text", ""), font=f1), d.textlength(o.get("subtext", ""), font=f2)) + bh * 0.9)
    d.rectangle([x, y, x + int(bh * 0.14), y + bh], fill=accent)
    d.rectangle([x + int(bh * 0.14), y, x + w, y + bh], fill=(18, 18, 22))
    d.text((x + bh * 0.4, y + bh * 0.06), o.get("text", ""), font=f1, fill=(255, 255, 255))
    if o.get("subtext"): d.text((x + bh * 0.4, y + bh * 0.58), o["subtext"], font=f2, fill=(200, 200, 205))

def scene_title(img, t, o, accent):
    pin, pout, vis = _window(t, o, 2.0)
    if not vis: return
    W, H = img.size; d = ImageDraw.Draw(img); k = pin * (1 - pout); txt = o.get("text", "").upper()
    f = font("display", int(W * 0.07)); tw = d.textlength(txt, font=f); y = int(H * float(o.get("y", 0.07)) - (1 - k) * 30)
    d.rounded_rectangle([W / 2 - tw / 2 - 22, y - 8, W / 2 + tw / 2 + 22, y + f.size * 1.25], radius=12, fill=accent)
    d.text((W / 2 - tw / 2, y), txt, font=f, fill=(15, 15, 15))

def stat(img, t, o, accent):
    pin, pout, vis = _window(t, o)
    if not vis: return
    W, H = img.size; k = pin * (1 - pout); lt = t - float(o.get("at", 0.3)); prog = ease_out(clamp(lt / 1.4))
    val = float(o.get("value", 0)) * prog; txt = f"{o.get('prefix', '')}{_fmt_num(val)}{o.get('suffix', '')}"
    bw, bh = int(W * 0.5), int(H * 0.14); x = int(W / 2 - bw / 2); y = int(H * float(o.get("y", 0.3)) + (1 - k) * 40)
    _glass(img, (x, y, x + bw, y + bh), int(bh * 0.2), alpha=int(70 * k))
    d = ImageDraw.Draw(img); f = font("bold", int(bh * 0.5)); tw = d.textlength(txt, font=f)
    d.text((W / 2 - tw / 2, y + bh * 0.08), txt, font=f, fill=(255, 255, 255))
    fl = font("medium", int(bh * 0.2)); lw = d.textlength(o.get("label", ""), font=fl); d.text((W / 2 - lw / 2, y + bh * 0.72), o.get("label", ""), font=fl, fill=(235, 235, 240))

def chart(img, t, o, accent):
    pin, pout, vis = _window(t, o, 4.0)
    if not vis: return
    W, H = img.size; k = pin * (1 - pout); data = list((o.get("data") or {}).items())
    if not data: return
    bw_, bh_ = int(W * 0.84), int(H * 0.30); x = int((W - bw_) / 2); y = int(H * float(o.get("y", 0.22)) + (1 - k) * 40)
    _glass(img, (x, y, x + bw_, y + bh_), 24, tint=(20, 20, 26), alpha=150)
    d = ImageDraw.Draw(img); d.text((x + 24, y + 14), o.get("title", ""), font=font("bold", int(W * 0.045)), fill=(255, 255, 255))
    mx = max(float(v) for _, v in data) or 1; area = (x + 28, y + bh_ * 0.28, x + bw_ - 28, y + bh_ - 38); n = len(data); slot = (area[2] - area[0]) / n
    lt = t - float(o.get("at", 0.3)); kind = o.get("kind", "bar"); pts = []
    for i, (lab, v) in enumerate(data):
        g = ease_out((lt - 0.2 - i * 0.12) / 0.6); hh = (area[3] - area[1]) * (float(v) / mx) * g; x0 = area[0] + i * slot + slot * 0.18; x1 = area[0] + (i + 1) * slot - slot * 0.18
        if kind == "bar": d.rounded_rectangle([x0, area[3] - hh, x1, area[3]], radius=6, fill=accent)
        else: pts.append(((x0 + x1) / 2, area[3] - hh))
        fl = font("medium", int(W * 0.03)); lw = d.textlength(str(lab), font=fl); d.text(((x0 + x1) / 2 - lw / 2, area[3] + 6), str(lab), font=fl, fill=(205, 205, 212))
    if kind != "bar" and len(pts) > 1:
        d.line(pts, fill=accent, width=5, joint="curve")
        for p in pts: d.ellipse([p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7], fill=(255, 255, 255))

def glass_text(img, t, o, accent):
    pin, pout, vis = _window(t, o)
    if not vis: return
    W, H = img.size; k = pin * (1 - pout); d = ImageDraw.Draw(img); f = font("bold", int(W * 0.055))
    lines = wrap_text(d, o.get("text", ""), f, int(W * 0.7)); lh = f.size * 1.25; bh = int(lh * len(lines) + 40); bw_ = int(W * 0.8)
    pos = o.get("position", "bottom"); y = int(H * float(o.get("y", 0.62 if pos == "bottom" else 0.10 if pos == "top" else 0.42)) + (1 - k) * 30)
    _glass(img, ((W - bw_) / 2, y, (W + bw_) / 2, y + bh), 28, alpha=int(80 * k))
    for i, ln in enumerate(lines):
        tw = d.textlength(ln, font=f); d.text((W / 2 - tw / 2, y + 20 + i * lh), ln, font=f, fill=(255, 255, 255))

OVERLAYS = {"lower_third": lower_third, "scene_title": scene_title, "stat": stat, "chart": chart, "glass": glass_text}

def draw_overlays(img, t, overlays, accent_hex):
    acc = hex2rgb(accent_hex)
    for o in overlays or []:
        fn = OVERLAYS.get(o.get("type"))
        if not fn: raise ValueError(f"unknown overlay '{o.get('type')}'. Available: {', '.join(OVERLAYS)}")
        fn(img, t, o, acc)
    return img

def progress_bar(img, T, total, cfg):
    d = ImageDraw.Draw(img); W, H = img.size; h = max(4, int(H * 0.006)); col = hex2rgb(cfg.get("color", "#FF6B35"))
    y = int(H * 0.02) if cfg.get("position", "top") == "top" else H - h - int(H * 0.02)
    d.rectangle([0, y, W, y + h], fill=(255, 255, 255)); d.rectangle([0, y, int(W * clamp(T / max(total, 0.01))), y + h], fill=col)
    return img
