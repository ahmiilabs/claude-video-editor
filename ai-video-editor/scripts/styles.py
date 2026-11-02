"""Style packs. Each style = function render(t, dur, scene, ctx) -> PIL.Image (RGB).

ctx: dict(W, H, index, total, accent, cache). Add a style by writing one function and
registering it in STYLES. Keep per-frame work light: pre-render backgrounds into ctx["cache"].
"""
import math, random, re, sys, os
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.dirname(__file__))
from common import font, hex2rgb, ease_out, ease_in_out, pop, clamp, wrap_text

_scratch = ImageDraw.Draw(Image.new("RGB", (8, 8)))

def fit_text(text, fname, box_w, box_h, max_size, min_size=22, line_gap=1.18):
    size = max_size
    while size >= min_size:
        f = font(fname, size)
        lines = wrap_text(_scratch, text, f, box_w)
        lh = size * line_gap
        widest = max((_scratch.textlength(l, font=f) for l in lines), default=0)
        if len(lines) * lh <= box_h and widest <= box_w:
            return f, lines, lh
        size -= 4
    f = font(fname, min_size)
    return f, wrap_text(_scratch, text, f, box_w), min_size * line_gap

def is_kw(word, kws):
    w = re.sub(r"\W+", "", word).lower()
    return bool(w) and any(w == re.sub(r"\W+", "", k).lower() for k in kws)

def _cached(ctx, key, fn):
    c = ctx["cache"]
    if key not in c: c[key] = fn()
    return c[key]

def base_bg(ctx, key, builder):
    """Background for a scene: its image/video frame (already scrimmed) if the scene has media, else the style's own drawn background."""
    m = ctx.get("media")
    return m.copy() if m is not None else _cached(ctx, key, builder).copy()

def text_of(scene):
    return scene.get("text") or scene.get("narration") or ""

# ---------------------------------------------------------------- kinetic
def kinetic(t, dur, scene, ctx):
    W, H = ctx["W"], ctx["H"]; acc = hex2rgb(ctx["accent"])
    def bg():
        im = Image.new("RGB", (W, H), (11, 11, 15)); d = ImageDraw.Draw(im)
        for y in range(H):
            k = y / H; d.line([(0, y), (W, y)], fill=(int(11 + 18 * k), int(11 + 6 * k), int(15 + 28 * k)))
        return im
    img = base_bg(ctx, "kin_bg", bg)
    glow = Image.new("RGB", (W, H), (0, 0, 0)); g = ImageDraw.Draw(glow)
    cx = W * (0.5 + 0.25 * math.sin(t * 0.8 + ctx["index"])); cy = H * (0.35 + 0.1 * math.cos(t * 0.6))
    r = W * 0.5
    g.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(int(c * 0.28) for c in acc))
    glow = glow.filter(ImageFilter.GaussianBlur(W * 0.18))
    img = Image.blend(img, Image.composite(glow, img, Image.new("L", (W, H), 255)), 0.0)
    from PIL import ImageChops
    img = ImageChops.add(img, glow)

    text = text_of(scene).upper(); kws = scene.get("keywords", [])
    box_w, box_h = int(W * 0.84), int(H * 0.5)
    f, lines, lh = fit_text(text, "display", box_w, box_h, int(W * 0.16), 40, 1.12)
    words_total = max(1, len(text.split()))
    span = min(dur * 0.7, words_total * 0.38)
    top = H * 0.5 - len(lines) * lh / 2
    idx = 0
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for li, line in enumerate(lines):
        ws = line.split(); total_w = sum(_scratch.textlength(x, font=f) for x in ws) + (len(ws) - 1) * f.size * 0.28
        x = (W - total_w) / 2
        for wd in ws:
            ww = _scratch.textlength(wd, font=f); t0 = span * idx / words_total; idx += 1
            k = pop((t - t0) / 0.28)
            if k > 0.01:
                kw = is_kw(wd, kws)
                pad = int(f.size * 0.12)
                tile = Image.new("RGBA", (int(ww) + pad * 2, int(f.size * 1.35)), (0, 0, 0, 0))
                td = ImageDraw.Draw(tile)
                if kw:
                    td.rounded_rectangle([0, int(f.size * .08), tile.width - 1, tile.height - 1], radius=int(f.size * .2), fill=acc + (255,))
                td.text((pad, int(f.size * 0.08)), wd, font=f, fill=(15, 15, 15, 255) if kw else (255, 255, 255, 255))
                nw, nh = max(1, int(tile.width * k)), max(1, int(tile.height * k))
                tile = tile.resize((nw, nh), Image.LANCZOS)
                cx2 = x + ww / 2; cy2 = top + li * lh + lh / 2
                layer.alpha_composite(tile, (int(cx2 - nw / 2), int(cy2 - nh / 2)))
            x += ww + f.size * 0.28
    img = img.convert("RGBA"); img.alpha_composite(layer)
    return img.convert("RGB")

# ---------------------------------------------------------------- clean card
def clean(t, dur, scene, ctx):
    W, H = ctx["W"], ctx["H"]; acc = hex2rgb(ctx["accent"])
    def bg():
        im = Image.new("RGB", (W, H), (244, 241, 234)); d = ImageDraw.Draw(im)
        for y in range(0, H, 36):
            for x in range(0, W, 36): d.ellipse([x, y, x + 3, y + 3], fill=(208, 203, 190))
        return im
    img = base_bg(ctx, "clean_bg", bg); d = ImageDraw.Draw(img)
    k = ease_out(t / 0.45)
    m = int(W * 0.07); cw = W - 2 * m; ch = int(H * 0.5)
    y0 = int(H * 0.22 + (1 - k) * H * 0.25); x0 = m
    off = int(W * 0.018)
    d.rounded_rectangle([x0 + off, y0 + off, x0 + cw + off, y0 + ch + off], radius=int(W * 0.05), fill=(20, 20, 20))
    d.rounded_rectangle([x0, y0, x0 + cw, y0 + ch], radius=int(W * 0.05), fill=(255, 255, 255), outline=(20, 20, 20), width=max(4, W // 150))
    label = scene.get("label") or f"// part {ctx['index'] + 1}"
    d.text((x0 + W * 0.05, y0 + W * 0.045), label, font=font("mono", int(W * 0.036)), fill=(110, 110, 110))
    bs = int(W * 0.13); bx = x0 + cw - bs - W * 0.04; by = y0 - bs * 0.45
    d.rounded_rectangle([bx + 5, by + 5, bx + bs + 5, by + bs + 5], radius=int(bs * .22), fill=(20, 20, 20))
    d.rounded_rectangle([bx, by, bx + bs, by + bs], radius=int(bs * .22), fill=acc, outline=(20, 20, 20), width=max(3, W // 200))
    num = str(scene.get("badge", ctx["index"] + 1)); fb = font("bold", int(bs * .6))
    tw = d.textlength(num, font=fb); d.text((bx + (bs - tw) / 2, by + bs * 0.1), num, font=fb, fill=(20, 20, 20))
    text = text_of(scene); kws = scene.get("keywords", [])
    f, lines, lh = fit_text(text, "bold", int(cw * 0.86), int(ch * 0.62), int(W * 0.095), 30)
    n_words = max(1, len(text.split())); shown = int(n_words * ease_out(min(1, max(0, (t - 0.3) / max(0.6, dur * 0.55)))) + 0.999)
    cnt = 0; ty = y0 + ch * 0.2
    for li, line in enumerate(lines):
        x = x0 + cw * 0.07
        for wd in line.split():
            cnt += 1
            if cnt > shown: break
            ww = d.textlength(wd, font=f)
            if is_kw(wd, kws):
                d.rounded_rectangle([x - 6, ty + li * lh + 4, x + ww + 6, ty + li * lh + f.size * 1.2], radius=10, fill=acc)
            d.text((x, ty + li * lh), wd, font=f, fill=(20, 20, 20))
            x += ww + f.size * 0.3
    total = ctx["total"]; dots_y = int(H * 0.8); gap = int(W * 0.045); startx = W / 2 - (total - 1) * gap / 2
    for i in range(min(total, 14)):
        r = 7 if i == ctx["index"] else 5
        d.ellipse([startx + i * gap - r, dots_y - r, startx + i * gap + r, dots_y + r], fill=acc if i == ctx["index"] else (190, 185, 172))
    return img

# ---------------------------------------------------------------- blueprint
def blueprint(t, dur, scene, ctx):
    W, H = ctx["W"], ctx["H"]
    def bg():
        im = Image.new("RGB", (W, H), (14, 62, 140)); d = ImageDraw.Draw(im)
        for x in range(0, W, 48): d.line([(x, 0), (x, H)], fill=(26, 82, 166), width=1)
        for y in range(0, H, 48): d.line([(0, y), (W, y)], fill=(26, 82, 166), width=1)
        return im
    img = base_bg(ctx, "bp_bg", bg); d = ImageDraw.Draw(img); white = (235, 244, 255)
    m = int(W * 0.06); p = ease_out(t / 0.8); L = int((W - 2 * m) * p); Hh = int((H - 2 * m) * p)
    d.line([(m, m), (m + L, m)], fill=white, width=3); d.line([(m, m), (m, m + Hh)], fill=white, width=3)
    d.line([(W - m, H - m), (W - m - L, H - m)], fill=white, width=3); d.line([(W - m, H - m), (W - m, H - m - Hh)], fill=white, width=3)
    d.text((m + 14, m + 12), (scene.get("label") or f"FIG.{ctx['index'] + 1:02d}").upper(), font=font("mono", int(W * 0.032)), fill=(150, 195, 255))
    text = text_of(scene); f, lines, lh = fit_text(text, "mono", int(W * 0.8), int(H * 0.28), int(W * 0.075), 24, 1.3)
    chars = int(len(text) * clamp((t - 0.3) / max(0.8, dur * 0.5))); y = H * 0.14; used = 0
    for line in lines:
        seg = line[: max(0, chars - used)]; used += len(line) + 1
        d.text((m + 24, y), seg, font=f, fill=white); y += lh
    if int(t * 2) % 2 == 0 and chars < len(text):
        d.rectangle([m + 24, y - lh * 0.1, m + 24 + f.size * .5, y + lh * .6], fill=white)
    boxes = scene.get("boxes", [])
    n_box = min(5, len(scene.get("boxes", []))); by = H * (0.44 if n_box <= 3 else 0.38); bh = int(H * (0.085 if n_box <= 3 else 0.07))
    for i, b in enumerate(boxes[:5]):
        k = ease_out((t - 0.8 - i * 0.35) / 0.4)
        if k <= 0: continue
        bw = int((W - 2 * m - 48) * k); x = m + 24
        d.rectangle([x, by + i * (bh + 22), x + bw, by + i * (bh + 22) + bh], outline=white, width=3, fill=(18, 78, 170) if k > .98 else None)
        if k > 0.9:
            d.text((x + 18, by + i * (bh + 22) + bh * 0.22), str(b).upper(), font=font("mono", int(bh * 0.4)), fill=white)
    return img

# ---------------------------------------------------------------- comic
def comic(t, dur, scene, ctx):
    W, H = ctx["W"], ctx["H"]; acc = hex2rgb(ctx["accent"])
    def bg():
        im = Image.new("RGB", (W, H), (255, 214, 64)); d = ImageDraw.Draw(im)
        for y in range(0, H, 28):
            for x in range(0, W, 28):
                r = 4 + 5 * (y / H); d.ellipse([x - r + (14 if (y // 28) % 2 else 0), y - r, x + r + (14 if (y // 28) % 2 else 0), y + r], fill=(255, 196, 30))
        return im
    img = base_bg(ctx, "cm_bg", bg); d = ImageDraw.Draw(img)
    cx, cy = W / 2, H * 0.42
    for i in range(18):
        a = (i / 18) * 2 * math.pi + t * 0.15
        d.polygon([(cx, cy), (cx + math.cos(a) * W, cy + math.sin(a) * W), (cx + math.cos(a + 0.12) * W, cy + math.sin(a + 0.12) * W)], fill=(255, 226, 110))
    k = pop(t / 0.35); shake = math.sin(t * 40) * 3 * max(0, 1 - t * 3)
    text = text_of(scene).upper()
    bw, bh = int(W * 0.84), int(H * 0.34)
    f, lines, lh = fit_text(text, "comic", int(bw * 0.8), int(bh * 0.72), int(W * 0.13), 30, 1.05)
    layer = Image.new("RGBA", (bw + 80, bh + 160), (0, 0, 0, 0)); ld = ImageDraw.Draw(layer)
    ox, oy = 40, 20; ow = max(5, W // 120)
    ld.ellipse([ox, oy, ox + bw, oy + bh], fill=(20, 20, 20)); ld.ellipse([ox + ow, oy + ow, ox + bw - ow, oy + bh - ow], fill=(255, 255, 255))
    ld.polygon([(ox + bw * .3, oy + bh * .9), (ox + bw * .22, oy + bh + 110), (ox + bw * .5, oy + bh * .95)], fill=(20, 20, 20))
    ld.polygon([(ox + bw * .31 + ow, oy + bh * .9 - ow), (ox + bw * .24 + ow, oy + bh + 90), (ox + bw * .48, oy + bh * .94 - ow)], fill=(255, 255, 255))
    y = oy + bh / 2 - len(lines) * lh / 2
    for line in lines:
        tw = ld.textlength(line, font=f); ld.text((ox + (bw - tw) / 2, y), line, font=f, fill=(20, 20, 20)); y += lh
    nw, nh = max(1, int(layer.width * k)), max(1, int(layer.height * k))
    layer = layer.resize((nw, nh), Image.LANCZOS)
    img = img.convert("RGBA"); img.alpha_composite(layer, (int((W - nw) / 2 + shake), int(H * 0.16 + (layer.height and 0))))
    sfx = scene.get("sfx")
    if sfx and t > 0.35:
        s = pop((t - 0.35) / 0.3); fs = font("comic", int(W * 0.15 * max(0.05, s))); dd = ImageDraw.Draw(img)
        tw = dd.textlength(sfx.upper(), font=fs)
        dd.text((W * 0.5 - tw / 2 + 4, H * 0.64 + 4), sfx.upper(), font=fs, fill=(20, 20, 20))
        dd.text((W * 0.5 - tw / 2, H * 0.64), sfx.upper(), font=fs, fill=acc)
    return img.convert("RGB")

# ---------------------------------------------------------------- sketch
def sketch(t, dur, scene, ctx):
    W, H = ctx["W"], ctx["H"]; ink = (34, 34, 38)
    def bg():
        rnd = random.Random(7); im = Image.new("RGB", (W, H), (247, 242, 231)); d = ImageDraw.Draw(im)
        for _ in range(int(W * H / 700)):
            x, y = rnd.randrange(W), rnd.randrange(H); v = rnd.randint(228, 240); d.point((x, y), fill=(v, v - 3, v - 12))
        return im
    img = base_bg(ctx, "sk_bg", bg); d = ImageDraw.Draw(img)
    rnd = random.Random(ctx["index"] * 31 + 5); m = int(W * 0.06)
    def wobble(p0, p1, n=24, amp=2.4):
        pts = []
        for i in range(n + 1):
            u = i / n; pts.append((p0[0] + (p1[0] - p0[0]) * u + rnd.uniform(-amp, amp), p0[1] + (p1[1] - p0[1]) * u + rnd.uniform(-amp, amp)))
        return pts
    prog = ease_out(t / 0.9)
    edges = [((m, m), (W - m, m)), ((W - m, m), (W - m, H - m)), ((W - m, H - m), (m, H - m)), ((m, H - m), (m, m))]
    total = prog * 4
    for i, (a, b) in enumerate(edges):
        u = clamp(total - i)
        if u > 0:
            pts = wobble(a, (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u))
            d.line(pts, fill=ink, width=3, joint="curve")
    text = text_of(scene); kws = scene.get("keywords", [])
    f, lines, lh = fit_text(text, "hand", int(W * 0.78), int(H * 0.5), int(W * 0.17), 40, 1.05)
    total_chars = max(1, sum(len(l) for l in lines)); reveal = clamp((t - 0.4) / max(0.8, dur * 0.6)) * total_chars
    y = H * 0.5 - len(lines) * lh / 2; seen = 0
    for line in lines:
        tw = d.textlength(line, font=f); x = (W - tw) / 2
        vis = clamp((reveal - seen) / max(1, len(line))); seen += len(line)
        if vis > 0:
            layer = Image.new("RGBA", (int(tw) + 20, int(lh) + 30), (0, 0, 0, 0)); ImageDraw.Draw(layer).text((8, 4), line, font=f, fill=ink + (255,))
            mask = Image.new("L", layer.size, 0); ImageDraw.Draw(mask).rectangle([0, 0, int(layer.width * vis), layer.height], fill=255)
            layer.putalpha(Image.composite(layer.split()[3], Image.new("L", layer.size, 0), mask))
            img.paste(layer, (int(x - 8), int(y)), layer)
            if vis >= 1:
                xx = x
                for wd in line.split():
                    ww = d.textlength(wd, font=f)
                    if is_kw(wd, kws):
                        u = ease_out((t - 0.4 - dur * 0.35) / 0.5)
                        if u > 0:
                            pts = wobble((xx, y + f.size * 1.02), (xx + ww * u, y + f.size * 1.06), 16, 3)
                            d.line(pts, fill=(214, 60, 50), width=5, joint="curve")
                    xx += ww + d.textlength(" ", font=f)
        y += lh
    return img

STYLES = {"kinetic": kinetic, "clean": clean, "blueprint": blueprint, "comic": comic, "sketch": sketch}
