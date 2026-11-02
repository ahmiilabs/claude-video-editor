"""Captions: word-timed ASS (highlight / karaoke / plain) and SRT export.

A *word list* is the common currency:  [{"w": "Hello", "start": 0.12, "end": 0.48, "approx": False, "scene": 0}, ...]
Producers:  estimate_words()  (legacy, timing guessed from word length)
            transcribe.py     (real speech recognition timestamps)
Consumers:  write_ass() / write_srt().  The legacy build_ass(scenes, durs, ...) is unchanged for old callers.
"""
import re
from pathlib import Path

def _t(sec):
    sec = max(0.0, sec); h = int(sec // 3600); m = int(sec % 3600 // 60); s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def _srt_t(sec):
    sec = max(0.0, sec); ms = int(round(sec * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _clean(w):
    return w.replace("{", "(").replace("}", ")").replace("\\", "")

def hex_to_ass(hexcol):
    h = hexcol.lstrip("#"); r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}&".upper().replace("&H00", "&H00")

def estimate_words(scenes, durs, tail=0.3):
    """Legacy timing: words share each scene's speaking window in proportion to word length. approx=True."""
    out, t0 = [], 0.0
    for i, (s, d) in enumerate(zip(scenes, durs)):
        words = re.findall(r"\S+", s.get("narration") or "")
        if words and s.get("captions", True) is not False:
            speak = max(0.5, d - tail); wt = [max(2, len(x)) for x in words]; tot = sum(wt); acc = 0.0
            for w, k in zip(words, wt):
                out.append({"w": w, "start": t0 + speak * acc / tot, "end": t0 + speak * (acc + k) / tot, "approx": True, "scene": i}); acc += k
        t0 += d
    return out

def _chunks(words, per, max_gap=0.9):
    cur = []
    for w in words:
        if cur and (len(cur) >= per or w["start"] - cur[-1]["end"] > max_gap or w.get("scene") != cur[-1].get("scene")):
            yield cur; cur = []
        cur.append(w)
    if cur: yield cur

def write_ass(words, w, h, path, style="highlight", font="Poppins", accent="&H0000D7FF&", position="lower", words_per_chunk=3,
              uppercase=True, pos_fn=None, size_ratio=0.075):
    """style: highlight (active word coloured) | karaoke (progressive fill, \\k) | plain.
    pos_fn(t) -> (x, y) pixel position for subject-following captions (anchor: bottom-centre)."""
    fs = int(w * size_ratio)
    align, mv = {"lower": (2, int(h * 0.20)), "center": (5, 0), "top": (8, int(h * 0.12))}.get(position, (2, int(h * 0.2)))
    sec = "&H00FFFFFF" if style == "karaoke" else "&H00FFFFFF"; prim = accent.replace("&H00", "&H00") if style == "karaoke" else "&H00FFFFFF"
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Cap,{font},{fs},{prim.rstrip('&')},{sec},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{max(3, fs // 10)},2,{align},{int(w*0.06)},{int(w*0.06)},{mv},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    X = (lambda s: s.upper()) if uppercase else (lambda s: s)
    lines, windows = [], []
    for chunk in _chunks(words, words_per_chunk):
        cs, ce = chunk[0]["start"], chunk[-1]["end"]
        pos = ""
        if pos_fn:
            x, y = pos_fn((cs + ce) / 2); pos = "{\\an2\\pos(%d,%d)}" % (x, y)
        if style == "highlight":
            for k, wk in enumerate(chunk):
                parts = [("{\\c" + accent + "}" + X(_clean(x["w"])) + "{\\c&H00FFFFFF&}") if j == k else X(_clean(x["w"])) for j, x in enumerate(chunk)]
                a = wk["start"]; b = chunk[k + 1]["start"] if k + 1 < len(chunk) else ce
                lines.append(f"Dialogue: 0,{_t(a)},{_t(max(b, a + 0.05))},Cap,,0,0,0,,{pos}{' '.join(parts)}"); windows.append((a, b))
        elif style == "karaoke":
            parts = []
            for j, x in enumerate(chunk):
                nxt = chunk[j + 1]["start"] if j + 1 < len(chunk) else ce
                parts.append("{\\k%d}%s" % (max(1, int(round((nxt - x["start"]) * 100))), X(_clean(x["w"]))))
            lines.append(f"Dialogue: 0,{_t(cs)},{_t(ce + 0.1)},Cap,,0,0,0,,{pos}{' '.join(parts)}"); windows.append((cs, ce))
        else:
            lines.append(f"Dialogue: 0,{_t(cs)},{_t(ce + 0.1)},Cap,,0,0,0,,{pos}{' '.join(X(_clean(x['w'])) for x in chunk)}"); windows.append((cs, ce))
    Path(path).write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(path), "windows": windows, "dialogues": len(lines)}

def write_srt(words, path, words_per_chunk=6, uppercase=False):
    out = []; chunks = list(_chunks(words, words_per_chunk, max_gap=0.7))
    for i, chunk in enumerate(chunks, 1):
        txt = " ".join(x["w"] for x in chunk); txt = txt.upper() if uppercase else txt
        end = chunk[-1]["end"] + 0.05
        if i < len(chunks): end = min(end, chunks[i][0]["start"])        # cues must never overlap
        out.append(f"{i}\n{_srt_t(chunk[0]['start'])} --> {_srt_t(max(end, chunk[0]['start'] + 0.05))}\n{txt}\n")
    Path(path).write_text("\n".join(out), encoding="utf-8")
    return str(path)

def build_ass(scenes, durs, w, h, path, style="highlight", font="Poppins", accent="&H0000D7FF&", position="lower", words_per_chunk=3):
    """Legacy entry point (unchanged signature): estimated timing."""
    write_ass(estimate_words(scenes, durs), w, h, path, style if style in ("highlight", "plain", "karaoke") else "highlight", font, accent, position, words_per_chunk)
    return path
