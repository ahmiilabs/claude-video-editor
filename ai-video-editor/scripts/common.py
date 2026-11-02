"""Shared helpers: paths, fonts, ffmpeg wrappers, easing. Pure stdlib + Pillow + numpy."""
import json, os, shutil, subprocess, sys
from pathlib import Path
if os.name == "nt" and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError, OSError): pass      # stream does not support reconfigure: keep the default encoding

SKILL_DIR = Path(__file__).resolve().parent.parent
FONT_DIR = SKILL_DIR / "assets" / "fonts"

ASPECTS = {"9:16": (720, 1280), "1:1": (720, 720), "16:9": (1280, 720)}

def size_for(aspect="9:16", quality="draft"):
    """draft = 720p (fast, safe on claude.ai web). hd = 1080p (Claude Code / desktop)."""
    w, h = ASPECTS.get(aspect, ASPECTS["9:16"])
    if quality == "hd":
        k = 1.5
        w, h = int(w * k) // 2 * 2, int(h * k) // 2 * 2
    return w, h

def _resolve(cmd):
    cmd = [str(c) for c in cmd]
    if cmd and cmd[0] in ("ffmpeg", "ffprobe"):
        cmd[0] = find_binary(cmd[0], required=True)
    return cmd

def run(cmd, check=True, quiet=True):
    r = subprocess.run(_resolve(cmd), capture_output=quiet, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        sys.stderr.write((r.stderr or "")[-2000:] + "\n")
        raise RuntimeError("command failed: " + " ".join(map(str, cmd))[:300])
    return r

class DependencyError(RuntimeError):
    """A required external tool/library is missing. The message says how to install it."""

INSTALL_HINTS = {
    "ffmpeg": {"windows": "winget install Gyan.FFmpeg   (or: choco install ffmpeg)",
               "darwin": "brew install ffmpeg", "linux": "sudo apt-get install -y ffmpeg",
               "termux": "pkg install ffmpeg"},
    "espeak-ng": {"windows": "winget install eSpeak-NG.eSpeak-NG", "darwin": "brew install espeak-ng",
                  "linux": "sudo apt-get install -y espeak-ng", "termux": "pkg install espeak"},
}

def os_family():
    if os.environ.get("TERMUX_VERSION") or "com.termux" in os.environ.get("PREFIX", ""):
        return "termux"
    return {"win32": "windows", "darwin": "darwin"}.get(sys.platform, "linux")

def find_binary(name, required=False):
    """Locate an executable on PATH (also AIVE_<NAME> override, e.g. AIVE_FFMPEG). Returns path or None."""
    override = os.environ.get("AIVE_" + name.upper().replace("-", "_"))
    if override and Path(override).exists():
        return str(override)
    found = shutil.which(name)
    if not found and os.name == "nt":
        for base in (os.environ.get("ProgramFiles", ""), os.environ.get("LOCALAPPDATA", "")):
            for cand in Path(base).glob(f"**/{name}.exe") if base and Path(base).exists() else []:
                found = str(cand); break
            if found: break
    if not found and required:
        hint = INSTALL_HINTS.get(name, {}).get(os_family(), f"install '{name}' and make sure it is on PATH")
        raise DependencyError(f"'{name}' was not found on PATH. Install it with: {hint}")
    return found

def have(binary):
    return find_binary(binary) is not None

def ffmpeg_filter_path(p):
    """Escape a file path for use inside an ffmpeg filter argument (ass=..., fontsdir=...).
    Windows drive colons and backslashes, quotes and spaces are handled string-wise so the result is identical on every OS for the same input."""
    import re
    s = str(p)
    if not re.match(r"^[A-Za-z]:[\\/]", s) and not s.startswith("\\\\"):      # POSIX / relative path: make it absolute
        s = str(Path(s).resolve())
    s = s.replace("\\", "/").replace("'", "\\'")
    s = re.sub(r"^([A-Za-z]):", r"\1\\:", s)                                  # C:/x -> C\:/x
    return s.replace(":", "\\:") if not re.match(r"^[A-Za-z]\\:", s) else s[:3] + s[3:].replace(":", "\\:")

def resolve_spec_paths(spec, base_dir):
    """Make relative media paths in a spec absolute. A path is looked up in the current folder first, then next to the spec file.
    Needed because specs are often copied (batch jobs) or run from a different folder than the one holding the media."""
    base = Path(base_dir)
    def fix(v):
        if not isinstance(v, str) or not v or v.startswith("synth:") or Path(v).is_absolute(): return v
        for cand in (Path.cwd() / v, base / v):
            if cand.exists(): return str(cand.resolve())
        return v                                                  # not found anywhere: leave as is so validation reports the original path
    for sc in spec.get("scenes", []) if isinstance(spec, dict) else []:
        for k in ("image", "video"):
            if sc.get(k): sc[k] = fix(sc[k])
        if isinstance(sc.get("audio"), dict) and sc["audio"].get("file"): sc["audio"]["file"] = fix(sc["audio"]["file"])
    if isinstance(spec, dict):
        if isinstance(spec.get("voice"), dict) and spec["voice"].get("file"): spec["voice"]["file"] = fix(spec["voice"]["file"])
        if spec.get("bgm"): spec["bgm"] = fix(spec["bgm"])
        if isinstance(spec.get("audio"), dict) and spec["audio"].get("music"): spec["audio"]["music"] = fix(spec["audio"]["music"])
    return spec

def get_logger(name="aive"):
    import logging
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr); h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        lg.addHandler(h); lg.setLevel(os.environ.get("AIVE_LOG_LEVEL", "INFO").upper()); lg.propagate = False
    return lg

class Workspace:
    """Unique temp directory that is ALWAYS cleaned up (normal exit, exception, Ctrl-C, SIGTERM).
    Concurrent renders each get their own, so jobs can never overwrite each other's temp files.
        with Workspace(prefix="reel_") as ws:  ws.path / "frames.mp4"
    keep=True (or AIVE_KEEP_TMP=1) leaves it in place for debugging."""
    def __init__(self, prefix="aive_", base=None, keep=None):
        self.prefix = prefix; self.base = base or os.environ.get("AIVE_TMP") or None
        self.keep = bool(os.environ.get("AIVE_KEEP_TMP")) if keep is None else keep
        self.path = None; self._old = {}
    def __enter__(self):
        import tempfile, signal, atexit
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix, dir=self.base))
        atexit.register(self.cleanup)
        if threading_main():
            for sig in (getattr(signal, "SIGTERM", None), signal.SIGINT):
                if sig is None: continue
                try:
                    self._old[sig] = signal.signal(sig, self._on_signal)
                except (ValueError, OSError):
                    pass
        return self
    def _on_signal(self, signum, frame):
        self.cleanup(); raise KeyboardInterrupt(f"interrupted by signal {signum}")
    def cleanup(self):
        if self.path and not self.keep and self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
    def __exit__(self, *exc):
        import signal
        for sig, old in self._old.items():
            try: signal.signal(sig, old)
            except (ValueError, OSError): pass
        self.cleanup(); return False

def threading_main():
    import threading
    return threading.current_thread() is threading.main_thread()

_enc_cache = {}
def pick_encoder(prefer_gpu=None):
    """Return ffmpeg video-encoder args. GPU encoder only if it really works (1-frame trial); CPU libx264 otherwise.
    AIVE_GPU = auto (default) | on | off."""
    mode = (os.environ.get("AIVE_GPU", "auto") if prefer_gpu is None else ("on" if prefer_gpu else "off")).lower()
    if mode in _enc_cache: return _enc_cache[mode]
    cpu = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    chosen = cpu
    if mode != "off" and find_binary("ffmpeg"):
        cands = {"windows": ["h264_nvenc", "h264_qsv"], "darwin": ["h264_videotoolbox"], "linux": ["h264_nvenc", "h264_qsv"], "termux": []}[os_family()]
        for enc in cands:
            args = {"h264_nvenc": ["-c:v", enc, "-preset", "p4", "-cq", "21"], "h264_qsv": ["-c:v", enc, "-global_quality", "21"],
                    "h264_videotoolbox": ["-c:v", enc, "-b:v", "8M"]}[enc]
            r = subprocess.run([find_binary("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2:r=10",
                                *args, "-pix_fmt", "yuv420p", "-f", "null", "-"], capture_output=True, text=True)
            if r.returncode == 0:
                chosen = args; break
    _enc_cache[mode] = chosen
    return chosen

def ffprobe(path):
    r = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    return json.loads(r.stdout)

def duration_of(path):
    try:
        return float(ffprobe(path)["format"]["duration"])
    except (RuntimeError, KeyError, ValueError, OSError):     # unreadable file => 0.0; callers validate and raise with context
        return 0.0

_font_cache = {}
def font(name, size):
    """name: bold | medium | display (Anton) | comic (Bangers) | hand (Caveat) | mono. Falls back to system fonts."""
    key = (name, int(size))
    if key in _font_cache:
        return _font_cache[key]
    from PIL import ImageFont
    files = {
        "bold": ["Poppins-Bold.ttf"], "medium": ["Poppins-Medium.ttf"],
        "display": ["Anton-Regular.ttf"], "comic": ["Bangers-Regular.ttf"],
        "hand": ["Caveat.ttf"], "mono": ["JetBrainsMono.ttf"],
    }.get(name, ["Poppins-Bold.ttf"])
    fallbacks = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
    for f in [str(FONT_DIR / x) for x in files] + fallbacks:
        try:
            ft = ImageFont.truetype(f, int(size))
            if name in ("hand", "mono") and hasattr(ft, "set_variation_by_axes"):
                try: ft.set_variation_by_axes([700])          # variable font: pick the bold weight
                except (OSError, ValueError, TypeError): pass  # static/unsupported font file: regular weight is fine
            _font_cache[key] = ft
            return ft
        except OSError:
            continue                                          # font file missing/unreadable: try the next candidate
    return ImageFont.load_default()

def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

# ---- easing -------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def ease_out(t):
    t = clamp(t); return 1 - (1 - t) ** 3

def ease_in_out(t):
    t = clamp(t); return 3 * t * t - 2 * t * t * t

def pop(t):
    """overshoot scale 0 -> 1.12 -> 1 (for word pop-in)"""
    t = clamp(t)
    if t < 0.6: return ease_out(t / 0.6) * 1.12
    return 1.12 - 0.12 * ease_in_out((t - 0.6) / 0.4)

def wrap_text(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

class FramePipe:
    """Pipe raw RGB frames into ffmpeg -> mp4 (no audio)."""
    def __init__(self, out, w, h, fps, crf=20, encoder=None):
        enc = encoder or pick_encoder()
        if encoder is None and crf != 20 and enc[:2] == ["-c:v", "libx264"]:
            enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]
        self.p = subprocess.Popen(
            [find_binary("ffmpeg", required=True), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-an", *enc, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
            stdin=subprocess.PIPE)
    def write(self, img):
        self.p.stdin.write(img.convert("RGB").tobytes())
    def close(self):
        self.p.stdin.close(); self.p.wait()
        if self.p.returncode != 0:
            raise RuntimeError("ffmpeg frame pipe failed")
