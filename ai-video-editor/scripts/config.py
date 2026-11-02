"""Configuration: .env + environment variables, provider settings, key validation, secret redaction.

Precedence (highest first): real environment variables > ./.env > <skill>/.env > ~/.config/ai-video-editor/.env
No python-dotenv needed. API keys are NEVER printed: use redact() on anything that may reach logs/reports.
"""
import os, re
from dataclasses import dataclass, field
from pathlib import Path
from common import SKILL_DIR

PLACEHOLDER_RE = re.compile(r"^(your[-_ ].*|changeme|xxx+|<.*>|none|)$", re.I)
SECRET_PATTERNS = [re.compile(p) for p in (r"sk-[A-Za-z0-9_\-]{8,}", r"Bearer\s+[A-Za-z0-9._\-]{8,}", r"(?i)api[_-]?key[\"'=:\s]+[A-Za-z0-9._\-]{8,}")]
_loaded = {"done": False, "secrets": set()}

OUTPUT_PRESETS = {  # name -> (aspect, quality, fps)
    "reels": ("9:16", None, None), "shorts": ("9:16", None, None), "tiktok": ("9:16", None, None),
    "youtube": ("16:9", None, None), "square": ("1:1", None, None),
}

def parse_env_file(path):
    out = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, v = line.split("=", 1); v = v.split(" #")[0].strip().strip('"').strip("'")
            out[k.strip()] = v
    except OSError:
        pass
    return out

def load_env(force=False):
    """Load .env files into os.environ without overriding variables already set."""
    if _loaded["done"] and not force: return
    for f in (Path.home() / ".config" / "ai-video-editor" / ".env", SKILL_DIR / ".env", Path.cwd() / ".env"):
        if f.exists():
            for k, v in parse_env_file(f).items():
                if v and not PLACEHOLDER_RE.match(v) and k not in os.environ:
                    os.environ[k] = v
    _loaded["done"] = True

def _env(name, default=None):
    load_env(); v = os.environ.get(name, default)
    return None if v is not None and PLACEHOLDER_RE.match(str(v)) and "KEY" in name else v

def _register_secret(v):
    if v and len(v) >= 8: _loaded["secrets"].add(v)
    return v

def redact(text):
    """Remove API keys and token-like strings from any text before logging, reporting or raising."""
    text = str(text)
    for sec in _loaded["secrets"]:
        text = text.replace(sec, "***")
    for k, v in os.environ.items():
        if v and len(v) >= 8 and ("KEY" in k or "TOKEN" in k or "SECRET" in k):
            text = text.replace(v, "***")
    for pat in SECRET_PATTERNS:
        text = pat.sub("***", text)
    return text

@dataclass
class Provider:
    kind: str                     # llm | image | video | transcribe
    name: str = "none"
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    model: str = ""
    timeout: float = 60.0
    retries: int = 3
    def configured(self):
        return self.name not in ("none", "") and bool(self.model or self.kind == "transcribe")
    def problems(self):
        out = []
        if self.name in ("none", ""): return ["provider not selected (set AIVE_%s_PROVIDER)" % self.kind.upper()]
        if not self.base_url and self.name != "anthropic": out.append(f"AIVE_{self.kind.upper()}_BASE_URL missing")
        if not self.model and self.kind != "transcribe": out.append(f"AIVE_{self.kind.upper()}_MODEL missing")
        local = any(h in (self.base_url or "") for h in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"))
        if not self.api_key and not local: out.append(f"API key missing (AIVE_{self.kind.upper()}_API_KEY or provider default variable)")
        return out
    def validate(self):
        p = self.problems()
        if p: raise ConfigError(f"{self.kind} provider '{self.name}' is not usable: " + "; ".join(p))
        return True

class ConfigError(RuntimeError):
    pass

def get_provider(kind):
    kind = kind.lower(); K = kind.upper()
    name = (_env(f"AIVE_{K}_PROVIDER", "none") or "none").lower()
    key = _env(f"AIVE_{K}_API_KEY") or ""
    if not key:
        fallback = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(name)
        key = (os.environ.get(fallback) if fallback else "") or ""
        if PLACEHOLDER_RE.match(key): key = ""
    default_base = {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com/v1"}.get(name, "")
    return Provider(kind=kind, name=name, base_url=(_env(f"AIVE_{K}_BASE_URL") or default_base).rstrip("/"),
                    api_key=_register_secret(key), model=_env(f"AIVE_{K}_MODEL", "") or "",
                    timeout=float(_env("AIVE_TIMEOUT", "60")), retries=int(_env("AIVE_RETRIES", "3")))

def general():
    load_env()
    return {"quality": os.environ.get("AIVE_QUALITY", "draft"), "preset": os.environ.get("AIVE_PRESET", "reels"),
            "workers": int(os.environ.get("AIVE_WORKERS", "1") or 1), "gpu": os.environ.get("AIVE_GPU", "auto"),
            "model_dir": os.environ.get("AIVE_MODEL_DIR") or "", "tmp": os.environ.get("AIVE_TMP") or ""}

def status():
    """Safe-to-print summary: which providers are configured, never any secret."""
    out = {}
    for k in ("llm", "image", "video"):
        p = get_provider(k)
        out[k] = {"provider": p.name, "model": p.model or None, "configured": p.configured() and not p.problems(), "problems": p.problems() if p.name != "none" else []}
    out["transcribe_engine"] = os.environ.get("AIVE_TRANSCRIBE_ENGINE", "auto")
    return out

if __name__ == "__main__":
    import json; print(json.dumps(status(), indent=2))
