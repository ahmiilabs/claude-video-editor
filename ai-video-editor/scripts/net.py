"""Small HTTP layer shared by every provider adapter: JSON requests with retry/backoff, downloads, multipart, and secret redaction.
stdlib only (urllib) so it works everywhere. Retries 429/5xx/timeouts with exponential backoff; 4xx (bad request, auth) fail immediately.
"""
import json, os, time, urllib.error, urllib.request, uuid
from pathlib import Path
import config

class ProviderError(RuntimeError):
    def __init__(self, msg, status=None, retryable=False):
        super().__init__(config.redact(msg)); self.status = status; self.retryable = retryable

def http_json(method, url, headers=None, payload=None, timeout=60, retries=3, backoff=1.5, raw_body=None, content_type=None):
    hdrs = {"User-Agent": "ai-video-editor/2", **(headers or {})}
    data = raw_body
    if payload is not None:
        data = json.dumps(payload).encode("utf-8"); hdrs.setdefault("Content-Type", "application/json")
    if content_type: hdrs["Content-Type"] = content_type
    last = None
    for attempt in range(max(1, retries + 1)):
        try:
            req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            try: return json.loads(body.decode("utf-8")) if body else {}
            except ValueError: raise ProviderError(f"{method} {_host(url)} returned non-JSON ({len(body)} bytes)", r.status)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]; retryable = e.code in (408, 409, 425, 429) or e.code >= 500
            last = ProviderError(f"HTTP {e.code} from {_host(url)}: {msg}", e.code, retryable)
            if not retryable: raise last
            ra = e.headers.get("Retry-After") if e.headers else None
            delay = float(ra) if ra and ra.replace(".", "").isdigit() else backoff ** attempt
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = ProviderError(f"network error contacting {_host(url)}: {type(e).__name__}: {e}", None, True); delay = backoff ** attempt
        if attempt < retries: time.sleep(min(delay, 20))
    raise last

def _host(url):
    from urllib.parse import urlparse
    return urlparse(url).netloc or url

def download(url, dest, headers=None, timeout=120, retries=2):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True); last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ai-video-editor/2", **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(str(dest) + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk: break
                    f.write(chunk)
            os.replace(str(dest) + ".part", dest); return dest
        except Exception as e:
            last = ProviderError(f"download failed from {_host(url)}: {type(e).__name__}: {e}", None, True)
            try: os.remove(str(dest) + ".part")
            except OSError: pass
            time.sleep(1.5 ** attempt)
    raise last

def multipart(fields, files):
    b = "----aive" + uuid.uuid4().hex; body = b""
    for k, v in fields: body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, (fn, data, ct) in files.items():
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{fn}"\r\nContent-Type: {ct}\r\n\r\n'.encode() + data + b"\r\n"
    return body + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"

def dig(obj, path, default=None):
    """dig({'a': {'b': [ {'c': 1} ]}}, 'a.b.0.c') -> 1"""
    cur = obj
    for part in str(path).split("."):
        if part == "": continue
        try: cur = cur[int(part)] if isinstance(cur, list) else cur[part]
        except (KeyError, IndexError, TypeError, ValueError): return default
    return cur
