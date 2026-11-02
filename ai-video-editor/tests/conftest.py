"""Shared fixtures. External AI services are MOCKED here (a tiny local HTTP server): those tests prove our adapters build correct requests,
parse responses, retry, validate and fail honestly. They do NOT prove that any real provider works; that needs a real key (see docs/providers.md)."""
import base64, io, json, os, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
os.environ.setdefault("AIVE_LOG_LEVEL", "WARNING")

def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *map(str, args)], check=True)

@pytest.fixture(scope="session")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    ff("-f", "lavfi", "-i", "testsrc2=s=360x640:d=3:r=24", "-f", "lavfi", "-i", "sine=f=330:d=3", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", d / "clip.mp4")
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (720, 1280), (40, 80, 160)); dr = ImageDraw.Draw(im)
    for i in range(0, 1280, 80): dr.rectangle([0, i, 720, i + 40], fill=(60 + i % 120, 120, 200))
    im.save(d / "photo.png")
    buf = io.BytesIO(); im.save(buf, "PNG"); (d / "photo_b64.txt").write_text(base64.b64encode(buf.getvalue()).decode())
    ff("-f", "lavfi", "-i", "sine=f=220:d=6", "-c:a", "pcm_s16le", d / "voice_tone.wav")
    return d

class MockState:
    def __init__(self): self.reset()
    def reset(self):
        self.requests = []; self.image_fail_first = 0; self.image_calls = 0; self.llm_replies = []; self.llm_calls = 0; self.video_polls = 0; self.video_mode = "ok"; self.image_bad = False

@pytest.fixture()
def mock_server(media):
    st = MockState(); png_b64 = (media / "photo_b64.txt").read_text()
    ff("-f", "lavfi", "-i", "testsrc=s=320x568:d=2:r=24", "-c:v", "libx264", "-pix_fmt", "yuv420p", media / "gen.mp4")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, obj=None, raw=None, ctype="application/json"):
            body = raw if raw is not None else json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def _body(self):
            n = int(self.headers.get("Content-Length", 0)); return self.rfile.read(n) if n else b""
        def do_POST(self):
            raw = self._body(); st.requests.append((self.path, {k.lower(): v for k, v in self.headers.items()}, raw))
            if self.path.endswith("/chat/completions"):
                st.llm_calls += 1; reply = st.llm_replies.pop(0) if st.llm_replies else "{}"
                return self._send(200, {"choices": [{"message": {"content": reply}}]})
            if self.path.endswith("/messages"):
                st.llm_calls += 1; reply = st.llm_replies.pop(0) if st.llm_replies else "{}"
                return self._send(200, {"content": [{"type": "text", "text": reply}]})
            if self.path.endswith("/images/generations") or self.path.endswith("/images/edits"):
                st.image_calls += 1
                if st.image_fail_first == 99: return self._send(401, {"error": "bad key sk-secretsecret123456"})
                if st.image_calls <= st.image_fail_first: return self._send(500, {"error": "temporary"})
                return self._send(200, {"data": [{"b64_json": "AAAA" if st.image_bad else png_b64}]})
            if self.path.endswith("/audio/transcriptions"):
                return self._send(200, {"text": "hello brave world", "words": [{"word": "hello", "start": 0.1, "end": 0.5}, {"word": "brave", "start": 0.6, "end": 1.0}, {"word": "world", "start": 1.1, "end": 1.6}],
                                        "segments": [{"start": 0.1, "end": 1.6, "text": "hello brave world"}]})
            if self.path.endswith("/generate"):
                if st.video_mode == "reject": return self._send(400, {"error": "unsupported"})
                return self._send(200, {"id": "remote-123"})
            self._send(404, {"error": "not found"})
        def do_GET(self):
            st.requests.append((self.path, {k.lower(): v for k, v in self.headers.items()}, b""))
            if self.path.startswith("/jobs/"):
                st.video_polls += 1
                if st.video_mode == "fail": return self._send(200, {"status": "failed", "error": "moderation"})
                if st.video_polls < 2: return self._send(200, {"status": "running"})
                return self._send(200, {"status": "completed", "output": {"url": f"http://127.0.0.1:{self.server.server_port}/files/clip.mp4"}})
            if self.path == "/files/clip.mp4":
                return self._send(200, raw=(media / ("gen.mp4" if st.video_mode != "corrupt" else "photo.png")).read_bytes(), ctype="video/mp4")
            self._send(404, {})
    srv = HTTPServer(("127.0.0.1", 0), H); t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    st.url = f"http://127.0.0.1:{srv.server_port}"; yield st; srv.shutdown()

@pytest.fixture()
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("AIVE_") and k not in ("AIVE_LOG_LEVEL", "AIVE_MODEL_DIR"): monkeypatch.delenv(k, raising=False)
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"): monkeypatch.delenv(k, raising=False)
    import config; config._loaded["done"] = True            # do not read a developer's real .env during tests
    return monkeypatch

def have_models(*names):
    from voice import neural_ready; from setup_models import matting_ready, whisper_ready
    ok = {"voice": neural_ready, "matting": matting_ready, "whisper": lambda: whisper_ready("tiny") or whisper_ready("base")}
    return all(ok[n]() for n in names)
