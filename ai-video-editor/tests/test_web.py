"""Behaviour needed on claude.ai web: read-only skill folder, relative media paths, time estimates, trimming, joining, voice notes."""
import json, os, shutil, subprocess, sys
from pathlib import Path
import pytest
from conftest import SCRIPTS, ff, have_models
import common, reel

def spec_with_image(): return {"voice": {"mode": "silent"}, "scenes": [{"text": "One", "narration": "hello there", "image": "bg.png", "duration": 1.5}]}

def test_relative_media_paths_resolve_next_to_the_spec_not_the_cwd(tmp_path, media):
    d = tmp_path / "proj"; d.mkdir(); shutil.copy(media / "photo.png", d / "bg.png"); (d / "spec.json").write_text(json.dumps(spec_with_image())); other = tmp_path / "elsewhere"; other.mkdir()
    r = subprocess.run([sys.executable, str(SCRIPTS / "reel.py"), str(d / "spec.json"), "--out", str(other / "o.mp4"), "--fps", "12"], capture_output=True, text=True, cwd=other); assert r.returncode == 0, r.stderr[-300:]
    assert json.loads((other / "o.mp4.render.json").read_text())["assets"][0] == str((d / "bg.png").resolve())

def test_resolve_spec_paths_covers_every_media_field(tmp_path):
    for n in ("a.png", "v.mp4", "s.wav", "m.mp3", "x.wav"): (tmp_path / n).write_bytes(b"x")
    sp = {"voice": {"file": "v.wav"}, "bgm": "m.mp3", "audio": {"music": "synth:calm"}, "scenes": [{"image": "a.png", "video": "v.mp4", "audio": {"file": "s.wav"}}]}; (tmp_path / "v.wav").write_bytes(b"x")
    out = common.resolve_spec_paths(sp, tmp_path); assert all(os.path.isabs(x) for x in (out["voice"]["file"], out["bgm"], out["scenes"][0]["image"], out["scenes"][0]["video"], out["scenes"][0]["audio"]["file"])) and out["audio"]["music"] == "synth:calm"
    assert common.resolve_spec_paths({"scenes": [{"image": "missing.png"}]}, tmp_path)["scenes"][0]["image"] == "missing.png"       # unresolved stays as written so the error names it

def test_batch_job_copies_keep_working_media_paths(tmp_path, media):
    import batch; d = tmp_path / "p"; d.mkdir(); shutil.copy(media / "photo.png", d / "bg.png"); (d / "s.json").write_text(json.dumps(spec_with_image()))
    b = batch.Batch(tmp_path / "b"); jid = b.add(d / "s.json"); assert os.path.exists(json.loads((tmp_path / "b" / "jobs" / jid / "spec.json").read_text())["scenes"][0]["image"])

def test_time_estimate_scales_and_is_reasonable():
    one = {"voice": {"mode": "silent"}, "scenes": [{"text": "a", "narration": "x " * 10, "duration": 10}]}; two = {"voice": {"mode": "silent"}, "scenes": [{"text": "a", "narration": "x " * 10, "duration": 20}]}
    a, b = reel.estimate_seconds(one), reel.estimate_seconds(two); assert b["render_s"] > a["render_s"] * 1.8 and a["voice_s"] == 0 and reel.estimate_seconds(one, "hd")["render_s"] > a["render_s"]
    assert reel.estimate_seconds({"voice": {"mode": "auto"}, "scenes": one["scenes"]})["voice_s"] > 0

def test_effects_start_and_duration_trim_the_clip(tmp_path, media):
    out = tmp_path / "t.mp4"; r = subprocess.run([sys.executable, str(SCRIPTS / "effects.py"), str(media / "clip.mp4"), "--effect", "color-grade", "--start", "1", "--duration", "1.5", "--out", str(out)], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout); assert 1.2 < d < 1.9
    assert not list(Path(os.environ.get("TMPDIR", "/tmp")).glob("fxtrim_*")), "trim workspace must be cleaned"

def test_join_concatenates_parts_with_matching_format(tmp_path, media):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"; ff("-i", media / "clip.mp4", "-t", "1.5", "-vf", "scale=360:640", a); ff("-i", media / "clip.mp4", "-ss", "1", "-t", "1.5", "-vf", "scale=360:640", b)
    out = tmp_path / "j.mp4"; assert subprocess.run([sys.executable, str(SCRIPTS / "join.py"), str(a), str(b), "--out", str(out)], capture_output=True).returncode == 0
    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout); assert abs(d - 3.0) < 0.3
    assert subprocess.run([sys.executable, str(SCRIPTS / "join.py"), str(a), str(tmp_path / "nope.mp4"), "--out", str(out)], capture_output=True).returncode != 0

@pytest.mark.parametrize("ext,codec", [("ogg", "libvorbis"), ("m4a", "aac"), ("opus", "libopus")])
def test_whatsapp_style_voice_notes_work_as_voice_file(tmp_path, ext, codec):
    note = tmp_path / f"note.{ext}"; r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=300:d=3", "-c:a", codec, str(note)], capture_output=True)
    if r.returncode != 0: pytest.skip(f"ffmpeg lacks {codec}")
    res = reel.render_reel({"voice": {"mode": "file", "file": str(note)}, "scenes": [{"text": "Hi", "narration": "one two three"}]}, tmp_path / "o.mp4", fps=12); assert res["qa_passed"] and res["voice_mode"] == "file"

@pytest.mark.skipif(not have_models("matting"), reason="matting model not installed")
def test_transparent_webm_has_alpha(tmp_path, media):
    out = tmp_path / "a.webm"; r = subprocess.run([sys.executable, str(SCRIPTS / "effects.py"), str(media / "clip.mp4"), "--effect", "bg-remove", "--alpha-format", "webm", "--max-width", "240", "--duration", "1", "--out", str(out)], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
    pf = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_name:stream_tags=alpha_mode", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout; assert "vp9" in pf and "1" in pf

def test_env_check_reports_web_notes_and_works_without_network():
    r = subprocess.run([sys.executable, str(SCRIPTS / "env_check.py"), "--no-network"], capture_output=True, text=True); d = json.loads(r.stdout); assert "web_notes" in d and "features" in d and d["network"] == {}

def test_web_setup_core_only_is_fast_and_reports_json():
    r = subprocess.run([sys.executable, str(SCRIPTS / "web_setup.py")], capture_output=True, text=True, timeout=180); d = json.loads(r.stdout); assert d["ready"] and d["requested_ok"] and d["status"]["pillow"]

@pytest.mark.skipif(os.geteuid() != 0 or not shutil.which("runuser"), reason="needs root + runuser to emulate a non-root user")
def test_skill_works_from_a_read_only_folder_as_non_root():
    import tempfile; tmp_path = Path(tempfile.mkdtemp(prefix="ro_skill_test_")); os.chmod(tmp_path, 0o755)     # pytest's own tmp dirs are 0700: nobody could not traverse them
    ro = tmp_path / "skill"; shutil.copytree(SCRIPTS.parent, ro, ignore=shutil.ignore_patterns("__pycache__", "tests", ".pytest_cache")); out = tmp_path / "out"; home = tmp_path / "home"; models = tmp_path / "models"
    for d in (out, home, models): d.mkdir()
    spec = {"voice": {"mode": "silent"}, "scenes": [{"text": "RO", "narration": "read only test", "duration": 1.5}]}; (tmp_path / "s.json").write_text(json.dumps(spec))
    subprocess.run(["chmod", "-R", "a-w", str(ro)], check=True); subprocess.run(["chmod", "-R", "a+rwX", str(out), str(home), str(models)], check=True); subprocess.run(["chmod", "a+rx", str(tmp_path), str(tmp_path / "s.json")], check=True)
    before = sorted(str(p) for p in ro.rglob("*"))
    try:
        r = subprocess.run(["runuser", "-u", "nobody", "--", "env", f"HOME={home}", f"AIVE_MODEL_DIR={models}", sys.executable, str(ro / "scripts" / "reel.py"), str(tmp_path / "s.json"), "--out", str(out / "o.mp4"), "--fps", "12"], capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-400:]; assert (out / "o.mp4").exists() and sorted(str(p) for p in ro.rglob("*")) == before, "nothing may be written into the skill folder"
    finally:
        subprocess.run(["chmod", "-R", "u+w", str(ro)]); shutil.rmtree(tmp_path, ignore_errors=True)
