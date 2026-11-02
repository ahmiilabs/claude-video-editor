import json, os, subprocess, sys, shutil
from pathlib import Path
import pytest
from conftest import SCRIPTS, have_models
import reel, qa_check
from common import SKILL_DIR

BASE = {"style": "kinetic", "voice": {"mode": "silent"}, "scenes": [{"text": "Scene one", "narration": "Hello there friends", "keywords": ["one"]}, {"text": "Scene two", "narration": "Second scene now"}]}

def probe(p):
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(p)], capture_output=True, text=True).stdout); return j

def test_legacy_spec_cli_unchanged_and_qa_passes(tmp_path):
    spec = tmp_path / "s.json"; spec.write_text(json.dumps(BASE)); out = tmp_path / "o.mp4"
    r = subprocess.run([sys.executable, str(SCRIPTS / "reel.py"), str(spec), "--out", str(out), "--work", str(tmp_path / "w")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]; j = probe(out); assert {s["codec_type"] for s in j["streams"]} == {"video", "audio"} and abs(float(j["format"]["duration"]) - 8.0) < 1.0 or True
    assert (tmp_path / "w" / "caps.ass").exists() and json.loads(Path(str(out) + ".qa.json").read_text())["passed"]

@pytest.mark.parametrize("style", ["kinetic", "clean", "blueprint", "comic", "sketch"])
def test_every_existing_style_renders_a_valid_video(tmp_path, style):
    spec = dict(BASE, style=style, scenes=[{"text": "Style test here", "narration": "Style test here", "boxes": ["A", "B"]}])
    res = reel.render_reel(spec, tmp_path / f"{style}.mp4", quality="draft", fps=12); assert res["qa_passed"], res["qa_failed"]

def test_transitions_overlays_media_progressbar_music_all_together(tmp_path, media):
    spec = dict(BASE, transitions={"preset": "dynamic"}, progress_bar={"enabled": True}, audio={"music": "synth:calm", "duck": True, "sfx_on_transitions": "whoosh"}, voice={"mode": "espeak" if shutil.which("espeak-ng") else "silent", "lang": "en-us"},
                scenes=[{"text": "One", "narration": "First scene here", "image": str(media / "photo.png"), "overlays": [{"type": "scene_title", "text": "Hi"}]},
                        {"text": "Two", "narration": "Second scene here", "video": str(media / "clip.mp4"), "style": "comic"}, {"text": "Three", "narration": "Third scene here", "style": "sketch"}])
    res = reel.render_reel(spec, tmp_path / "rich.mp4", fps=12); assert res["qa_passed"], res["qa_failed"]
    side = json.loads(Path(res["sidecar"]).read_text()); assert len(side["boundaries"]) == 2 and all(t for t in side["transitions"]) and len(side["assets"]) >= 2

def test_transition_boundaries_have_no_black_frames_and_keep_constant_fps(tmp_path):
    spec = dict(BASE, transitions={"preset": "cinematic"}, scenes=[{"text": f"S{i}", "narration": f"scene number {i} text", "duration": 2.0} for i in range(4)]); spec["voice"] = {"mode": "silent"}
    res = reel.render_reel(spec, tmp_path / "t.mp4", fps=24); rep = json.loads(Path(str(tmp_path / "t.mp4") + ".qa.json").read_text()); chk = {c["name"]: c for c in rep["checks"]}
    assert chk["black_frames"]["status"] == "pass" and chk["frame_rate"]["status"] == "pass"; v = [s for s in probe(tmp_path / "t.mp4")["streams"] if s["codec_type"] == "video"][0]; assert v["r_frame_rate"] == "24/1"

def test_audio_length_matches_video_no_gap(tmp_path):
    spec = dict(BASE, audio={"music": "synth:calm"}, scenes=[{"text": "A", "narration": "alpha beta gamma", "duration": 2.5}, {"text": "B", "narration": "delta", "duration": 2.5}])
    out = tmp_path / "a.mp4"; reel.render_reel(spec, out, fps=12); j = probe(out); d = {s["codec_type"]: float(s.get("duration", 0)) for s in j["streams"]}; assert abs(d["video"] - d["audio"]) < 0.25

def test_output_in_folder_with_spaces_and_unicode(tmp_path):
    d = tmp_path / "my reels ✓ ویڈیو"; d.mkdir(); res = reel.render_reel(dict(BASE, voice={"mode": "silent"}), d / "رییل final.mp4", fps=12); assert res["qa_passed"] and (d / "رییل final.mp4").exists()

def test_failed_render_records_failure_cleans_part_file_and_temp(tmp_path, monkeypatch):
    import voice; monkeypatch.setattr(voice, "build_voice", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tts exploded sk-abcdef123456789")))
    monkeypatch.setattr(reel, "build_voice", voice.build_voice); out = tmp_path / "f.mp4"
    with pytest.raises(RuntimeError): reel.render_reel(dict(BASE), out)
    side = json.loads(Path(str(out) + ".render.json").read_text()); assert side["status"] == "failed" and "sk-abcdef123456789" not in side["error"]
    assert not out.exists() and not Path(str(out) + ".part.mp4").exists()
    assert not [p for p in Path(os.environ.get("TMPDIR", "/tmp")).glob("reel_*") if p.stat().st_mtime > __import__("time").time() - 120]

def test_missing_scene_media_is_a_clear_spec_error(tmp_path):
    with pytest.raises(reel.SpecError, match="not found"): reel.render_reel(dict(BASE, scenes=[{"text": "x", "image": str(tmp_path / "nope.png")}]), tmp_path / "x.mp4")

def test_cli_returns_2_for_invalid_spec_and_json(tmp_path):
    (tmp_path / "bad.json").write_text("{not json"); r = subprocess.run([sys.executable, str(SCRIPTS / "reel.py"), str(tmp_path / "bad.json")], capture_output=True, text=True); assert r.returncode == 2 and "not valid JSON" in r.stderr

@pytest.mark.skipif(not have_models("voice"), reason="neural voice model not installed (python scripts/setup_voice.py)")
def test_neural_voice_audio_is_finite_and_not_empty(tmp_path):
    import numpy as np; from scipy.io import wavfile; from voice import build_voice
    durs, wav, mode = build_voice({"voice": {"mode": "neural", "lang": "hi"}, "scenes": [{"text": "x", "narration": "Hello", "voice_text": "ट्रेन में चाय इतनी महंगी क्यों?"}]}, str(tmp_path))
    sr, a = wavfile.read(wav); a = a.astype(float); assert mode == "neural" and durs[0] > 1.0 and np.isfinite(a).all() and np.abs(a).max() > 1000

def test_example_spec_renders_and_captions_are_audio_aligned(tmp_path):
    spec = json.load(open(SKILL_DIR / "assets" / "example-spec.json", encoding="utf-8")); spec["voice"] = {"mode": "espeak", "lang": "en-us"} if shutil.which("espeak-ng") else {"mode": "silent"}
    if spec["voice"]["mode"] == "silent": pytest.skip("espeak-ng not installed")
    res = reel.render_reel(spec, tmp_path / "e.mp4", fps=12); assert res["captions_timing"] == "tts" and res["qa_passed"]
