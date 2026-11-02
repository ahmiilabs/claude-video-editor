import json, subprocess, sys
from pathlib import Path
import pytest
from conftest import ff, SCRIPTS
import qa_check, audio

def mk(path, vf="testsrc2=s=360x640:d=4:r=24", af="sine=f=440:d=4", extra=(), acodec="aac"):
    ff("-f", "lavfi", "-i", vf, "-f", "lavfi", "-i", af, "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", acodec, *extra, path); return str(path)

def failed(rep): return set(rep["failed_checks"])

# ---------------------------------------------------------------- QA: positive + every negative
def test_good_video_passes(tmp_path):
    rep = qa_check.run_qa(mk(tmp_path / "ok.mp4"), aspect="9:16", quiet=True); assert rep["passed"], rep["failed_checks"]

def test_black_video_fails(tmp_path):
    assert "black_frames" in failed(qa_check.run_qa(mk(tmp_path / "b.mp4", vf="color=c=black:s=360x640:d=4:r=24"), quiet=True))

def test_frozen_video_fails(tmp_path):
    assert "frozen_frames" in failed(qa_check.run_qa(mk(tmp_path / "f.mp4", vf="color=c=blue:s=360x640:d=4:r=24"), quiet=True))

def test_silent_audio_fails_when_audio_expected(tmp_path):
    assert "silence" in failed(qa_check.run_qa(mk(tmp_path / "s.mp4", af="anullsrc=r=24000:cl=mono"), quiet=True))

def test_clipped_audio_fails(tmp_path):
    p = tmp_path / "c.mov"; ff("-f", "lavfi", "-i", "testsrc2=s=360x640:d=3:r=24", "-f", "lavfi", "-i", "sine=f=440:d=3", "-af", "volume=40dB,alimiter=limit=1:level=disabled", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", p)
    assert "audio_clipping" in failed(qa_check.run_qa(str(p), quiet=True))

def test_wrong_aspect_and_resolution_fail(tmp_path):
    p = mk(tmp_path / "w.mp4", vf="testsrc2=s=640x360:d=3:r=24"); assert "aspect_ratio" in failed(qa_check.run_qa(p, aspect="9:16", quiet=True))
    assert "resolution" in failed(qa_check.run_qa(mk(tmp_path / "r.mp4"), res=(720, 1280), quiet=True))

def test_wrong_fps_and_duration_fail(tmp_path):
    p = mk(tmp_path / "d.mp4"); assert "frame_rate" in failed(qa_check.run_qa(p, fps=30, quiet=True)); assert "duration" in failed(qa_check.run_qa(p, expect_duration=10, quiet=True))

def test_missing_audio_stream_fails_only_when_expected(tmp_path):
    p = tmp_path / "na.mp4"; ff("-f", "lavfi", "-i", "testsrc2=s=360x640:d=3:r=24", "-c:v", "libx264", "-pix_fmt", "yuv420p", p)
    assert "audio_stream" in failed(qa_check.run_qa(str(p), quiet=True)); assert qa_check.run_qa(str(p), expect_audio=False, quiet=True)["passed"]

def test_corrupt_and_truncated_files_fail(tmp_path):
    good = Path(mk(tmp_path / "g.mp4")); data = bytearray(good.read_bytes())
    for i in range(len(data) // 3, len(data) // 3 + 4000): data[i] = (data[i] + 97) % 256
    (tmp_path / "corrupt.mp4").write_bytes(bytes(data)); (tmp_path / "trunc.mp4").write_bytes(good.read_bytes()[:3000])
    assert not qa_check.run_qa(str(tmp_path / "corrupt.mp4"), quiet=True)["passed"]; assert "file_integrity" in failed(qa_check.run_qa(str(tmp_path / "trunc.mp4"), quiet=True))
    assert "file_integrity" in failed(qa_check.run_qa(str(tmp_path / "missing.mp4"), quiet=True))

def test_missing_assets_and_missing_captions_from_sidecar(tmp_path):
    p = mk(tmp_path / "m.mp4"); (tmp_path / "m.mp4.render.json").write_text(json.dumps({"status": "ok", "assets": [str(tmp_path / "gone.png")], "captions": {"expected": True, "windows": [[0.5, 1.5], [2.0, 3.0]], "dialogues": 2}}))
    f = failed(qa_check.run_qa(p, quiet=True)); assert "missing_assets" in f and "captions" in f              # test pattern has no caption text in the caption zone

def test_failed_render_sidecar_fails_qa(tmp_path):
    p = mk(tmp_path / "x.mp4"); (tmp_path / "x.mp4.render.json").write_text(json.dumps({"status": "failed", "error": "boom"}))
    assert "render_status" in failed(qa_check.run_qa(p, quiet=True))

def test_failed_mandatory_check_gives_nonzero_exit_and_json_report(tmp_path):
    bad = mk(tmp_path / "b.mp4", vf="color=c=black:s=360x640:d=3:r=24"); rep = tmp_path / "r.json"
    r = subprocess.run([sys.executable, str(SCRIPTS / "qa_check.py"), bad, "--report", str(rep)], capture_output=True, text=True)
    assert r.returncode == 1 and "FAIL" in r.stdout; d = json.loads(rep.read_text()); assert d["passed"] is False and d["failed_checks"]
    ok = subprocess.run([sys.executable, str(SCRIPTS / "qa_check.py"), mk(tmp_path / "g.mp4")], capture_output=True, text=True); assert ok.returncode == 0

def test_old_always_true_bug_is_gone():
    assert "or True" not in (SCRIPTS / "qa_check.py").read_text()

# ---------------------------------------------------------------- audio pipeline
def test_enhance_voice_normalises_loudness_without_clipping(tmp_path, media):
    audio.enhance_voice(media / "voice_tone.wav", tmp_path / "e.wav"); a = audio.analyse(tmp_path / "e.wav")
    assert -19 < a["lufs"] < -13 and a["peak_db"] < -0.5

def test_mix_has_exact_length_valid_stream_and_no_clipping(tmp_path, media):
    audio.synth_music("calm", 10, tmp_path / "m.wav"); audio.synth_sfx("whoosh", tmp_path / "w.wav")
    r = audio.mix(tmp_path / "mix.wav", 8.0, voice=str(media / "voice_tone.wav"), music=str(tmp_path / "m.wav"), sfx=[{"file": str(tmp_path / "w.wav"), "at": 7.0, "volume": 0.8}])
    assert r["ducking"] and r["sfx"] == 1 and abs(r["duration"] - 8.0) < 0.1; a = audio.analyse(tmp_path / "mix.wav"); assert a["peak_db"] < -0.5

def test_sidechain_ducking_lowers_music_during_narration_and_recovers(tmp_path, media):
    audio.synth_music("calm", 14, tmp_path / "m.wav"); ff("-i", media / "voice_tone.wav", "-t", "6", "-af", "volume=0.6", tmp_path / "v6.wav")
    def music_levels(duck):                       # voice_volume 0 keeps the sidechain key but leaves ONLY the music audible
        audio.mix(tmp_path / f"o{duck}.wav", 12.0, voice=str(tmp_path / "v6.wav"), music=str(tmp_path / "m.wav"), voice_vol=0.0, duck=duck)
        return audio.analyse(tmp_path / f"o{duck}.wav", (1.0, 5.0))["mean_db"], audio.analyse(tmp_path / f"o{duck}.wav", (8.0, 11.0))["mean_db"]
    d_in, d_out = music_levels(True); n_in, n_out = music_levels(False)
    assert d_out - d_in > 3.0, f"music should drop >3 dB under speech (drop {d_out - d_in:.1f})"; assert abs(n_out - n_in) < 1.0
