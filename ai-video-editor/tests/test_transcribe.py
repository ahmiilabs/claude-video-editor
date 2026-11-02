"""Whisper tests: the sherpa test runs REAL offline Whisper if installed. faster-whisper / openai-whisper / openai-api paths are MOCKED
(no model or key available in CI): they prove the adapter plumbing, not those engines' accuracy."""
import json, os, shutil, subprocess, sys, types
from pathlib import Path
import numpy as np, pytest
from conftest import ff, have_models, SCRIPTS
import transcribe as TX, captions

def tone_burst_wav(path, bursts, total):
    sr = 16000; t = np.arange(int(sr * total)) / sr; x = np.zeros_like(t)
    for a, b in bursts: m = (t >= a) & (t < b); x[m] = 0.4 * np.sin(2 * np.pi * 220 * t[m]) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t[m]))
    from scipy.io import wavfile; wavfile.write(str(path), sr, (x * 32767).astype(np.int16)); return str(path)

def test_words_are_placed_only_in_voiced_audio_and_pauses_take_no_time(tmp_path):
    w = tone_burst_wav(tmp_path / "t.wav", [(0.5, 1.5), (3.0, 4.0)], 5.0); a = TX.load_audio(w); env, hop = TX.energy_envelope(a); mask = TX.voiced_mask(env, hop)
    words = TX.place_words("one two three four", 0.0, 5.0, mask, hop)
    assert len(words) == 4 and all(x["approx"] for x in words) and all(words[i]["start"] <= words[i + 1]["start"] for i in range(3))
    assert words[0]["start"] >= 0.45 and words[-1]["end"] <= 4.1, "no word may start before speech or end after it"
    assert not any(1.6 < x["start"] < 2.9 or 1.6 < x["end"] < 2.9 for x in words), "the 1.5s pause must contain no words"

def test_align_scenes_keeps_every_word_inside_its_scene_window(tmp_path):
    w = tone_burst_wav(tmp_path / "t.wav", [(0.2, 1.8), (2.2, 3.8)], 4.0); scenes = [{}, {}]; words = TX.align_scenes(w, scenes, [2.0, 2.0], ["alpha beta gamma", "delta epsilon"])
    assert [x["scene"] for x in words] == [0, 0, 0, 1, 1]; assert all(0 <= x["start"] and x["end"] <= 2.0 + 1e-6 for x in words if x["scene"] == 0) and all(x["start"] >= 2.0 for x in words if x["scene"] == 1)

def test_engine_errors_are_clear(tmp_path, media, clean_env):
    with pytest.raises(TX.TranscribeError, match="unknown engine"): TX.transcribe(str(media / "clip.mp4"), engine="magic")
    silent = tmp_path / "s.wav"; ff("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "2", silent)
    if have_models("whisper"):
        with pytest.raises(TX.TranscribeError, match="no speech"): TX.transcribe(str(silent), "en", engine="sherpa")
    with pytest.raises(TX.TranscribeError, match="cannot decode|empty|shorter"): TX.transcribe(str(tmp_path / "missing.wav"), engine="sherpa")

def test_auto_with_no_engine_lists_how_to_install(clean_env, monkeypatch, media):
    monkeypatch.setattr(TX, "_sherpa", lambda *a, **k: (_ for _ in ()).throw(TX.TranscribeError("sherpa: not installed"))); monkeypatch.delenv("AIVE_TRANSCRIBE_API_KEY", raising=False)
    with pytest.raises(TX.TranscribeError, match="setup_models.py --whisper"): TX.transcribe(str(media / "clip.mp4"), engine="auto")

@pytest.mark.skipif(not have_models("whisper") or not shutil.which("espeak-ng"), reason="needs offline whisper (setup_models.py --whisper tiny) and espeak-ng")
def test_real_offline_whisper_transcribes_and_reports_honest_granularity(tmp_path):
    wav = tmp_path / "s.wav"; subprocess.run(["espeak-ng", "-v", "en-us", "-s", "140", "-w", str(wav), "Why is tea so expensive on trains? Because you have no other option."], check=True)
    tr = TX.transcribe(str(wav), "en", engine="sherpa"); text = " ".join(s["text"] for s in tr["segments"]).lower()
    assert "expensive" in text and "option" in text and tr["granularity"] == "segment" and all(w["approx"] for w in tr["words"]); assert all(0 <= w["start"] < w["end"] <= tr["audio_seconds"] + 0.1 for w in tr["words"])
    captions.write_srt(tr["words"], tmp_path / "t.srt"); assert "-->" in (tmp_path / "t.srt").read_text()

def test_faster_whisper_adapter_with_a_MOCKED_module(monkeypatch, media):
    class W: 
        def __init__(s, word, a, b): s.word, s.start, s.end = word, a, b
    class S: 
        def __init__(s): s.start, s.end, s.text, s.words = 0.1, 1.2, " hello world", [W(" hello", 0.1, 0.6), W(" world", 0.7, 1.2)]
    class M:
        def __init__(s, *a, **k): pass
        def transcribe(s, *a, **k): assert k["word_timestamps"] is True; return iter([S()]), None
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=M)); tr = TX.transcribe(str(media / "clip.mp4"), "en", engine="faster-whisper")
    assert tr["granularity"] == "word" and [w["w"] for w in tr["words"]] == ["hello", "world"] and not any(w["approx"] for w in tr["words"])

def test_openai_api_adapter_against_mock_server(clean_env, mock_server, media):
    clean_env.setenv("AIVE_TRANSCRIBE_API_KEY", "sk-trkey123456789"); clean_env.setenv("AIVE_TRANSCRIBE_BASE_URL", mock_server.url + "/v1")
    tr = TX.transcribe(str(media / "clip.mp4"), "en", engine="openai-api"); path, headers, body = mock_server.requests[-1]
    assert path.endswith("/audio/transcriptions") and headers["authorization"] == "Bearer sk-trkey123456789" and b"timestamp_granularities[]" in body and tr["granularity"] == "word" and [w["w"] for w in tr["words"]] == ["hello", "brave", "world"]

def test_transcribe_cli_writes_srt_ass_json(tmp_path, media, mock_server, clean_env):
    env = {**os.environ, "AIVE_TRANSCRIBE_API_KEY": "sk-trkey123456789", "AIVE_TRANSCRIBE_BASE_URL": mock_server.url + "/v1"}
    r = subprocess.run([sys.executable, str(SCRIPTS / "transcribe.py"), str(media / "clip.mp4"), "--engine", "openai-api", "--lang", "en", "--srt", str(tmp_path / "o.srt"), "--ass", str(tmp_path / "o.ass"), "--json", str(tmp_path / "o.json"), "--style", "karaoke"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-300:]; assert "\\k" in (tmp_path / "o.ass").read_text(encoding="utf-8") and json.loads((tmp_path / "o.json").read_text())["engine"] == "openai-api"
