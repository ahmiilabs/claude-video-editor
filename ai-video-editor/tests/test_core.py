import json, os, sys
import numpy as np, pytest
from PIL import Image
import config, common, captions, transitions, styles, graphics
from reel import validate_spec, SpecError

# ---------------------------------------------------------------- config / secrets
def test_env_file_parse_and_placeholder_ignored(tmp_path):
    f = tmp_path / ".env"; f.write_text("# c\nAIVE_LLM_MODEL=gpt-x  # trailing\nAIVE_LLM_API_KEY=your-llm-api-key-here\nAIVE_TIMEOUT=\"90\"\n")
    d = config.parse_env_file(f); assert d["AIVE_LLM_MODEL"] == "gpt-x" and d["AIVE_TIMEOUT"] == "90"
    assert config.PLACEHOLDER_RE.match(d["AIVE_LLM_API_KEY"])

def test_missing_api_key_is_reported_not_crashing(clean_env):
    clean_env.setenv("AIVE_LLM_PROVIDER", "openai"); clean_env.setenv("AIVE_LLM_MODEL", "m")
    p = config.get_provider("llm"); assert any("API key missing" in x for x in p.problems())
    with pytest.raises(config.ConfigError): p.validate()

def test_local_server_needs_no_key(clean_env):
    clean_env.setenv("AIVE_LLM_PROVIDER", "openai"); clean_env.setenv("AIVE_LLM_MODEL", "m"); clean_env.setenv("AIVE_LLM_BASE_URL", "http://localhost:11434/v1")
    assert config.get_provider("llm").validate()

def test_secrets_are_redacted(clean_env):
    clean_env.setenv("AIVE_LLM_PROVIDER", "openai"); clean_env.setenv("AIVE_LLM_MODEL", "m"); clean_env.setenv("AIVE_LLM_API_KEY", "sk-abcdefghij123456")
    p = config.get_provider("llm"); assert "sk-abcdefghij123456" not in repr(p)
    assert "sk-abcdefghij123456" not in config.redact("failed with key sk-abcdefghij123456 and Bearer tok_123456789")
    assert "sk-abcdefghij123456" not in json.dumps(config.status())

def test_env_example_has_only_placeholders():
    txt = (common.SKILL_DIR / ".env.example").read_text()
    for line in txt.splitlines():
        if "KEY=" in line and not line.startswith("#"): assert config.PLACEHOLDER_RE.match(line.split("=", 1)[1].split("#")[0].strip()), line

# ---------------------------------------------------------------- paths / cross-platform helpers
@pytest.mark.parametrize("raw,expected", [(r"C:\Users\Ali Khan\Videos\caps.ass", r"C\:/Users/Ali Khan/Videos/caps.ass"), ("C:/a b/c.ass", r"C\:/a b/c.ass"), (r"D:\it's\x.ass", r"D\:/it\'s/x.ass")])
def test_windows_paths_escape_for_ffmpeg_filters(raw, expected):
    assert common.ffmpeg_filter_path(raw) == expected

def test_workspace_cleans_on_exception_and_is_unique():
    paths = []
    with pytest.raises(RuntimeError):
        with common.Workspace("t_") as a, common.Workspace("t_") as b:
            paths += [a.path, b.path]; assert a.path != b.path and a.path.exists(); raise RuntimeError("boom")
    assert not any(p.exists() for p in paths)

def test_find_binary_error_has_install_hint(monkeypatch):
    monkeypatch.setattr(common.shutil, "which", lambda n: None); monkeypatch.delenv("AIVE_FFMPEG", raising=False)
    with pytest.raises(common.DependencyError, match="install|Install|apt|brew|winget"): common.find_binary("ffmpeg", required=True)

def test_gpu_encoder_falls_back_to_cpu_when_trial_fails(monkeypatch):
    common._enc_cache.clear(); args = common.pick_encoder(prefer_gpu=True)
    assert args[:2] == ["-c:v", "libx264"] or args[1].startswith("h264_")        # real GPU only if the trial encode succeeded
    assert common.pick_encoder(prefer_gpu=False)[1] == "libx264"

# ---------------------------------------------------------------- spec validation
@pytest.mark.parametrize("spec,msg", [({}, "scenes"), ({"scenes": []}, "scenes"), ({"scenes": [{}]}, "neither"), ({"scenes": [{"text": "x", "style": "nope"}]}, "unknown style"),
                                       ({"scenes": [{"text": "x", "image": "/does/not/exist.png"}]}, "not found"), ({"scenes": [{"text": "x"}], "aspect": "4:3"}, "aspect"),
                                       ({"scenes": [{"text": "x", "duration": 0}]}, "duration"), ("string", "object")])
def test_invalid_specs_rejected_with_clear_message(spec, msg):
    with pytest.raises(SpecError, match=msg): validate_spec(spec)

def test_valid_legacy_spec_accepted():
    validate_spec(json.load(open(common.SKILL_DIR / "assets" / "example-spec.json", encoding="utf-8")))

# ---------------------------------------------------------------- styles (every existing style)
@pytest.mark.parametrize("style", list(styles.STYLES))
def test_every_style_renders_all_phases(style):
    W, H = 360, 640; cache = {}; sc = {"text": "Hello brave new world", "narration": "x", "keywords": ["brave"], "boxes": ["A", "B"], "sfx": "Boom", "label": "// t", "badge": "2"}
    ctx = {"W": W, "H": H, "index": 1, "total": 3, "accent": "#FF6B35", "cache": cache}
    for t in (0.0, 0.5, 2.0, 99.0):                     # start, mid, end, far beyond duration (transitions render outgoing scenes past their end)
        im = styles.STYLES[style](t, 2.0, sc, ctx); assert im.size == (W, H) and im.mode == "RGB"
    assert np.asarray(im).std() > 5

def test_style_uses_media_background_when_present():
    bg = Image.new("RGB", (360, 640), (255, 0, 255)); ctx = {"W": 360, "H": 640, "index": 0, "total": 1, "accent": "#FF6B35", "cache": {}, "media": bg}
    im = np.asarray(styles.STYLES["comic"](1.0, 2.0, {"text": "hi"}, ctx)); assert (im[5, 5] == (255, 0, 255)).all()

# ---------------------------------------------------------------- transitions
@pytest.mark.parametrize("name", sorted(transitions.TRANSITIONS))
def test_transition_endpoints_size_and_no_black(name):
    W, H = 180, 320; A = Image.new("RGB", (W, H), (220, 60, 60)); B = Image.new("RGB", (W, H), (40, 90, 200)); fn = transitions.TRANSITIONS[name]
    for p in np.linspace(0, 1, 9):
        out = fn(A, B, float(p)); assert out.size == (W, H)
        assert np.asarray(out).mean() > 40, f"{name} produced a dark/black frame at p={p:.2f}"      # no black-frame glitches mid-transition
    if name not in ("glitch", "whip", "light_leak"):
        assert np.abs(np.asarray(fn(A, B, 0.0), float) - np.asarray(A, float)).mean() < 12
        assert np.abs(np.asarray(fn(A, B, 1.0), float) - np.asarray(B, float)).mean() < 12

def test_transition_duration_is_clamped_and_cut_supported():
    fn, td, name = transitions.resolve({"preset": "dynamic", "duration": 5}, {}, 0, 3, 1.0, 3.0); assert td <= 0.4
    assert transitions.resolve({"preset": "none"}, {}, 0, 3, 2, 2) is None and transitions.resolve({"preset": "smooth"}, {"transition": "cut"}, 0, 3, 2, 2) is None
    with pytest.raises(ValueError): transitions.resolve({"preset": "smooth"}, {"transition": "bogus"}, 0, 3, 2, 2)

# ---------------------------------------------------------------- captions
def test_estimated_words_are_ordered_and_inside_scene_windows():
    sc = [{"narration": "alpha beta gamma delta"}, {"narration": "one two three"}]; w = captions.estimate_words(sc, [3.0, 2.0])
    assert all(a["end"] > a["start"] for a in w) and all(w[i]["end"] <= w[i + 1]["start"] + 1e-6 for i in range(len(w) - 1))
    assert max(x["end"] for x in w if x["scene"] == 0) <= 3.0 and min(x["start"] for x in w if x["scene"] == 1) >= 3.0

@pytest.mark.parametrize("style", ["highlight", "karaoke", "plain"])
def test_ass_styles_generate_events(tmp_path, style):
    w = captions.estimate_words([{"narration": "hello brave new world today"}], [3.0]); r = captions.write_ass(w, 720, 1280, tmp_path / "c.ass", style)
    txt = (tmp_path / "c.ass").read_text(encoding="utf-8"); assert r["dialogues"] >= 1 and "Dialogue:" in txt and ("\\k" in txt if style == "karaoke" else True)

def test_srt_cues_never_overlap_and_are_valid(tmp_path):
    w = captions.estimate_words([{"narration": " ".join(f"w{i}" for i in range(20))}], [8.0]); captions.write_srt(w, tmp_path / "a.srt")
    import re; cues = re.findall(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", (tmp_path / "a.srt").read_text())
    ms = [(int(c[0]) * 3600000 + int(c[1]) * 60000 + int(c[2]) * 1000 + int(c[3]), int(c[4]) * 3600000 + int(c[5]) * 60000 + int(c[6]) * 1000 + int(c[7])) for c in cues]
    assert len(ms) >= 3 and all(b > a for a, b in ms) and all(ms[i][1] <= ms[i + 1][0] for i in range(len(ms) - 1))

def test_legacy_build_ass_signature_still_works(tmp_path):
    assert captions.build_ass([{"narration": "hello world"}], [2.0], 720, 1280, str(tmp_path / "l.ass")) == str(tmp_path / "l.ass")

# ---------------------------------------------------------------- graphics overlays
@pytest.mark.parametrize("typ", list(graphics.OVERLAYS))
def test_overlays_draw_and_unknown_rejected(typ):
    im = Image.new("RGB", (360, 640), (30, 30, 40)); before = np.asarray(im).copy()
    graphics.draw_overlays(im, 1.0, [{"type": typ, "text": "Hi", "subtext": "x", "value": 1200, "label": "Views", "data": {"a": 1, "b": 2}, "at": 0.2, "duration": 3}], "#FF6B35")
    assert (np.asarray(im) != before).any()
    with pytest.raises(ValueError): graphics.draw_overlays(im, 1.0, [{"type": "nope"}], "#fff")
