"""MOCKED provider tests: they verify request format, parsing, retries, validation, fallbacks and honest failure.
They do NOT verify that any real service (OpenAI, Anthropic, a video vendor...) accepts these requests with a real key."""
import json, os
from pathlib import Path
import pytest
import config, scriptwriter as SW, imagegen as IG, videogen as VG
from assets import AssetManager, AssetError, validate_image, validate_video
from net import ProviderError

GOOD = {"title": "Chai", "hook": "Why so pricey?", "music_suggestion": "calm lo-fi", "visual_style_anchor": "flat illustration, warm palette",
        "scenes": [{"text": "Why is chai pricey?", "narration": "Why is chai so pricey on trains?", "voice_text": "ट्रेन में चाय इतनी महंगी क्यों?", "keywords": ["chai"], "duration_estimate_seconds": 4, "visual_description": "a train", "image_prompt": "a steaming cup on a train seat", "video_prompt": "slow push-in on cup", "transition": "whip", "sound_effect": "whoosh"},
                   {"text": "Follow for more", "narration": "Follow for more.", "voice_text": "और जानने के लिए फॉलो करो।", "duration_estimate_seconds": 3, "image_prompt": "a phone with a follow button", "transition": "fade"}]}

@pytest.fixture()
def llm_env(clean_env, mock_server):
    clean_env.setenv("AIVE_LLM_PROVIDER", "openai"); clean_env.setenv("AIVE_LLM_BASE_URL", mock_server.url + "/v1"); clean_env.setenv("AIVE_LLM_MODEL", "m"); clean_env.setenv("AIVE_LLM_API_KEY", "sk-testkey123456789"); clean_env.setenv("AIVE_RETRIES", "1")
    return mock_server

# ---------------------------------------------------------------- script writer
def test_no_llm_gives_offline_workflow_not_a_fake_script(clean_env):
    with pytest.raises(SW.ScriptError, match="Offline/manual workflow"): SW.generate("topic")

def test_openai_compatible_script_generation_sends_auth_and_builds_spec(llm_env):
    llm_env.llm_replies = ["```json\n" + json.dumps(GOOD) + "\n```"]; spec = SW.generate("chai on trains", "roman-urdu", 30, "reels", "energetic", "education")
    path, headers, body = llm_env.requests[0]; assert path == "/v1/chat/completions" and headers["authorization"] == "Bearer sk-testkey123456789"; assert json.loads(body)["model"] == "m"
    assert spec["aspect"] == "9:16" and spec["voice"]["lang"] == "hi" and spec["scenes"][0]["voice_text"] and spec["scenes"][0]["image_prompt"] and spec["scenes"][0]["transition"] == "whip"
    from reel import validate_spec; validate_spec({k: v for k, v in spec.items()})                         # the output integrates directly with the renderer

def test_anthropic_adapter_uses_messages_api(clean_env, mock_server):
    for k, v in {"AIVE_LLM_PROVIDER": "anthropic", "AIVE_LLM_BASE_URL": mock_server.url + "/v1", "AIVE_LLM_MODEL": "m", "AIVE_LLM_API_KEY": "sk-ant-abcdefghij"}.items(): clean_env.setenv(k, v)
    mock_server.llm_replies = [json.dumps(GOOD)]; SW.generate("t", "roman-urdu"); path, headers, body = mock_server.requests[0]
    assert path.endswith("/messages") and headers["x-api-key"] == "sk-ant-abcdefghij" and "anthropic-version" in headers and "system" in json.loads(body)

def test_invalid_model_json_is_retried_with_feedback_then_fails_honestly(llm_env):
    llm_env.llm_replies = ["not json at all", json.dumps({"scenes": []}), json.dumps(GOOD)]; assert SW.generate("t", "roman-urdu")["title"] == "Chai"; assert llm_env.llm_calls == 3
    llm_env.llm_replies = ["nope"] * 3; llm_env.llm_calls = 0
    with pytest.raises(SW.ScriptError, match="failed to produce a valid script"): SW.generate("t", "roman-urdu")

def test_roman_urdu_requires_devanagari_voice_text():
    bad = json.loads(json.dumps(GOOD)); del bad["scenes"][0]["voice_text"]
    with pytest.raises(SW.ScriptError, match="voice_text"): SW.to_spec(bad, "roman-urdu")

@pytest.mark.parametrize("lang", ["en", "ur", "roman-urdu", "hi", "multi"])
def test_prompt_supports_all_languages(lang):
    s, u = SW.build_prompt("t", language=lang); assert "JSON" in s and ("Devanagari" in u) == (lang in ("ur", "roman-urdu", "hi"))

def test_manual_import_workflow(tmp_path):
    f = tmp_path / "answer.json"; f.write_text("Here you go:\n" + json.dumps(GOOD)); spec = SW.to_spec(SW.extract_json(f.read_text()), "roman-urdu"); assert len(spec["scenes"]) == 2

# ---------------------------------------------------------------- image generation + assets
@pytest.fixture()
def img_env(clean_env, mock_server):
    for k, v in {"AIVE_IMAGE_PROVIDER": "openai", "AIVE_IMAGE_BASE_URL": mock_server.url + "/v1", "AIVE_IMAGE_MODEL": "img", "AIVE_IMAGE_API_KEY": "sk-imgkey123456789", "AIVE_RETRIES": "2"}.items(): clean_env.setenv(k, v)
    return mock_server

def spec2(): return {"title": "t", "aspect": "9:16", "script_meta": {"visual_style_anchor": "flat warm illustration"}, "scenes": [dict(s) for s in GOOD["scenes"]]}

def test_image_generation_saves_validates_and_anchors_style(img_env, tmp_path):
    sp = spec2(); rep = IG.generate_scene_images(sp, tmp_path / "a", sleep=0); assert rep["generated"] == [1, 2] and not rep["failed"]
    for s in sp["scenes"]: assert Path(s["image"]).exists() and validate_image(s["image"])
    sent = json.loads(img_env.requests[0][2]); assert sent["prompt"].startswith("flat warm illustration") and sent["size"] == "1024x1792" and sent["model"] == "img"
    names = [Path(s["image"]).name for s in sp["scenes"]]; assert len(set(names)) == 2 and names[0].startswith("s01_image_")
    m = json.loads((tmp_path / "a" / "manifest.json").read_text()); assert all(i["status"] == "ok" and i["sha256"] for i in m)

def test_image_retries_transient_errors(img_env, tmp_path):
    img_env.image_fail_first = 2; rep = IG.generate_scene_images(spec2(), tmp_path / "a", only={1}, sleep=0); assert rep["generated"] == [1] and img_env.image_calls == 3

def test_image_auth_failure_is_reported_not_retried_and_never_faked(img_env, tmp_path):
    img_env.image_fail_first = 99; sp = spec2(); rep = IG.generate_scene_images(sp, tmp_path / "a", sleep=0)
    assert len(rep["failed"]) == 2 and "image" not in sp["scenes"][0] and img_env.image_calls == 2          # 401 is not retryable
    assert "sk-secretsecret123456" not in json.dumps(rep) and "sk-imgkey123456789" not in json.dumps(rep)
    assert [i["status"] for i in json.loads((tmp_path / "a" / "manifest.json").read_text())] == ["failed", "failed"]

def test_corrupt_image_from_provider_is_rejected(img_env, tmp_path):
    img_env.image_bad = True; sp = spec2(); rep = IG.generate_scene_images(sp, tmp_path / "a", only={1}, sleep=0); assert rep["failed"] and "image" not in sp["scenes"][0] and not list((tmp_path / "a").glob("*.png"))

def test_no_image_provider_fails_clearly(clean_env, tmp_path):
    rep = IG.generate_scene_images(spec2(), tmp_path / "a", sleep=0); assert len(rep["failed"]) == 2 and "no image provider" in rep["failed"][0]["error"]

def test_user_supplied_images_are_registered_not_regenerated(img_env, tmp_path, media):
    sp = spec2(); sp["scenes"][0]["image"] = str(media / "photo.png"); rep = IG.generate_scene_images(sp, tmp_path / "a", sleep=0)
    assert rep["user"] == [1] and rep["generated"] == [2] and img_env.image_calls == 1 and Path(sp["scenes"][0]["image"]).parent == tmp_path / "a"

def test_prompt_cache_reuses_existing_image(img_env, tmp_path):
    sp = spec2(); IG.generate_scene_images(sp, tmp_path / "a", sleep=0); calls = img_env.image_calls; sp2 = spec2(); rep = IG.generate_scene_images(sp2, tmp_path / "a", sleep=0); assert rep["reused"] == [1, 2] and img_env.image_calls == calls

def test_asset_validation_rules(tmp_path, media):
    with pytest.raises(AssetError): validate_image(tmp_path / "missing.png")
    (tmp_path / "bad.png").write_bytes(b"x" * 5000)
    with pytest.raises(AssetError, match="corrupt"): validate_image(tmp_path / "bad.png")
    with pytest.raises(AssetError, match="aspect"): validate_image(media / "photo.png", aspect="16:9")
    with pytest.raises(AssetError): validate_video(media / "photo.png"); assert validate_video(media / "clip.mp4")["duration"] > 2

# ---------------------------------------------------------------- video generation
VCFG = {"submit": {"method": "POST", "path": "/generate", "body": {"model": "{model}", "prompt": "{prompt}", "duration": "{duration}", "aspect_ratio": "{aspect}", "image": "{image_b64}"}, "id_path": "id"},
        "status": {"path": "/jobs/{id}", "state_path": "status", "done": ["completed"], "failed": ["failed"], "result_url_path": "output.url"}, "limits": {"durations": [4, 8], "aspects": ["9:16", "16:9"]}, "poll_seconds": 0.05, "timeout_seconds": 20}

@pytest.fixture()
def vid_env(clean_env, mock_server, tmp_path):
    cfg = tmp_path / "vcfg.json"; cfg.write_text(json.dumps(VCFG))
    for k, v in {"AIVE_VIDEO_PROVIDER": "http", "AIVE_VIDEO_BASE_URL": mock_server.url, "AIVE_VIDEO_MODEL": "vid", "AIVE_VIDEO_API_KEY": "sk-vidkey123456789", "AIVE_VIDEO_CONFIG": str(cfg), "AIVE_RETRIES": "0"}.items(): clean_env.setenv(k, v)
    return mock_server

def test_async_video_flow_submit_poll_download_validate_integrate(vid_env, tmp_path, media):
    sp = spec2(); sp["scenes"][0]["image"] = str(media / "photo.png"); sp["scenes"] = sp["scenes"][:1]; sp["scenes"][0]["duration"] = 5
    rep = VG.generate_scene_videos(sp, tmp_path / "a", only={1}, poll_override=0.05); assert rep["generated"] == [1] and not rep["failed"]
    assert Path(sp["scenes"][0]["video"]).exists() and validate_video(sp["scenes"][0]["video"])["width"] == 720
    sub = json.loads(vid_env.requests[0][2]); assert sub["duration"] == 4 and sub["aspect_ratio"] == "9:16" and sub["image"] and sub["model"] == "vid"        # 5s snapped to a supported 4s
    assert list(json.loads((tmp_path / "a" / "jobs.json").read_text()).values())[0]["status"] == "downloaded"
    from reel import validate_spec; validate_spec(sp)

def test_unsupported_aspect_fails_before_any_request(vid_env, tmp_path):
    sp = spec2(); sp["aspect"] = "1:1"; rep = VG.generate_scene_videos(sp, tmp_path / "a", only={1}); assert rep["failed"] and "aspect" in rep["failed"][0]["error"] and not [r for r in vid_env.requests if r[0] == "/generate"]

def test_provider_failure_falls_back_to_kenburns_when_image_exists(vid_env, tmp_path, media):
    vid_env.video_mode = "fail"; sp = spec2(); sp["scenes"][0]["image"] = str(media / "photo.png"); rep = VG.generate_scene_videos(sp, tmp_path / "a", only={1}, poll_override=0.05)
    assert rep["failed"] and rep["fallback_kenburns"] == [1] and "video" not in sp["scenes"][0] and sp["scenes"][0]["image"]

def test_corrupt_download_is_marked_invalid(vid_env, tmp_path):
    vid_env.video_mode = "corrupt"; sp = spec2(); rep = VG.generate_scene_videos(sp, tmp_path / "a", only={1}, poll_override=0.05); assert rep["failed"] and "video" not in sp["scenes"][0]
    assert list(json.loads((tmp_path / "a" / "jobs.json").read_text()).values())[0]["status"] == "invalid"

def test_no_video_provider_fails_clearly(clean_env, tmp_path):
    rep = VG.generate_scene_videos(spec2(), tmp_path / "a"); assert rep["failed"] and "no video provider" in rep["failed"][0]["error"]

def test_kenburns_fallback_clip_is_real(tmp_path, media):
    out = VG.kenburns_clip(media / "photo.png", tmp_path / "k.mp4", 2.0, "9:16", fps=12); assert validate_video(out)["duration"] > 1.5
