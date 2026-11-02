import json, os, subprocess, sys
from pathlib import Path
import numpy as np, pytest
from conftest import SCRIPTS, have_models, ff
import vision as V, effects as E, qa_check

def run_fx(clip, out, *args):
    r = subprocess.run([sys.executable, str(SCRIPTS / "effects.py"), str(clip), "--out", str(out), *args], capture_output=True, text=True); return r

def fake_ctx(alpha, **kw):
    import argparse
    a = argparse.Namespace(title_text="HI", title_color="#FFFFFF", title_size=0.5, title_y=0.5, strength=0.8, bg="gradient:#ff0000:#0000ff", bg_sigma=10, feather=0, push=1.2, aspect="1:1")
    return {"args": a, "_alpha": alpha, "orig": np.zeros(alpha.shape + (3,), np.uint8), "matter": None, "duration": 2.0, "subject": None, "cam": V.Smoother(), **kw}

# ---------------------------------------------------------------- compositing logic (no model needed)
def test_text_is_hidden_behind_the_person_and_visible_elsewhere():
    H, W = 200, 400; alpha = np.zeros((H, W), np.float32); alpha[:, :200] = 1.0            # left half = "person"
    fr = np.full((H, W, 3), 60, np.uint8); out = E.fx_text_behind(fr, 5.0, fake_ctx(alpha))
    left, right = out[:, :200], out[:, 200:]
    assert (left == 60).all(), "text must NOT be drawn over the person (depth ordering)"; assert (right > 200).any(), "text must be visible over the background"

def test_soft_edges_blend_proportionally():
    fg = np.full((4, 4, 3), 200, np.uint8); bg = np.zeros((4, 4, 3), np.uint8); a = np.full((4, 4), 0.5, np.float32); assert abs(int(V.composite(fg, a, bg)[0, 0, 0]) - 100) <= 1

def test_bg_replace_keeps_person_and_replaces_background():
    H, W = 100, 100; alpha = np.zeros((H, W), np.float32); alpha[:, :50] = 1; fr = np.full((H, W, 3), 123, np.uint8); out = E.fx_bg_replace(fr, 0, fake_ctx(alpha))
    assert (out[:, :50] == 123).all() and not (out[:, 60:] == 123).all()

def test_smart_spotlight_dims_only_background():
    H, W = 100, 100; alpha = np.zeros((H, W), np.float32); alpha[:, :50] = 1; fr = np.full((H, W, 3), 200, np.uint8); out = E.fx_smart_spotlight(fr, 0, fake_ctx(alpha))
    assert out[:, :50].mean() >= 200 and out[:, 60:].mean() < 80

def test_refine_alpha_range_and_edge_softening():
    a = np.zeros((60, 60), np.float32); a[:, 30:] = 1; r = V.refine_alpha(a, erode=1, feather=2.0); assert r.min() >= 0 and r.max() <= 1 and 0.05 < r[30, 30] < 0.95

def test_smoother_limits_speed():
    s = V.Smoother(k=1.0, max_step=0.01); s((0.0, 0.0), scale=100); x, _ = s((1000.0, 0.0), scale=100); assert x <= 1.0 + 1e-6

# ---------------------------------------------------------------- face tracking logic (detector injected: tests tracking, not Haar accuracy)
def test_tracker_ignores_small_jitter_keeps_ids_and_holds_then_drops(monkeypatch):
    t = V.SubjectTracker(every=1, fps=10, hold=0.5); frame = np.zeros((400, 400, 3), np.uint8); seq = iter([[(100, 100, 80, 80)], [(102, 101, 80, 80)], [(99, 100, 81, 80)], [(101, 99, 80, 80)]] + [[]] * 3 + [[(100, 100, 80, 80)]] + [[]] * 12)
    monkeypatch.setattr(t, "_detect", lambda rgb: next(seq)); boxes = []
    for _ in range(4): t.update(frame); boxes.append(t.faces()[0]["x"])
    assert max(boxes) - min(boxes) < 0.5, "sub-dead-zone jitter must not move the box"; fid = t.faces()[0]["id"]
    for _ in range(3): t.update(frame)
    assert t.faces() and t.faces()[0]["held"], "box is held briefly when detection is lost"
    t.update(frame); assert t.faces()[0]["id"] == fid and not t.faces()[0]["held"]
    for _ in range(12): t.update(frame)
    assert t.faces() == [], "track dropped after the hold time"

def test_tracker_supports_multiple_faces(monkeypatch):
    t = V.SubjectTracker(every=1, max_faces=3, fps=10); monkeypatch.setattr(t, "_detect", lambda rgb: [(50, 50, 90, 90), (250, 60, 60, 60)]); t.update(np.zeros((400, 400, 3), np.uint8)); f = t.faces()
    assert len(f) == 2 and {x["id"] for x in f} == {1, 2} and t.primary()["w"] == 90

def test_subject_box_from_matte():
    a = np.zeros((200, 100), np.float32); a[40:190, 30:70] = 1; b = V.SubjectTracker.box_from_alpha(a); assert b and abs(b["top"] - 40) <= 1 and 40 < b["cx"] < 60

def test_backgrounds_parse_and_reject_unknown(tmp_path):
    fr = np.full((40, 60, 3), 90, np.uint8)
    for spec in ("blur:5", "color:#112233", "gradient:#ff0000:#0000ff"): assert V.Background(spec, 60, 40).frame(fr).shape == (40, 60, 3)
    with pytest.raises(V.VisionError): V.Background("hologram", 60, 40)
    with pytest.raises(V.VisionError, match="not found"): V.Background("image:/no/such.png", 60, 40)

# ---------------------------------------------------------------- real model (skipped when not installed)
@pytest.mark.skipif(not have_models("matting"), reason="matting model not installed (python scripts/setup_models.py --matting)")
def test_matter_output_contract_and_temporal_state():
    m = V.Matter(); fr = (np.random.RandomState(1).rand(360, 202, 3) * 255).astype(np.uint8)
    for _ in range(3): a = m.matte(fr); assert a.shape == (360, 202) and a.dtype == np.float32 and 0 <= a.min() and a.max() <= 1
    m.matte(np.zeros((480, 270, 3), np.uint8))                      # geometry change must not crash (state resets)

@pytest.mark.skipif(not have_models("matting") or not os.environ.get("AIVE_TEST_PERSON_CLIP"), reason="set AIVE_TEST_PERSON_CLIP=<clip with a person> and install the matting model")
def test_real_person_is_segmented_and_tracked():
    import cv2; cap = cv2.VideoCapture(os.environ["AIVE_TEST_PERSON_CLIP"]); m = V.Matter(); trk = V.SubjectTracker(fps=30); means = []; faces = 0
    for i in range(30):
        ok, bgr = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); a = V.refine_alpha(m.matte(rgb)); trk.update(rgb, a); means.append(a.mean()); faces += trk.primary() is not None and trk.primary()["source"] == "face"
    assert 0.08 < np.mean(means) < 0.7 and faces >= 20

# ---------------------------------------------------------------- effect CLI on a synthetic clip (every existing + new non-matte effect)
@pytest.mark.parametrize("args", [["--effect", "spotlight"], ["--effect", "punch-zoom", "--every", "0.5"], ["--effect", "color-grade", "--look", "warm"], ["--effect", "color-grade", "--look", "bw"],
                                   ["--effect", "color-grade", "--look", "cinematic"], ["--effect", "glass-widgets"], ["--effect", "split-chart", "--title", "T"], ["--effect", "captions", "--text", "hello brave new world"],
                                   ["--effect", "cinematic-zoom"], ["--effect", "spotlight,color-grade,punch-zoom"]])
def test_effect_produces_valid_video_with_audio_preserved(tmp_path, media, args):
    out = tmp_path / "o.mp4"; r = run_fx(media / "clip.mp4", out, *args); assert r.returncode == 0, r.stderr[-400:]
    rep = qa_check.run_qa(str(out), aspect="9:16", quiet=True); bad = [c for c in rep["failed_checks"] if c not in ("silence",)]; assert not bad, rep["failed_checks"]

def test_face_crop_changes_aspect_and_must_be_first(tmp_path, media):
    out = tmp_path / "sq.mp4"; assert run_fx(media / "clip.mp4", out, "--effect", "face-crop", "--aspect", "1:1").returncode == 0
    w, h = (int(x) for x in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.strip().split(",")); assert w == h
    r = run_fx(media / "clip.mp4", tmp_path / "x.mp4", "--effect", "color-grade,face-crop"); assert r.returncode != 0 and "first" in r.stderr

def test_unknown_effect_and_bad_input_fail_clearly(tmp_path, media):
    assert "unknown effect" in run_fx(media / "clip.mp4", tmp_path / "x.mp4", "--effect", "sparkle").stderr
    assert "not found" in run_fx(tmp_path / "nope.mp4", tmp_path / "x.mp4", "--effect", "spotlight").stderr

def test_matte_effect_without_model_fails_honestly(tmp_path, media, monkeypatch):
    env = {**os.environ, "AIVE_MODEL_DIR": str(tmp_path / "empty_models")}
    r = subprocess.run([sys.executable, str(SCRIPTS / "effects.py"), str(media / "clip.mp4"), "--effect", "text-behind", "--title-text", "X", "--out", str(tmp_path / "o.mp4")], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "setup_models.py --matting" in r.stderr and not (tmp_path / "o.mp4").exists()

@pytest.mark.skipif(not have_models("matting"), reason="matting model not installed")
@pytest.mark.parametrize("args", [["--effect", "bg-blur"], ["--effect", "bg-replace", "--bg", "gradient:#ff6b35:#1a1a2e"], ["--effect", "text-behind", "--title-text", "HELLO"], ["--effect", "smart-spotlight"]])
def test_matte_effects_run_end_to_end(tmp_path, media, args):
    out = tmp_path / "m.mp4"; r = run_fx(media / "clip.mp4", out, "--max-width", "360", *args); assert r.returncode == 0, r.stderr[-400:]; assert qa_check.run_qa(str(out), quiet=True)["checks"]

@pytest.mark.skipif(not have_models("matting"), reason="matting model not installed")
def test_transparent_export_has_real_alpha_channel(tmp_path, media):
    out = tmp_path / "a.mov"; assert run_fx(media / "clip.mp4", out, "--effect", "bg-remove", "--alpha-format", "mov", "--max-width", "360").returncode == 0
    pf = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=pix_fmt,codec_name", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.strip(); assert "qtrle" in pf and "argb" in pf
    d = tmp_path / "png"; assert run_fx(media / "clip.mp4", d, "--effect", "bg-remove", "--alpha-format", "png", "--max-width", "360").returncode == 0
    from PIL import Image; fr = sorted(d.glob("*.png")); assert len(fr) >= 60 and Image.open(fr[0]).mode == "RGBA"
