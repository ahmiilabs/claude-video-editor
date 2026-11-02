#!/usr/bin/env python3
"""Quality-assurance framework for rendered videos.

  python scripts/qa_check.py out.mp4 [--no-audio] [--aspect 9:16] [--expect-res 720x1280] [--expect-fps 24]
                             [--expect-duration 21 --tolerance 0.6] [--report qa.json] [--sidecar out.mp4.render.json]

Every check yields PASS / FAIL / WARN / SKIP. A FAILED *mandatory* check -> exit code 1.
If `<video>.render.json` (written by reel.py) exists, it supplies the expectations: duration, scene boundaries,
caption windows, voice windows, assets. Backward compatible: check(path, expect_audio, expect_aspect) -> bool.
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from common import ffprobe, find_binary, run

ASPECTS = {"9:16": 9 / 16, "1:1": 1.0, "16:9": 16 / 9, "4:5": 4 / 5}

def _ff(args, timeout=300):
    r = subprocess.run([find_binary("ffmpeg", required=True), "-hide_banner", "-nostats", *args], capture_output=True, text=True, errors="replace", timeout=timeout)
    return r.returncode, r.stderr

def _res(name, status, detail, mandatory=True, value=None):
    return {"name": name, "status": status, "mandatory": mandatory, "detail": detail, "value": value}

def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))

# ---------------------------------------------------------------- individual checks
def check_integrity(path):
    if not os.path.exists(path): return _res("file_integrity", "fail", "output file does not exist")
    size = os.path.getsize(path)
    if size < 2000: return _res("file_integrity", "fail", f"file is only {size} bytes", value=size)
    try: ffprobe(path)
    except Exception as e: return _res("file_integrity", "fail", f"ffprobe cannot read the file: {str(e)[:120]}")
    code, err = _ff(["-v", "error", "-i", str(path), "-f", "null", "-"])
    bad = [l for l in err.splitlines() if l.strip()]
    if code != 0 or bad:
        return _res("file_integrity", "fail", f"decode errors ({len(bad)}): {bad[0][:140] if bad else 'non-zero exit'}", value=len(bad))
    return _res("file_integrity", "pass", f"{size / 1024:.0f} KB, decodes cleanly", value=size)

def check_streams(info, expect_audio):
    vs = [s for s in info["streams"] if s["codec_type"] == "video"]; au = [s for s in info["streams"] if s["codec_type"] == "audio"]
    out = [_res("video_stream", "pass" if vs else "fail", "video stream present" if vs else "no video stream")]
    if expect_audio:
        out.append(_res("audio_stream", "pass" if au else "fail", "audio stream present" if au else "audio expected but missing"))
    else:
        out.append(_res("audio_stream", "skip", "audio not expected", mandatory=False))
    return out, vs, au

def check_geometry(vs, aspect, res, fps):
    out = []
    if not vs: return out
    w, h = int(vs[0]["width"]), int(vs[0]["height"])
    if aspect:
        want = ASPECTS.get(aspect)
        if want is None: out.append(_res("aspect_ratio", "warn", f"unknown aspect '{aspect}'", False))
        else:
            ok = abs(w / h - want) < 0.02
            out.append(_res("aspect_ratio", "pass" if ok else "fail", f"{w}x{h} ({w / h:.3f}) vs expected {aspect}", value=f"{w}x{h}"))
    if res:
        ok = (w, h) == tuple(res)
        out.append(_res("resolution", "pass" if ok else "fail", f"{w}x{h} vs expected {res[0]}x{res[1]}", value=f"{w}x{h}"))
    else:
        out.append(_res("resolution", "pass" if w % 2 == 0 and h % 2 == 0 and min(w, h) >= 240 else "fail", f"{w}x{h} (even, >=240px)", value=f"{w}x{h}"))
    num, den = (vs[0].get("avg_frame_rate") or "0/1").split("/"); real = float(num) / float(den or 1) if float(den or 1) else 0
    if fps:
        out.append(_res("frame_rate", "pass" if abs(real - fps) < 0.6 else "fail", f"{real:.2f} fps vs expected {fps}", value=round(real, 2)))
    else:
        out.append(_res("frame_rate", "pass" if 10 <= real <= 120 else "fail", f"{real:.2f} fps", value=round(real, 2)))
    return out

def check_duration(info, expect, tol, minimum=1.0):
    dur = float(info["format"].get("duration") or 0)
    if expect:
        ok = abs(dur - expect) <= tol
        return _res("duration", "pass" if ok else "fail", f"{dur:.2f}s vs expected {expect:.2f}s (±{tol}s)", value=round(dur, 2)), dur
    return _res("duration", "pass" if dur >= minimum else "fail", f"{dur:.2f}s (min {minimum}s)", value=round(dur, 2)), dur

def check_video_signal(path, dur, boundaries, max_black_ratio=0.15, max_frozen_ratio=0.6):
    """blackdetect + freezedetect in one pass."""
    code, err = _ff(["-v", "info", "-i", str(path), "-an", "-vf", "blackdetect=d=0.12:pix_th=0.10:pic_th=0.98,freezedetect=n=-55dB:d=1.5", "-f", "null", "-"], timeout=600)
    blacks = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", err)]
    fz_s = [float(x) for x in re.findall(r"freeze_start: ([\d.]+)", err)]; fz_e = [float(x) for x in re.findall(r"freeze_end: ([\d.]+)", err)]
    frozen = []
    for i, s in enumerate(fz_s):
        frozen.append((s, fz_e[i] if i < len(fz_e) else dur))
    out = []
    bt = sum(b - a for a, b in blacks); ratio = bt / dur if dur else 0
    near = [(a, b) for a, b in blacks for bd in boundaries if _overlap(a, b, bd - 0.35, bd + 0.35) > 0]
    if near:
        out.append(_res("black_frames", "fail", f"black frames at scene boundary ~{near[0][0]:.2f}s (transition glitch)", value=len(blacks)))
    elif ratio > max_black_ratio:
        out.append(_res("black_frames", "fail", f"{ratio * 100:.0f}% of the video is black ({len(blacks)} segment(s))", value=round(ratio, 3)))
    elif blacks:
        out.append(_res("black_frames", "warn", f"{len(blacks)} short black segment(s), {bt:.2f}s total (could be intentional)", False, round(ratio, 3)))
    else:
        out.append(_res("black_frames", "pass", "no black frames", value=0))
    ft = sum(b - a for a, b in frozen); fr = ft / dur if dur else 0
    if fr > max_frozen_ratio:
        out.append(_res("frozen_frames", "fail", f"{fr * 100:.0f}% of the video is frozen (identical frames)", value=round(fr, 3)))
    elif frozen:
        out.append(_res("frozen_frames", "warn", f"{len(frozen)} static segment(s), {ft:.1f}s total (normal for still scenes)", False, round(fr, 3)))
    else:
        out.append(_res("frozen_frames", "pass", "no frozen segments", value=0))
    return out

def check_audio_signal(path, dur, voice_windows, expect_audio):
    if not expect_audio: return []
    code, err = _ff(["-v", "info", "-i", str(path), "-vn", "-af", "silencedetect=noise=-45dB:d=1.2,volumedetect", "-f", "null", "-"], timeout=600)
    mx = re.search(r"max_volume: (-?[\d.]+) dB", err); mean = re.search(r"mean_volume: (-?[\d.]+) dB", err)
    out = []
    if not mx:
        return [_res("audio_signal", "fail", "could not analyse audio (corrupt or empty audio stream)")]
    peak = float(mx.group(1)); out.append(_res("audio_clipping", "fail" if peak >= -0.05 else "pass", f"peak {peak:.1f} dBFS" + (" (clipping!)" if peak >= -0.05 else ""), value=peak))
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", err)]; ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    sil = [(max(0, s), (ends[i] if i < len(ends) else dur)) for i, s in enumerate(starts)]
    total = sum(b - a for a, b in sil); ratio = total / dur if dur else 0
    bad_voice = [(a, b) for a, b in sil for v0, v1 in voice_windows if _overlap(a, b, v0, v1) >= 1.2]
    if mean and float(mean.group(1)) < -60:
        out.append(_res("silence", "fail", f"audio is essentially silent (mean {mean.group(1)} dB)", value=float(mean.group(1))))
    elif bad_voice:
        out.append(_res("silence", "fail", f"unexpected silence during narration at {bad_voice[0][0]:.1f}-{bad_voice[0][1]:.1f}s", value=len(bad_voice)))
    elif ratio > 0.6:
        out.append(_res("silence", "fail", f"{ratio * 100:.0f}% of the audio is silent", value=round(ratio, 3)))
    elif ratio > 0.3:
        out.append(_res("silence", "warn", f"{ratio * 100:.0f}% silence", False, round(ratio, 3)))
    else:
        out.append(_res("silence", "pass", f"{ratio * 100:.0f}% silence, mean {mean.group(1) if mean else '?'} dB", value=round(ratio, 3)))
    return out

def check_assets(side):
    missing = [a for a in side.get("assets", []) if a and not os.path.exists(a)]
    if "assets" not in side: return _res("missing_assets", "skip", "no asset list in sidecar", False)
    return _res("missing_assets", "fail" if missing else "pass", f"missing: {missing[:3]}" if missing else f"{len(side['assets'])} asset(s) present", value=len(missing))

def check_captions(path, side, w, h):
    caps = side.get("captions") or {}
    if not caps.get("expected"): return _res("captions", "skip", "captions not expected", False)
    ass = caps.get("ass")
    if ass and (not os.path.exists(ass) or "Dialogue:" not in Path(ass).read_text(encoding="utf-8", errors="ignore")):
        # the .ass lives in the (cleaned) work dir; trust the recorded dialogue count instead
        if not caps.get("dialogues"): return _res("captions", "fail", "caption file missing/empty and no dialogue recorded")
    wins = caps.get("windows") or []
    if not wins: return _res("captions", "fail", "captions expected but no caption windows were recorded")
    if caps.get("verified") is not None:                      # evidence recorded at render time: caption-less render vs final file
        v = float(caps["verified"])
        return _res("captions", "pass" if v >= 0.6 else "fail", f"burned-in captions verified in {v * 100:.0f}% of sampled caption windows (render-time comparison)", value=v)
    try:
        import cv2, numpy as np
    except ImportError:
        return _res("captions", "skip", "opencv not installed: cannot verify burned-in captions", False)
    cap = cv2.VideoCapture(str(path)); picks = wins[:: max(1, len(wins) // 8)][:8]; hits = 0; zone = caps.get("zone", [0.55, 0.88]); k = np.ones((7, 7), np.uint8)
    for w0, w1 in picks:
        cap.set(cv2.CAP_PROP_POS_MSEC, (w0 + (w1 - w0) * 0.5) * 1000); ok, fr = cap.read()
        if not ok: continue
        H = fr.shape[0]; reg = fr[int(H * zone[0]):int(H * zone[1])]
        white = (reg.min(axis=2) > 235); black = cv2.dilate((reg.max(axis=2) < 45).astype(np.uint8), k).astype(bool)
        gold = (reg[..., 2] > 225) & (reg[..., 1] > 170) & (reg[..., 0] < 90)
        if ((white | gold) & black).mean() > 0.0015: hits += 1          # caption glyphs are bright AND outlined in black
    cap.release(); frac = hits / max(1, len(picks))
    return _res("captions", "pass" if frac >= 0.6 else "fail", f"outlined caption text found in {hits}/{len(picks)} sampled caption windows (heuristic: no render-time evidence)", value=round(frac, 2))

def check_motion(path):
    try:
        import cv2
    except ImportError:
        return _res("motion", "skip", "opencv not installed", False)
    cap = cv2.VideoCapture(str(path)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); prev = None; diffs = []
    for k in range(10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, n - 1) * k / 9)); ok, fr = cap.read()
        if not ok: continue
        g = cv2.cvtColor(cv2.resize(fr, (96, 160)), cv2.COLOR_BGR2GRAY).astype("float32")
        if prev is not None: diffs.append(float(abs(g - prev).mean()))
        prev = g
    cap.release()
    ok = bool(diffs) and max(diffs) > 1.0
    return _res("motion", "pass" if ok else "fail", f"max change between samples {max(diffs) if diffs else 0:.1f}", True, round(max(diffs) if diffs else 0, 1))

# ---------------------------------------------------------------- orchestration
def run_qa(path, expect_audio=True, aspect=None, res=None, fps=None, expect_duration=None, tolerance=0.6, sidecar=None, report_path=None, quiet=False):
    t0 = time.time(); path = str(path)
    sc = sidecar or (path + ".render.json"); side = {}
    if os.path.exists(sc):
        try: side = json.load(open(sc, encoding="utf-8"))
        except Exception: side = {}
    if side.get("status") == "failed":
        results = [_res("render_status", "fail", f"render failed: {side.get('error', 'unknown')}")]
        return _finish(path, results, report_path, t0, quiet)
    if expect_duration is None: expect_duration = side.get("duration")
    if aspect is None: aspect = side.get("aspect")
    if res is None and side.get("size"): res = tuple(side["size"])
    if fps is None: fps = side.get("fps")
    if "expect_audio" in side and expect_audio is True: expect_audio = side["expect_audio"]
    results = [check_integrity(path)]
    if results[0]["status"] == "fail":
        return _finish(path, results, report_path, t0, quiet)
    info = ffprobe(path); st, vs, au = check_streams(info, expect_audio); results += st
    results += check_geometry(vs, aspect, res, fps)
    dres, dur = check_duration(info, expect_duration, tolerance); results.append(dres)
    if vs:
        results += check_video_signal(path, dur, side.get("boundaries", []))
        results.append(check_motion(path))
    if au or expect_audio:
        results += check_audio_signal(path, dur, side.get("voice_windows", []), expect_audio and bool(au) and side.get("audio_signal_expected", True))
    if side: results.append(check_assets(side))
    if side and vs: results.append(check_captions(path, side, int(vs[0]["width"]), int(vs[0]["height"])))
    return _finish(path, results, report_path, t0, quiet)

def _finish(path, results, report_path, t0, quiet):
    failed = [r for r in results if r["status"] == "fail" and r["mandatory"]]
    report = {"file": str(path), "passed": not failed, "failed_checks": [r["name"] for r in failed],
              "warnings": [r["name"] for r in results if r["status"] == "warn"], "seconds": round(time.time() - t0, 1), "checks": results}
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not quiet:
        for r in results:
            print(f"{r['status'].upper():5s} {r['name']:16s} {r['detail']}")
        print("QA " + ("PASSED" if report["passed"] else "FAILED: " + ", ".join(report["failed_checks"])))
    return report

def check(path, expect_audio=True, expect_aspect="9:16"):
    """Backward-compatible helper used by older callers: True when every mandatory check passes."""
    return run_qa(path, expect_audio=expect_audio, aspect=expect_aspect)["passed"]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video"); ap.add_argument("--no-audio", action="store_true"); ap.add_argument("--aspect")
    ap.add_argument("--expect-res"); ap.add_argument("--expect-fps", type=float); ap.add_argument("--expect-duration", type=float)
    ap.add_argument("--tolerance", type=float, default=0.6); ap.add_argument("--report"); ap.add_argument("--sidecar")
    a = ap.parse_args()
    res = tuple(int(x) for x in a.expect_res.lower().split("x")) if a.expect_res else None
    rep = run_qa(a.video, expect_audio=not a.no_audio, aspect=a.aspect, res=res, fps=a.expect_fps, expect_duration=a.expect_duration,
                 tolerance=a.tolerance, sidecar=a.sidecar, report_path=a.report)
    sys.exit(0 if rep["passed"] else 1)

if __name__ == "__main__":
    main()
