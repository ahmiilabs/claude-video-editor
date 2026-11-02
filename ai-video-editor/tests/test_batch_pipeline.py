import json, os, signal, subprocess, sys, time
from pathlib import Path
import pytest
from conftest import SCRIPTS
import batch, pipeline

BATCH = [sys.executable, str(SCRIPTS / "batch.py")]
def spec(title, n=2, **kw): return {"title": title, "voice": {"mode": "silent"}, "transitions": {"preset": "smooth"}, "scenes": [{"text": f"{title} {i}", "narration": f"scene {i} of {title}", "duration": 1.2} for i in range(n)], **kw}
def write(tmp, name, s): p = tmp / f"{name}.json"; p.write_text(json.dumps(s)); return str(p)
def state(root): return json.loads((Path(root) / "batch.json").read_text())["jobs"]
def wait_for(pred, timeout=60, step=0.2):
    end = time.time() + timeout
    while time.time() < end:
        if pred(): return True
        time.sleep(step)
    return False

def test_parallel_jobs_isolated_outputs_and_summary(tmp_path):
    specs = [write(tmp_path, n, spec(n)) for n in "abc"]; r = subprocess.run([*BATCH, "run", *specs, "--out", str(tmp_path / "b"), "--workers", "2"], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-400:]
    jobs = state(tmp_path / "b"); assert {j["status"] for j in jobs.values()} == {"succeeded"} and len({j["output"] for j in jobs.values()}) == 3
    for j in jobs.values(): assert Path(j["output"]).exists() and j["qa_passed"] is True and Path(j["output"]).parent.name == j["id"]
    s = json.loads((tmp_path / "b" / "summary.json").read_text()); assert s["counts"] == {"succeeded": 3}; assert "| job |" in (tmp_path / "b" / "summary.md").read_text()
    starts = sorted((j["started"], j["finished"]) for j in jobs.values()); assert starts[1][0] < starts[0][1] + 5               # second job started while workers=2 (overlap or immediate follow-up)

def test_invalid_spec_fails_once_without_pointless_retries(tmp_path):
    bad = write(tmp_path, "bad", {"scenes": [{"text": "x", "style": "nope"}]}); good = write(tmp_path, "good", spec("g", 1))
    r = subprocess.run([*BATCH, "run", bad, good, "--out", str(tmp_path / "b"), "--retries", "3"], capture_output=True, text=True); j = state(tmp_path / "b")
    assert r.returncode == 1 and sorted(x["status"] for x in j.values()) == ["failed", "succeeded"]; badjob = [x for x in j.values() if x["title"] == "bad"][0]; assert badjob["attempts"] == 1 and "unknown style" in badjob["error"]

def test_transient_failure_is_retried_then_retry_command_recovers(tmp_path):
    voice = tmp_path / "late.wav"; s = spec("late", 1); s["voice"] = {"mode": "file", "file": str(voice)}
    sp = write(tmp_path, "late", s); r = subprocess.run([*BATCH, "run", sp, "--out", str(tmp_path / "b"), "--retries", "1"], capture_output=True, text=True); j = list(state(tmp_path / "b").values())[0]
    assert j["status"] == "failed" and j["attempts"] == 2 and "(" not in (j["error"] or "")[:0]                       # exit code 1 => retried up to retries+1
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=300:d=3", str(voice)], check=True)
    r = subprocess.run([*BATCH, "retry", str(tmp_path / "b")], capture_output=True, text=True); assert r.returncode == 0, r.stdout[-300:] + r.stderr[-300:]; assert list(state(tmp_path / "b").values())[0]["status"] == "succeeded"

def slow_spec(title): return spec(title, n=14, transitions={"preset": "cinematic"}, quality="hd")

def start_manager(tmp_path, specs, out, extra=()):
    return subprocess.Popen([*BATCH, "run", *specs, "--out", str(out), "--workers", "1", *extra], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def test_cancel_stops_running_process_and_leaves_no_part_files(tmp_path):
    out = tmp_path / "b"; mgr = start_manager(tmp_path, [write(tmp_path, "s", slow_spec("slow"))], out)
    try:
        assert wait_for(lambda: out.exists() and (out / "batch.json").exists() and any(j["status"] == "running" for j in state(out).values()), 40), "job never started"
        time.sleep(1.5); subprocess.run([*BATCH, "cancel", str(out)], check=True, capture_output=True); mgr.wait(timeout=40)
    finally:
        if mgr.poll() is None: mgr.kill()
    j = list(state(out).values())[0]; assert j["status"] == "cancelled" and not list((out / "jobs").glob("*/*.part.mp4")) and not (out / "batch.lock").exists()
    ps = subprocess.run(["pgrep", "-f", "reel.py.*" + j["id"]], capture_output=True, text=True); assert ps.stdout.strip() == "", "render process must be gone"

def test_graceful_shutdown_marks_interrupted_then_resume_completes(tmp_path):
    out = tmp_path / "b"; mgr = start_manager(tmp_path, [write(tmp_path, "s", slow_spec("slow2"))], out)
    try:
        assert wait_for(lambda: out.exists() and (out / "batch.json").exists() and any(j["status"] == "running" for j in state(out).values()), 40)
        time.sleep(1.5); mgr.send_signal(signal.SIGTERM); code = mgr.wait(timeout=40)
    finally:
        if mgr.poll() is None: mgr.kill()
    j = list(state(out).values())[0]; assert code == 130 and j["status"] == "interrupted" and not list((out / "jobs").glob("*/*.part.mp4")) and not (out / "batch.lock").exists()
    # simulate a hard crash too: a stale lock + a job stuck in 'running' must be recovered
    st = json.loads((out / "batch.json").read_text()); jid = list(st["jobs"])[0]; st["jobs"][jid]["status"] = "running"; (out / "batch.json").write_text(json.dumps(st)); (out / "batch.lock").write_text("999999")
    sp = json.loads((out / "jobs" / jid / "spec.json").read_text()); sp["scenes"] = sp["scenes"][:2]; sp.pop("quality", None); (out / "jobs" / jid / "spec.json").write_text(json.dumps(sp))      # shorten so the test is quick
    r = subprocess.run([*BATCH, "resume", str(out)], capture_output=True, text=True, timeout=120); assert r.returncode == 0, r.stdout[-300:] + r.stderr[-300:]
    j = list(state(out).values())[0]; assert j["status"] == "succeeded" and Path(j["output"]).exists()

def test_second_manager_on_same_batch_is_refused(tmp_path):
    out = tmp_path / "b"; mgr = start_manager(tmp_path, [write(tmp_path, "s", slow_spec("slow3"))], out)
    try:
        assert wait_for(lambda: (out / "batch.lock").exists(), 30)
        r = subprocess.run([*BATCH, "run", write(tmp_path, "x", spec("x")), "--out", str(out)], capture_output=True, text=True); assert r.returncode != 0 and "already running" in (r.stderr + r.stdout)
    finally:
        subprocess.run([*BATCH, "cancel", str(out)], capture_output=True); mgr.wait(timeout=60)

# ---------------------------------------------------------------- unified pipeline
def run_pipe(tmp_path, s, *args, workdir="run"):
    sp = write(tmp_path, "p", s); return subprocess.run([sys.executable, str(SCRIPTS / "pipeline.py"), "--spec", sp, "--out", str(tmp_path / "o.mp4"), "--workdir", str(tmp_path / workdir), *args], capture_output=True, text=True), tmp_path / workdir / "run.json"

def test_pipeline_runs_all_stages_with_contracts_and_logs(tmp_path):
    r, rj = run_pipe(tmp_path, spec("p", 2)); assert r.returncode == 0, r.stdout[-500:] + r.stderr[-300:]; d = json.loads(rj.read_text()); st = d["stages"]
    assert d["ok"] and all(st[s]["status"] in ("ok", "skipped") for s in pipeline.STAGES) and st["render"]["outputs"] and st["qa"]["status"] == "ok"; assert (tmp_path / "run" / "run.log").read_text().count("\n") >= 6

def test_stages_can_be_skipped_or_selected(tmp_path):
    r, rj = run_pipe(tmp_path, spec("p", 1), "--skip", "images,videos,voice,transcribe,effects,qa"); st = json.loads(rj.read_text())["stages"]; assert r.returncode == 0 and st["voice"]["status"] == "skipped" and st["render"]["status"] == "ok"
    r, rj = run_pipe(tmp_path, spec("p", 1), "--only", "voice", workdir="r2"); st = json.loads(rj.read_text())["stages"]; assert st["voice"]["status"] == "ok" and st["render"]["status"] == "skipped" and not (tmp_path / "o2.mp4").exists()

def test_optional_stage_failure_continues_with_fallback_unless_strict(tmp_path):
    s = spec("p", 1); s["scenes"][0]["image_prompt"] = "a cup"; env_clean = {k: v for k, v in os.environ.items() if not k.startswith("AIVE_IMAGE")}
    sp = write(tmp_path, "p", s); base = [sys.executable, str(SCRIPTS / "pipeline.py"), "--spec", sp, "--out", str(tmp_path / "o.mp4")]
    r = subprocess.run([*base, "--workdir", str(tmp_path / "r1")], capture_output=True, text=True, env=env_clean); st = json.loads((tmp_path / "r1" / "run.json").read_text())["stages"]
    assert r.returncode == 0 and st["images"]["status"] == "failed" and "no image provider" in st["images"]["error"] and st["render"]["status"] == "ok"
    r = subprocess.run([*base, "--workdir", str(tmp_path / "r2"), "--strict"], capture_output=True, text=True, env=env_clean); assert r.returncode == 1 and json.loads((tmp_path / "r2" / "run.json").read_text())["stages"]["render"]["status"] == "not_run"

def test_mandatory_failure_aborts_and_missing_input_is_exit_2(tmp_path):
    s = spec("p", 1); s["voice"] = {"mode": "file", "file": str(tmp_path / "missing.wav")}; r, rj = run_pipe(tmp_path, s); st = json.loads(rj.read_text())["stages"]; assert r.returncode == 1 and st["voice"]["status"] == "failed"
    r = subprocess.run([sys.executable, str(SCRIPTS / "pipeline.py"), "--out", str(tmp_path / "x.mp4"), "--workdir", str(tmp_path / "r")], capture_output=True, text=True); assert r.returncode == 2

def test_post_effects_stage_produces_second_file(tmp_path):
    r, rj = run_pipe(tmp_path, spec("p", 1), "--post-effects", "--effect color-grade --look warm"); assert r.returncode == 0 and (tmp_path / "o_fx.mp4").exists()

def test_pipeline_writes_no_api_keys_into_logs(tmp_path):
    s = spec("p", 1); s["scenes"][0]["image_prompt"] = "x"; sp = write(tmp_path, "p", s)
    env = {**os.environ, "AIVE_IMAGE_PROVIDER": "openai", "AIVE_IMAGE_MODEL": "m", "AIVE_IMAGE_API_KEY": "sk-leaktest123456789", "AIVE_IMAGE_BASE_URL": "http://127.0.0.1:9/v1", "AIVE_RETRIES": "0"}
    subprocess.run([sys.executable, str(SCRIPTS / "pipeline.py"), "--spec", sp, "--out", str(tmp_path / "o.mp4"), "--workdir", str(tmp_path / "r")], capture_output=True, text=True, env=env)
    blob = "".join(p.read_text(errors="ignore") for p in (tmp_path / "r").rglob("*") if p.is_file() and p.suffix in (".json", ".log")); assert "sk-leaktest123456789" not in blob and "image" in blob
