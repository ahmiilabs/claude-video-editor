#!/usr/bin/env python3
"""Batch renderer: many specs, a job queue, parallel workers, retries, cancellation, crash/interrupt recovery, per-job logs, summary report.

  python scripts/batch.py run specs/*.json --out batch_out --workers 2 --retries 1
  python scripts/batch.py status batch_out
  python scripts/batch.py cancel batch_out [JOB_ID]       # cancel one job or the whole batch (works while `run` is active)
  python scripts/batch.py resume batch_out                # re-queue interrupted jobs and continue
  python scripts/batch.py retry batch_out                 # re-queue failed jobs
  python scripts/batch.py report batch_out

Layout: batch_out/batch.json (state), batch_out/jobs/<job_id>/{spec.json, out.mp4, out.mp4.qa.json, log.txt, tmp/}, batch_out/summary.{json,md}
Isolation: every job is its own OS process with its own output folder and its own AIVE_TMP, so concurrent jobs can never overwrite each other.
"""
import argparse, hashlib, json, os, re, signal, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import config
from common import get_logger, resolve_spec_paths

log = get_logger("aive.batch")
HERE = Path(__file__).resolve().parent
TERMINAL = {"succeeded", "failed", "cancelled"}

class Batch:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.jobs_dir = self.root / "jobs"; self.state_file = self.root / "batch.json"; self.lock = threading.Lock()
        self.state = json.loads(self.state_file.read_text(encoding="utf-8")) if self.state_file.exists() else {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "jobs": {}}
    def save(self):
        with self.lock:
            tmp = str(self.state_file) + ".tmp"; Path(tmp).write_text(json.dumps(self.state, indent=2), encoding="utf-8"); os.replace(tmp, self.state_file)
    def add(self, spec_path, quality=None, max_attempts=2):
        spec_path = Path(spec_path); data = json.loads(spec_path.read_text(encoding="utf-8")); resolve_spec_paths(data, spec_path.resolve().parent)   # job gets its own spec copy: make media paths absolute first
        jid = "job_%03d_%s" % (len(self.state["jobs"]) + 1, hashlib.sha1((str(spec_path.resolve()) + str(time.time_ns())).encode()).hexdigest()[:6])
        jd = self.jobs_dir / jid; (jd / "tmp").mkdir(parents=True, exist_ok=True)
        (jd / "spec.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.state["jobs"][jid] = {"id": jid, "title": data.get("title") or spec_path.stem, "source": str(spec_path), "status": "queued", "attempts": 0, "max_attempts": max_attempts,
                                   "quality": quality, "progress": 0.0, "started": None, "finished": None, "seconds": None, "error": None, "qa_passed": None, "output": str(jd / "out.mp4")}
        self.save(); return jid
    def job(self, jid): return self.state["jobs"][jid]
    def cancel_flag(self, jid): return (self.jobs_dir / jid / "CANCEL")

def mem_ok(quality):
    try:
        import psutil
        need = 1400 if quality == "hd" else 700
        return psutil.virtual_memory().available / 1e6 > need
    except ImportError: return True

class Runner:
    def __init__(self, batch, workers, retries, quality=None, poll=0.4):
        self.b, self.workers, self.quality, self.poll = batch, max(1, workers), quality, poll; self.running = {}; self.stop = False; self.cancel_all = False
        for j in batch.state["jobs"].values(): j["max_attempts"] = max(j.get("max_attempts", 1), retries + 1)
    # ---- process management
    def start(self, jid):
        j = self.b.job(jid); jd = self.b.jobs_dir / jid; j["attempts"] += 1; j["status"] = "running"; j["started"] = time.time(); j["error"] = None
        env = {**os.environ, "AIVE_TMP": str(jd / "tmp"), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        cmd = [sys.executable, str(HERE / "reel.py"), str(jd / "spec.json"), "--out", str(jd / "out.mp4")] + (["--quality", j["quality"] or self.quality] if (j["quality"] or self.quality) else [])
        logf = open(jd / "log.txt", "a", encoding="utf-8"); logf.write(f"\n=== attempt {j['attempts']} {time.strftime('%H:%M:%S')} ===\n"); logf.flush()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(HERE))
        self.running[jid] = (proc, logf); self.b.save()
    def kill(self, jid, why):
        proc, logf = self.running.pop(jid)
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        logf.write(f"\n=== stopped: {why} ===\n"); logf.close()
        for junk in (self.b.jobs_dir / jid).glob("*.part.mp4"): junk.unlink(missing_ok=True)
    def progress_of(self, jid):
        try:
            txt = (self.b.jobs_dir / jid / "log.txt").read_text(encoding="utf-8", errors="replace")[-4000:]; m = re.findall(r"\[render\] scene (\d+)/(\d+)", txt)
            return (int(m[-1][0]) - 1) / int(m[-1][1]) if m else 0.0
        except OSError: return 0.0
    def finish(self, jid, code):
        proc, logf = self.running.pop(jid); logf.close(); j = self.b.job(jid); j["finished"] = time.time(); j["seconds"] = round(j["finished"] - j["started"], 1)
        qa = self.b.jobs_dir / jid / "out.mp4.qa.json"
        if qa.exists():
            try: j["qa_passed"] = json.loads(qa.read_text(encoding="utf-8"))["passed"]
            except (OSError, ValueError, KeyError) as e: log.warning("%s: could not read QA report (%s)", jid, e)
        if code == 0 and Path(j["output"]).exists(): j["status"] = "succeeded"; j["progress"] = 1.0
        else:
            tail = (self.b.jobs_dir / jid / "log.txt").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
            j["error"] = config.redact(f"exit code {code}: " + " | ".join(tail))[:400]
            j["status"] = "queued" if j["attempts"] < j["max_attempts"] and code not in (2,) else "failed"      # code 2 = invalid spec: retrying cannot help
            if j["status"] == "queued": j["not_before"] = time.time() + min(30, 2 ** j["attempts"])
        self.b.save()
    # ---- main loop
    def run(self):
        def on_sig(signum, frame): self.stop = True
        old = {s: signal.signal(s, on_sig) for s in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT))}
        try:
            while True:
                jobs = self.b.state["jobs"]
                for jid, j in jobs.items():                                      # cancellation requests (per job or batch-wide)
                    if (self.b.cancel_flag(jid).exists() or self.cancel_all or (self.b.root / "CANCEL_ALL").exists()) and j["status"] in ("queued", "running"):
                        if jid in self.running: self.kill(jid, "cancelled")
                        j["status"] = "cancelled"; j["finished"] = time.time(); self.b.save()
                for jid in list(self.running):
                    proc = self.running[jid][0]; self.b.job(jid)["progress"] = round(self.progress_of(jid), 2)
                    if proc.poll() is not None: self.finish(jid, proc.returncode)
                if self.stop:
                    for jid in list(self.running): self.kill(jid, "interrupted"); self.b.job(jid)["status"] = "interrupted"
                    self.b.save(); self.write_summary(); return 130
                queued = [jid for jid, j in jobs.items() if j["status"] == "queued" and j.get("not_before", 0) <= time.time()]
                while queued and len(self.running) < self.workers and mem_ok(self.quality): self.start(queued.pop(0))
                if not self.running and not any(j["status"] == "queued" for j in jobs.values()): break
                time.sleep(self.poll)
        finally:
            for s, h in old.items(): signal.signal(s, h)
        self.b.save(); self.write_summary(); return 0 if all(j["status"] == "succeeded" for j in self.b.state["jobs"].values()) else 1
    def write_summary(self):
        write_summary(self.b)

def write_summary(b):
    js = list(b.state["jobs"].values()); cnt = {}
    for j in js: cnt[j["status"]] = cnt.get(j["status"], 0) + 1
    summ = {"batch": str(b.root), "total": len(js), "counts": cnt, "jobs": js}
    (b.root / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    lines = [f"# Batch summary\n", f"{len(js)} jobs: " + ", ".join(f"{v} {k}" for k, v in sorted(cnt.items())) + "\n", "| job | title | status | attempts | seconds | QA | output |", "|---|---|---|---|---|---|---|"]
    for j in js: lines.append(f"| {j['id']} | {j['title']} | {j['status']} | {j['attempts']} | {j['seconds'] or ''} | {'pass' if j['qa_passed'] else ('FAIL' if j['qa_passed'] is False else '-')} | {j['output'] if j['status'] == 'succeeded' else (j['error'] or '')} |")
    (b.root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summ

def acquire(root):
    lock = Path(root) / "batch.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip()); os.kill(pid, 0)
            raise SystemExit(f"another batch manager (pid {pid}) is already running on {root}. Use `cancel` or wait.")
        except (ValueError, ProcessLookupError, PermissionError, OSError): pass
    lock.write_text(str(os.getpid())); return lock

def recover(b):
    n = 0
    for j in b.state["jobs"].values():
        if j["status"] in ("running", "interrupted"):
            j["status"] = "queued"; n += 1
            for junk in (b.jobs_dir / j["id"]).glob("*.part.mp4"): junk.unlink(missing_ok=True)
    if n: log.info("recovered %d interrupted job(s)", n)
    return n

def cmd_run(a):
    root = Path(a.out).resolve(); root.mkdir(parents=True, exist_ok=True); lock = acquire(root)
    try:
        b = Batch(root); recover(b)
        for sp in a.specs: b.add(sp, a.quality)
        if not b.state["jobs"]: raise SystemExit("no jobs: give at least one spec file")
        (root / "CANCEL_ALL").unlink(missing_ok=True); workers = a.workers or config.general()["workers"]     # an explicit --workers is honoured
        if workers > (os.cpu_count() or 1): log.warning("%d workers requested but only %d CPU(s): jobs will compete for CPU (each render is CPU-bound)", workers, os.cpu_count() or 1)
        print(f"batch: {len(b.state['jobs'])} job(s), {workers} worker(s) -> {root}")
        code = Runner(b, workers, a.retries, a.quality).run(); print(json.dumps(summary_counts(b), indent=2)); return code
    finally: lock.unlink(missing_ok=True)

def summary_counts(b):
    c = {}
    for j in b.state["jobs"].values(): c[j["status"]] = c.get(j["status"], 0) + 1
    return c

def cmd_status(a):
    b = Batch(a.dir)
    for j in b.state["jobs"].values(): print(f"{j['id']}  {j['status']:11s} {int(j['progress'] * 100):3d}%  attempts={j['attempts']}  {j['title']}" + (f"  ERROR: {j['error']}" if j["error"] and j["status"] == "failed" else ""))
    print(summary_counts(b))

def cmd_cancel(a):
    root = Path(a.dir).resolve()
    if a.job: (root / "jobs" / a.job / "CANCEL").write_text("1"); print(f"cancel requested for {a.job}")
    else: (root / "CANCEL_ALL").write_text("1"); print("cancel requested for the whole batch")
    b = Batch(root)                                                         # if no manager is running, apply directly
    if not (root / "batch.lock").exists():
        for jid, j in b.state["jobs"].items():
            if (not a.job or jid == a.job) and j["status"] in ("queued", "running", "interrupted"): j["status"] = "cancelled"
        b.save(); (root / "CANCEL_ALL").unlink(missing_ok=True); write_summary(b)

def cmd_retry(a, only_failed=True):
    root = Path(a.dir).resolve(); b = Batch(root); n = 0
    for j in b.state["jobs"].values():
        if j["status"] in ("failed", "cancelled") or j["status"] in ("interrupted",):
            j["status"] = "queued"; j["attempts"] = 0; j["error"] = None; j.pop("not_before", None); (b.jobs_dir / j["id"] / "CANCEL").unlink(missing_ok=True); n += 1
    b.save(); print(f"re-queued {n} job(s)")
    if n: a.specs = []; a.out = a.dir; a.workers = a.workers; a.quality = None; a.retries = a.retries; a.workers_cap = True; return cmd_run(a)

def cmd_resume(a):
    root = Path(a.dir).resolve(); b = Batch(root); recover(b); b.save()
    a.specs = []; a.out = a.dir; a.quality = None; a.workers_cap = True; return cmd_run(a)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("specs", nargs="+"); r.add_argument("--out", default="batch_out"); r.add_argument("--workers", type=int, default=0); r.add_argument("--retries", type=int, default=1); r.add_argument("--quality", choices=["draft", "hd"]); r.set_defaults(workers_cap=True)
    for n in ("status", "cancel", "retry", "resume", "report"):
        p = sub.add_parser(n); p.add_argument("dir")
        if n == "cancel": p.add_argument("job", nargs="?")
        if n in ("retry", "resume"): p.add_argument("--workers", type=int, default=0); p.add_argument("--retries", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "run": sys.exit(cmd_run(a))
    elif a.cmd == "status": cmd_status(a)
    elif a.cmd == "cancel": cmd_cancel(a)
    elif a.cmd == "retry": sys.exit(cmd_retry(a) or 0)
    elif a.cmd == "resume": sys.exit(cmd_resume(a))
    else: print(open(Path(a.dir) / "summary.md", encoding="utf-8").read())

if __name__ == "__main__":
    main()
