# Batch rendering

`batch.py run specs/*.json --out batch_out --workers N --retries R` creates `batch_out/jobs/<job_id>/{spec.json,out.mp4,out.mp4.qa.json,log.txt,tmp/}` plus `batch.json` (state), `summary.json` and `summary.md`.

* **Isolation**: every job is its own OS process with its own `AIVE_TMP` and output folder; outputs are written atomically (`.part.mp4` then rename), so concurrent jobs cannot overwrite each other.
* **Job IDs & progress**: `job_001_<hash>`; progress is parsed from the job log (`status`).
* **Retries**: failed jobs retry up to R times with backoff, except invalid specs (exit code 2), which cannot be fixed by retrying. `retry` re-queues failed/cancelled jobs.
* **Cancel**: `cancel <dir> [job_id]` stops the process (terminate, then kill), removes partial files and marks `cancelled`; works while `run` is active, or marks state directly when idle.
* **Graceful shutdown / recovery**: SIGINT/SIGTERM stops running jobs, marks them `interrupted`, writes the summary and exits 130. `resume` (or a new `run` on the same folder) re-queues interrupted/crashed jobs and clears stale locks and partial files.
* **Resource awareness**: a job is not started when free RAM is below the estimate for its quality (700 MB draft / 1.4 GB hd, needs `psutil`); workers above the CPU count log a warning. A lock file prevents two managers from using one batch folder.
