"""Collect reference fingerprints for models only reachable through Cursor (cursor-agent CLI).

Replays the same 36 reference prompts as research/collect_ollama.py. cursor-agent cannot set a
system prompt, so system-transport environments are sent as a prefix of the user message, and
every request also carries Cursor's own agent prompt (~13k tokens). Validate the channel with a
control model already in the bank (see --id) before enrolling anything collected here.
Runs in ask mode (read-only) inside an empty workspace. Resumable like collect_ollama.py.

Usage:
  python3 research/collect_cursor.py --model gemini-4-argon-medium --id gemini-4-argon \
      [--out data/cursor_reference.jsonl] [--workers 4] [--reps 1]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fingerprint import parse_numbers  # noqa: E402
from collect_ollama import family_of, templates  # noqa: E402


def call(cursor_model: str, t: dict, workspace: str, timeout: int) -> dict:
    content = (t["system_prompt"] + "\n\n" if t["system_prompt"] else "") + t["prompt"]
    t0 = time.time()
    proc = subprocess.run(
        ["cursor-agent", "-p", "--mode", "ask", "--trust", "--workspace", workspace,
         "--output-format", "json", "--model", cursor_model, content],
        cwd=workspace, capture_output=True, text=True, timeout=timeout,
    )
    try:
        j = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        raise RuntimeError(f"exit {proc.returncode}: {(proc.stderr or proc.stdout)[:160]}")
    if j.get("is_error"):
        raise RuntimeError(str(j.get("result"))[:160])
    return {"text": j.get("result") or "", "usage": j.get("usage"), "request_id": j.get("request_id"),
            "request_seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="cursor-agent model name, e.g. gemini-4-argon-medium")
    ap.add_argument("--id", required=True, help="model id stored in the bank, e.g. gemini-4-argon")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "cursor_reference.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reps", type=int, default=1, help="samples per query; rep>1 gets a row_id suffix")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("strict_valid"):
                done.add(r["row_id"])
    jobs = []
    for t in templates():
        for rep in range(args.reps):
            rid = f"{args.id}__{t['challenge_id']}" + (f"__r{rep + 1}" if rep else "")
            if rid not in done:
                jobs.append((t, rid))
    print(f"{len(jobs)} jobs ({len(done)} already collected)", file=sys.stderr)
    lock = threading.Lock()
    workspace = tempfile.mkdtemp(prefix="cursor-collect-")

    def run(job):
        t, rid = job
        res, last = None, None
        for attempt in range(args.retries + 1):
            try:
                res = call(args.model, t, workspace, args.timeout)
                parsed = len(parse_numbers(res["text"]))
                if parsed >= t["strict_threshold"]:
                    break
                last = f"short output ({parsed} numbers)"
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                res, last = None, f"{type(e).__name__}: {str(e)[:160]}"
            if attempt < args.retries:
                time.sleep(min(60, 5 * 2 ** attempt))
        if res is None:
            return f"FAIL {rid}: {last}"
        parsed = len(parse_numbers(res["text"]))
        row = {"row_id": rid, "parent_row_id": rid, "bank_id": "open", "source": args.id, "model_id": args.id,
               "response_model": args.model, "exact_version": args.model, "family_id": family_of(args.id),
               **t, "parsed_count": parsed, "strict_valid": parsed >= t["strict_threshold"],
               "temperature": "provider_default", "config_id": "cursor-agent", "provider": "cursor",
               "text": res["text"], "error": None, "usage": res["usage"], "request_id": res["request_id"],
               "request_seconds": res["request_seconds"], "collected_at": datetime.now(timezone.utc).isoformat()}
        with lock, args.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return f"{'ok ' if row['strict_valid'] else 'BAD'} {rid} {parsed}/{t['requested_count']} {res['request_seconds']:.0f}s"

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(jobs)}] {f.result()}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
