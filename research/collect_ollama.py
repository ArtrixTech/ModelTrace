"""Collect reference fingerprints for open-weight models via Ollama Cloud.

Replays the exact 36 reference prompts (12 environments x 3 queries, same system prompt /
user prefix as data/{gpt,claude}_reference.jsonl) against each model, so new rows are
directly comparable and can be enrolled by bank_builder. Resumable: rows already present
in the output file (same row_id) are skipped; failed calls are retried.

Usage:
  OLLAMA_API_KEY=... python3 research/collect_ollama.py --models glm-5.3,kimi-k3 \
      [--out data/open_reference.jsonl] [--workers 6] [--reps 1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fingerprint import parse_numbers  # noqa: E402

API = "https://ollama.com/v1/chat/completions"
TEMPLATE_SOURCE = ROOT / "data" / "claude_reference.jsonl"


def family_of(model: str) -> str:
    m = model.lower()
    for prefix in ("gpt-oss", "deepseek", "glm", "kimi", "minimax", "mistral", "nemotron", "gemma", "qwen"):
        if m.startswith(prefix):
            return prefix
    return m.split("-")[0].split(":")[0]


def templates() -> list[dict]:
    seen = {}
    for line in TEMPLATE_SOURCE.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        seen.setdefault(r["challenge_id"], r)
    keep = ("condition_id", "nuisance_condition_id", "wrapper_id", "wrapper_transport", "challenge_id",
            "task_index", "requested_count", "strict_threshold", "prompt", "base_prompt", "system_prompt", "user_prefix")
    return [{k: r[k] for k in keep} for r in sorted(seen.values(), key=lambda r: r["challenge_id"])]


def call(model: str, t: dict, key: str, timeout: int) -> dict:
    messages = ([{"role": "system", "content": t["system_prompt"]}] if t["system_prompt"] else []) + \
               [{"role": "user", "content": t["prompt"]}]
    body = json.dumps({"model": model, "messages": messages, "max_tokens": 16384, "stream": False}).encode()
    req = urllib.request.Request(API, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        j = json.loads(resp.read())
    msg = j["choices"][0]["message"]
    return {"text": msg.get("content") or "", "reasoning_chars": len(msg.get("reasoning") or msg.get("reasoning_content") or ""),
            "finish_reason": j["choices"][0].get("finish_reason"), "usage": j.get("usage"),
            "response_model": j.get("model"), "request_seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "open_reference.jsonl")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reps", type=int, default=1, help="samples per (model, query); rep>1 gets a row_id suffix")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    key = os.environ["OLLAMA_API_KEY"]

    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("strict_valid"):
                done.add(r["row_id"])
    jobs = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for t in templates():
            for rep in range(args.reps):
                rid = f"{model}__{t['challenge_id']}" + (f"__r{rep + 1}" if rep else "")
                if rid not in done:
                    jobs.append((model, t, rid))
    print(f"{len(jobs)} jobs ({len(done)} already collected)", file=sys.stderr)
    lock = threading.Lock()

    def run(job):
        model, t, rid = job
        last = None
        attempt = 0
        throttled = 0
        while True:
            try:
                res = call(model, t, key, args.timeout)
                parsed = len(parse_numbers(res["text"]))
                if parsed >= t["strict_threshold"] or attempt >= args.retries:
                    break
                last = f"short output ({parsed} numbers)"
            except urllib.error.HTTPError as e:
                res, last = None, f"HTTP {e.code}"
                if e.code == 429 and throttled < 40:
                    # Concurrency / rate cap: wait it out without spending a retry.
                    throttled += 1
                    if throttled in (1, 10, 20, 30):
                        print(f"  429 throttled {rid} x{throttled}", file=sys.stderr, flush=True)
                    time.sleep(20 + 10 * min(throttled, 6))
                    continue
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ConnectionError) as e:
                res, last = None, f"{type(e).__name__}: {str(e)[:120]}"
            if attempt >= args.retries:
                break
            time.sleep(min(60, 5 * 2 ** attempt))
            attempt += 1
        if res is None:
            return f"FAIL {rid}: {last}"
        parsed = len(parse_numbers(res["text"]))
        row = {"row_id": rid, "parent_row_id": rid, "bank_id": "open", "source": model, "model_id": model,
               "response_model": res["response_model"], "exact_version": model, "family_id": family_of(model),
               **t, "parsed_count": parsed, "strict_valid": parsed >= t["strict_threshold"],
               "temperature": "provider_default", "config_id": "ollama-cloud", "provider": "ollama-cloud",
               "text": res["text"], "error": None, "reasoning_chars": res["reasoning_chars"],
               "finish_reason": res["finish_reason"], "usage": res["usage"], "request_seconds": res["request_seconds"],
               "collected_at": datetime.now(timezone.utc).isoformat()}
        with lock, args.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return f"{'ok ' if row['strict_valid'] else 'BAD'} {rid} {parsed}/{t['requested_count']} {res['request_seconds']:.0f}s"

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(jobs)}] {f.result()}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
