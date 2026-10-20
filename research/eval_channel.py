"""Check whether a collection channel (e.g. Cursor) preserves attribution.

Scores collected rows against a bank, per environment (its 3 queries pooled, as in normal use)
and per single response, and prints top-1 hit rates against --expect.

Usage:  python3 research/eval_channel.py data/cursor_control.jsonl --expect claude-opus-5-5 \
            [--bank data/unified_bank.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bank_builder import read_rows  # noqa: E402
from fingerprint import analyze_global_outputs, load_bank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", type=Path)
    ap.add_argument("--expect", required=True, help="bank model id the rows should attribute to")
    ap.add_argument("--bank", type=Path, default=ROOT / "data" / "unified_bank.json")
    args = ap.parse_args()
    bank = load_bank(args.bank)
    rows = [r for r in read_rows(args.rows) if r.get("strict_valid")]
    by_env = defaultdict(list)
    for r in rows:
        by_env[r["condition_id"]].append({"text": r["text"], "expected_count": r["requested_count"]})

    env_hits, env_preds = 0, Counter()
    for env, outputs in sorted(by_env.items()):
        res = analyze_global_outputs(outputs, bank)
        rank = [x["model"] for x in res["results"]].index(args.expect) + 1 if args.expect in \
            [x["model"] for x in res["results"]] else None
        env_hits += res["prediction"] == args.expect
        env_preds[res["prediction"]] += 1
        print(f"{env}  n={len(outputs)}  top1={res['prediction']} p={res['probability']:.2f}  "
              f"family={res['family_prediction']}  expect_rank={rank}")
    single = Counter(analyze_global_outputs([o], bank)["prediction"] for outs in by_env.values() for o in outs)
    print(f"\nenvironment top-1: {env_hits}/{len(by_env)}  {dict(env_preds)}")
    print(f"single-response top-1: {single[args.expect]}/{sum(single.values())}  {dict(single.most_common(5))}")


if __name__ == "__main__":
    main()
