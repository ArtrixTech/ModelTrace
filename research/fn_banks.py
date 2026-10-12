"""Build held-out banks for false-negative (dilution) simulation.

Environments are split into folds. For every fold we write
  fold{k}.json            bank fit on the other environments (all 18 models)
  fold{k}-no-{model}.json same, with `model` removed (out-of-bank substitute)
  fold{k}-test.jsonl      held-out responses (ground truth `source`)
Consumers (llms-fireye scripts/eval-fn.mjs) replay relay scenarios against these.

Usage: OMP_NUM_THREADS=1 python3 research/fn_banks.py --out /tmp/fn [--folds 3] [--lomo]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bank_builder import build_bank  # noqa: E402
from rebuild_unified_bank import reference_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--lomo", action="store_true", help="also build leave-one-model-out banks")
    ap.add_argument("--jobs", type=int, default=6, help="parallel processes for LOMO banks")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = reference_rows()
    envs = sorted({r["condition_id"] for r in rows})
    random.Random(20261012).shuffle(envs)
    size = -(-len(envs) // args.folds)
    models = list(dict.fromkeys(r["source"] for r in rows))
    for k in range(args.folds):
        held = set(envs[k * size:(k + 1) * size])
        train = [r for r in rows if r["condition_id"] not in held]
        test = [r for r in rows if r["condition_id"] in held]
        (args.out / f"fold{k}.json").write_text(json.dumps(build_bank(train)))
        with (args.out / f"fold{k}-test.jsonl").open("w") as fh:
            for r in test:
                fh.write(json.dumps({"source": r["source"], "family": r["family_id"], "env": r["condition_id"],
                                     "text": r["text"], "expected_count": r["requested_count"]}) + "\n")
        if args.lomo:
            todo = [(k, m, [r for r in train if r["source"] != m]) for m in models]
            with ProcessPoolExecutor(args.jobs) as pool:
                for path, bank in pool.map(_lomo, [(args.out / f"fold{k}-no-{m}.json", rows_m) for k, m, rows_m in todo]):
                    path.write_text(json.dumps(bank))
        print(f"fold {k}: held {sorted(held)}", file=sys.stderr)


def _lomo(job):
    path, rows = job
    return path, build_bank(rows)


if __name__ == "__main__":
    main()
