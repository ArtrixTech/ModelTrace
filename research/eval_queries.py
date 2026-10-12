"""Offline study: how does attribution accuracy / calibration scale with the number of
queries N, and which aggregation is best?

Grouped cross-validation on the reference data (12 environments x 3 challenges x 18 models):
environments are partitioned into held-out groups, robust artifacts are refit on the rest,
and every held-out response is scored once. Random N-subsets of a model's held-out
responses are then aggregated with several methods.

Usage:  OMP_NUM_THREADS=1 python3 research/eval_queries.py [--group 4] [--repeats 4]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bank_builder import fit_robust_artifacts, read_rows  # noqa: E402
from fingerprint import count_numbers, robust_score_counts, ordered_block_scores, robust_score_numbers  # noqa: E402

SOURCES = {"gpt": ROOT / "data" / "gpt_reference.jsonl", "claude": ROOT / "data" / "claude_reference.jsonl"}


def load_rows():
    rows = []
    for family, path in SOURCES.items():
        rows.extend({**row, "family_id": family} for row in read_rows(path))
    return rows


def zs(v):
    v = np.asarray(v, dtype=np.float64)
    return (v - v.mean()) / max(v.std(), 1e-12)


def log_softmax(x):
    m = x.max()
    return x - (m + math.log(np.exp(x - m).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=4, help="environments held out per fold")
    ap.add_argument("--repeats", type=int, default=4, help="random environment partitions")
    ap.add_argument("--subsets", type=int, default=40, help="random N-subsets per model per fold")
    ap.add_argument("--max-n", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "eval_queries.json")
    args = ap.parse_args()

    rows = load_rows()
    models = list(dict.fromkeys(r["source"] for r in rows))
    family = {r["source"]: r["family_id"] for r in rows}
    envs = sorted({r["condition_id"] for r in rows})
    wrapper = {r["condition_id"]: r["wrapper_transport"] for r in rows}
    rng = random.Random(20261012)

    # records[method][N] -> list of (score_vector, truth_index)
    methods = ("meanz", "pooled")
    records = {m: {n: [] for n in range(1, args.max_n + 1)} for m in methods}
    single_env = {n: [] for n in (1, 2, 3)}  # all N queries from the same unseen environment
    per_wrapper = {}

    for rep in range(args.repeats):
        order = envs[:]
        rng.shuffle(order)
        groups = [order[i:i + args.group] for i in range(0, len(order), args.group)]
        for held in groups:
            held_set = set(held)
            train = [r for r in rows if r["condition_id"] not in held_set]
            bank = {"robust": fit_robust_artifacts(train, models)}
            test = [r for r in rows if r["condition_id"] in held_set]
            for r in test:
                s = robust_score_numbers(r["numbers"], bank)
                r["_fused"] = np.asarray(s["fused"])
                r["_ordered"] = np.asarray(s["ordered"])
            for ti, model in enumerate(models):
                mine = [r for r in test if r["source"] == model]
                # single-environment subsets (realistic relay: one fixed wrapper)
                if rep == 0:
                    for env in held:
                        in_env = [r for r in mine if r["condition_id"] == env]
                        for n in (1, 2, 3):
                            for sub in _combos(in_env, n):
                                v = np.mean([r["_fused"] for r in sub], axis=0)
                                single_env[n].append((v, ti))
                                if n == 3:
                                    per_wrapper.setdefault(wrapper[env], []).append(int(np.argmax(v) == ti))
                for n in range(1, min(args.max_n, len(mine)) + 1):
                    for _ in range(args.subsets):
                        sub = rng.sample(mine, n)
                        records["meanz"][n].append((np.mean([r["_fused"] for r in sub], axis=0), ti))
                        pooled_counts = np.sum([count_numbers(r["numbers"]) for r in sub], axis=0).tolist()
                        hel = np.asarray(robust_score_counts(pooled_counts, bank)["fused"])
                        ordered = np.mean([r["_ordered"] for r in sub], axis=0)
                        records["pooled"][n].append((zs(0.75 * hel + 0.25 * ordered), ti))
        print(f"repeat {rep + 1}/{args.repeats} done", file=sys.stderr)

    is_fam = np.asarray([[family[a] == family[b] for b in models] for a in models])
    upstream_beta = json.loads((ROOT / "data" / "unified_bank.json").read_text())["calibration"]

    def summarize(recs, beta):
        acc = fam = nll = conf = 0.0
        bins = np.zeros((10, 3))
        for v, t in recs:
            lp = log_softmax(beta * v)
            p = np.exp(lp)
            top = int(np.argmax(v))
            ok = top == t
            acc += ok
            fam += is_fam[top, t]
            nll -= lp[t]
            conf += p[top]
            b = min(9, int(p[top] * 10))
            bins[b] += (1, p[top], ok)
        k = len(recs)
        ece = sum(abs(b[1] - b[2]) for b in bins if b[0]) / k
        return {"n": k, "acc": acc / k, "family_acc": fam / k, "nll": nll / k, "mean_conf": conf / k, "ece": ece}

    def fit_beta(recs):
        grid = np.exp(np.linspace(math.log(0.05), math.log(200.0), 400))
        best = min(grid, key=lambda b: sum(-log_softmax(b * v)[t] for v, t in recs[::3]))
        return float(best)

    out = {"config": vars(args) | {"out": str(args.out)}, "multi_env": {}, "single_env": {}, "per_wrapper_acc_n3": {}}
    for m in methods:
        out["multi_env"][m] = {}
        for n, recs in records[m].items():
            if not recs:
                continue
            ub = float(upstream_beta[str(min(n, 3))]["beta"])
            fb = fit_beta(recs)
            out["multi_env"][m][n] = {"upstream_beta": ub, **{f"up_{k}": v for k, v in summarize(recs, ub).items()},
                                      "fitted_beta": fb, **{f"fit_{k}": v for k, v in summarize(recs, fb).items()}}
    for n, recs in single_env.items():
        ub = float(upstream_beta[str(n)]["beta"])
        out["single_env"][n] = {"upstream_beta": ub, **summarize(recs, ub), "fitted_beta": fit_beta(recs)}
    out["per_wrapper_acc_n3"] = {k: sum(v) / len(v) for k, v in per_wrapper.items()}
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


def _combos(items, n):
    import itertools
    return list(itertools.combinations(items, n))


if __name__ == "__main__":
    main()
