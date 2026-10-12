from __future__ import annotations

import json
import sys
from pathlib import Path

from bank_builder import build_bank, read_rows


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "data" / "unified_bank.json"
SOURCES = {
    "gpt": ("GPT", PROJECT / "data" / "gpt_reference.jsonl"),
    "claude": ("Claude", PROJECT / "data" / "claude_reference.jsonl"),
}
# [fireye fork] Open-weight models collected via research/collect_ollama.py. Each row carries
# its own family_id; the file is optional so upstream-only checkouts still rebuild.
OPEN_SOURCE = PROJECT / "data" / "open_reference.jsonl"
OPEN_FAMILY_NAMES = {
    "glm": "GLM", "kimi": "Kimi", "deepseek": "DeepSeek", "minimax": "MiniMax", "mistral": "Mistral",
    "nemotron": "Nemotron", "gemma": "Gemma", "gpt-oss": "gpt-oss", "qwen": "Qwen",
}


def reference_rows() -> list[dict]:
    rows = []
    for family_id, (family_name, path) in SOURCES.items():
        rows.extend({**row, "family_id": family_id, "family_name": family_name} for row in read_rows(path))
    if OPEN_SOURCE.exists():
        open_rows = read_rows(OPEN_SOURCE)
        # Enrol only models with every reference query: incomplete models would shrink the
        # "complete environment" set used by the robust nuisance fit for everyone.
        needed = {row["challenge_id"] for row in rows}
        have = {}
        for row in open_rows:
            have.setdefault(row["source"], set()).add(row["challenge_id"])
        complete = {m for m, ids in have.items() if needed <= ids}
        for m in sorted(set(have) - complete):
            print(f"skip {m}: {len(have[m])}/{len(needed)} reference queries collected", file=sys.stderr)
        rows.extend(
            {**row, "family_name": OPEN_FAMILY_NAMES.get(row["family_id"], row["family_id"])}
            for row in open_rows if row["source"] in complete
        )
    return rows


def main() -> None:
    rows = reference_rows()
    bank = build_bank(rows)
    OUTPUT.write_text(
        json.dumps(bank, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "models": len(bank["models"]),
                "responses": sum(model["response_count"] for model in bank["models"]),
                "calibration": bank["calibration"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
