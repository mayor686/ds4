#!/usr/bin/env python3
"""Compare two JSON files produced by ds4 --dump-logits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> tuple[dict, list[float]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    values = payload.get("logits")
    if not isinstance(values, list):
        raise ValueError(f"{path}: missing logits array")
    if any(value is None or not math.isfinite(float(value)) for value in values):
        raise ValueError(f"{path}: logits contain non-finite values")
    return payload, [float(value) for value in values]


def top_indices(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=values.__getitem__, reverse=True)[:count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ref_meta, ref = load(args.reference)
    cand_meta, cand = load(args.candidate)
    if len(ref) != len(cand):
        raise SystemExit("compare_logits: vocabulary sizes differ")

    deltas = [b - a for a, b in zip(ref, cand)]
    abs_deltas = [abs(delta) for delta in deltas]
    worst = max(range(len(ref)), key=abs_deltas.__getitem__)
    ref_top20 = top_indices(ref, 20)
    cand_top20 = top_indices(cand, 20)
    ref_top5 = set(ref_top20[:5])
    cand_top5 = set(cand_top20[:5])
    top_union = set(ref_top20) | set(cand_top20)
    result = {
        "reference_prompt_tokens": ref_meta.get("prompt_tokens"),
        "candidate_prompt_tokens": cand_meta.get("prompt_tokens"),
        "vocab": len(ref),
        "argmax_reference": ref_top20[0],
        "argmax_candidate": cand_top20[0],
        "argmax_match": ref_top20[0] == cand_top20[0],
        "top5_overlap": len(ref_top5 & cand_top5),
        "top20_overlap": len(set(ref_top20) & set(cand_top20)),
        "mean_abs": sum(abs_deltas) / len(abs_deltas),
        "rms": math.sqrt(sum(delta * delta for delta in deltas) / len(deltas)),
        "max_abs": abs_deltas[worst],
        "max_abs_token": worst,
        "max_abs_reference": ref[worst],
        "max_abs_candidate": cand[worst],
        "top20_union_max_abs": max(abs_deltas[index] for index in top_union),
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
