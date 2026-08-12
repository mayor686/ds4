#!/usr/bin/env python3
"""Analyze ds4-bench distributed graph telemetry and model layer splits.

The tool consumes the ``ds4-bench: graph`` lines emitted by ds4-bench.  It is
deliberately backend-agnostic: CUDA, ROCm, and Metal profiles use the same
format.  With ``--max-layers`` it also searches integer contiguous layer
partitions using the measured per-layer cost as a first-order model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


GRAPH_RE = re.compile(
    r"^ds4-bench: graph phase=(?P<phase>\S+) "
    r"layers=(?P<start>\d+):(?P<end>\d+) "
    r'device="(?P<device>[^"]*)" memory_gib=(?P<memory>[0-9.]+) '
    r"calls=(?P<calls>\d+) tokens=(?P<tokens>\d+) "
    r"eval_ms=(?P<eval>[0-9.]+) avg_eval_ms=(?P<avg>[0-9.]+) "
    r"wait_ms=(?P<wait>[0-9.]+) send_ms=(?P<send>[0-9.]+) "
    r"compute_percent=(?P<percent>[0-9.]+)$"
)


@dataclass(frozen=True)
class Stage:
    phase: str
    start: int
    end: int
    device: str
    memory_gib: float
    calls: int
    tokens: int
    eval_ms: float
    avg_eval_ms: float
    wait_ms: float
    send_ms: float
    compute_percent: float

    @property
    def layers(self) -> int:
        return self.end - self.start + 1

    @property
    def avg_send_ms(self) -> float:
        return self.send_ms / self.calls if self.calls else 0.0


def parse_graph(path: Path) -> dict[str, list[Stage]]:
    phases: dict[str, list[Stage]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for raw in fp:
            match = GRAPH_RE.match(raw.rstrip("\n"))
            if not match:
                continue
            g = match.groupdict()
            stage = Stage(
                phase=g["phase"],
                start=int(g["start"]),
                end=int(g["end"]),
                device=g["device"],
                memory_gib=float(g["memory"]),
                calls=int(g["calls"]),
                tokens=int(g["tokens"]),
                eval_ms=float(g["eval"]),
                avg_eval_ms=float(g["avg"]),
                wait_ms=float(g["wait"]),
                send_ms=float(g["send"]),
                compute_percent=float(g["percent"]),
            )
            phases.setdefault(stage.phase, []).append(stage)
    for stages in phases.values():
        stages.sort(key=lambda stage: stage.start)
    if not phases:
        raise ValueError(f"{path}: no distributed graph telemetry found")
    return phases


def find_csv(log_path: Path) -> Path | None:
    direct = log_path.with_suffix(".csv")
    if direct.exists():
        return direct
    csvs = sorted(log_path.parent.glob("*.csv"))
    return csvs[0] if len(csvs) == 1 else None


def parse_csv(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        return {}
    return {key: float(value) for key, value in rows[-1].items() if value != ""}


def phase_summary(stages: list[Stage]) -> dict[str, float | int | str]:
    total_avg = sum(stage.avg_eval_ms for stage in stages)
    bottleneck = max(stages, key=lambda stage: stage.avg_eval_ms)
    ideal_avg = total_avg / len(stages)
    utilization = total_avg / (len(stages) * bottleneck.avg_eval_ms)
    send_avg = sum(stage.avg_send_ms for stage in stages)
    return {
        "stage_count": len(stages),
        "total_compute_ms": total_avg,
        "bottleneck_layers": f"{bottleneck.start}:{bottleneck.end}",
        "bottleneck_device": bottleneck.device,
        "bottleneck_ms": bottleneck.avg_eval_ms,
        "perfect_balance_ms": ideal_avg,
        "pipeline_utilization_percent": utilization * 100.0,
        "perfect_balance_headroom_percent":
            (bottleneck.avg_eval_ms / ideal_avg - 1.0) * 100.0,
        "send_ms": send_avg,
        "send_vs_compute_percent":
            (send_avg / total_avg * 100.0) if total_avg else 0.0,
    }


def parse_int_list(value: str, expected: int, option: str) -> list[int]:
    try:
        parsed = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"{option} must be a comma-separated integer list") from exc
    if len(parsed) != expected or any(item < 0 for item in parsed):
        raise ValueError(f"{option} needs {expected} non-negative values")
    return parsed


def partitions(total: int, minimum: list[int], maximum: list[int]):
    current = [0] * len(minimum)

    def visit(index: int, left: int):
        if index == len(current) - 1:
            if minimum[index] <= left <= maximum[index]:
                current[index] = left
                yield tuple(current)
            return
        tail_min = sum(minimum[index + 1 :])
        tail_max = sum(maximum[index + 1 :])
        low = max(minimum[index], left - tail_max)
        high = min(maximum[index], left - tail_min)
        for count in range(low, high + 1):
            current[index] = count
            yield from visit(index + 1, left - count)

    yield from visit(0, total)


def optimize_layers(
    prefill: list[Stage],
    generation: list[Stage],
    minimum: list[int],
    maximum: list[int],
    prefill_weight: float,
    device_order: list[str],
) -> dict[str, object]:
    if len(prefill) != len(generation):
        raise ValueError("prefill and generation stage counts differ")
    if any((a.start, a.end) != (b.start, b.end)
           for a, b in zip(prefill, generation)):
        raise ValueError("prefill and generation layer ranges differ")
    total_layers = sum(stage.layers for stage in prefill)
    current = tuple(stage.layers for stage in prefill)
    prefill_unit = [stage.avg_eval_ms / stage.layers for stage in prefill]
    generation_unit = [stage.avg_eval_ms / stage.layers for stage in generation]
    base_prefill = max(stage.avg_eval_ms for stage in prefill)
    base_generation = sum(stage.avg_eval_ms for stage in generation)
    best = None
    evaluated = 0
    for counts in partitions(total_layers, minimum, maximum):
        evaluated += 1
        predicted_prefill = max(unit * count for unit, count in zip(prefill_unit, counts))
        predicted_generation = sum(
            unit * count for unit, count in zip(generation_unit, counts)
        )
        score = (
            prefill_weight * predicted_prefill / base_prefill
            + (1.0 - prefill_weight) * predicted_generation / base_generation
        )
        candidate = (score, predicted_prefill, predicted_generation, counts)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("no layer partition satisfies the supplied limits")
    score, predicted_prefill, predicted_generation, counts = best
    ranges = []
    start = 0
    for device, count in zip(device_order, counts):
        ranges.append({"device": device, "start": start, "end": start + count - 1,
                       "layers": count})
        start += count
    return {
        "warning": "linear estimate; benchmark the proposed split before deployment",
        "evaluated_partitions": evaluated,
        "prefill_weight": prefill_weight,
        "current_counts": current,
        "proposed_counts": counts,
        "ranges": ranges,
        "predicted_prefill_stage_ms": predicted_prefill,
        "predicted_prefill_change_percent":
            (base_prefill / predicted_prefill - 1.0) * 100.0,
        "predicted_generation_ms": predicted_generation,
        "predicted_generation_change_percent":
            (base_generation / predicted_generation - 1.0) * 100.0,
        "score": score,
    }


def markdown_report(log_path: Path, phases: dict[str, list[Stage]],
                    csv_row: dict[str, float], optimization: dict[str, object] | None) -> str:
    lines = [f"# Distributed inference profile: `{log_path.name}`", ""]
    if csv_row:
        lines += [
            f"- Prefill: **{csv_row.get('prefill_tps', math.nan):.2f} token/s**",
            f"- Generation: **{csv_row.get('gen_tps', math.nan):.2f} token/s**",
            "",
        ]
    for phase in ("prefill", "generation"):
        stages = phases.get(phase)
        if not stages:
            continue
        summary = phase_summary(stages)
        lines += [
            f"## {phase.capitalize()}", "",
            "| Layers | Device | VRAM GiB | Compute ms | Share | Send ms |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        total = float(summary["total_compute_ms"])
        for stage in stages:
            share = 100.0 * stage.avg_eval_ms / total if total else 0.0
            lines.append(
                f"| {stage.start}:{stage.end} | {stage.device} | {stage.memory_gib:.1f} "
                f"| {stage.avg_eval_ms:.3f} | {share:.1f}% | {stage.avg_send_ms:.3f} |"
            )
        if phase == "prefill":
            conclusion = (
                f"Bottleneck: `{summary['bottleneck_layers']}` on "
                f"{summary['bottleneck_device']} at {summary['bottleneck_ms']:.3f} ms. "
                f"Pipeline utilization is {summary['pipeline_utilization_percent']:.1f}%; "
                f"perfect stage balance leaves at most "
                f"{summary['perfect_balance_headroom_percent']:.1f}% scheduling headroom."
            )
        else:
            conclusion = (
                f"Generation compute is serial across stages: "
                f"{summary['total_compute_ms']:.3f} ms/token in total. The largest share is "
                f"`{summary['bottleneck_layers']}` on {summary['bottleneck_device']} "
                f"at {summary['bottleneck_ms']:.3f} ms/token; redistributing unchanged "
                f"work cannot remove the other stage costs."
            )
        lines += ["", conclusion, ""]
    if optimization:
        lines += ["## Linear layer-partition estimate", ""]
        for item in optimization["ranges"]:
            lines.append(
                f"- device {item['device']}: layers {item['start']}:{item['end']} "
                f"({item['layers']})"
            )
        lines += [
            "",
            f"Predicted prefill change: {optimization['predicted_prefill_change_percent']:+.1f}%; "
            f"generation change: {optimization['predicted_generation_change_percent']:+.1f}%.",
            "",
            "This is a linear estimate. Validate the proposed split with the same prompt, "
            "context allocation, chunk, and pipeline window.",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="ds4-bench stderr log")
    parser.add_argument("--csv", type=Path, help="matching ds4-bench CSV")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    parser.add_argument("--max-layers", help="per-stage maximum layer counts")
    parser.add_argument("--min-layers", help="per-stage minimum layer counts (default: 1)")
    parser.add_argument("--device-order", help="comma-separated launcher device identifiers")
    parser.add_argument("--prefill-weight", type=float, default=0.5,
                        help="partition objective weight, 0=generation, 1=prefill")
    args = parser.parse_args()
    try:
        phases = parse_graph(args.log)
        csv_row = parse_csv(args.csv if args.csv else find_csv(args.log))
        optimization = None
        if args.max_layers:
            prefill = phases.get("prefill", [])
            generation = phases.get("generation", [])
            count = len(prefill)
            maximum = parse_int_list(args.max_layers, count, "--max-layers")
            minimum = (
                parse_int_list(args.min_layers, count, "--min-layers")
                if args.min_layers else [1] * count
            )
            devices = (
                args.device_order.split(",") if args.device_order
                else [str(index) for index in range(count)]
            )
            if len(devices) != count:
                raise ValueError(f"--device-order needs {count} values")
            if not 0.0 <= args.prefill_weight <= 1.0:
                raise ValueError("--prefill-weight must be between 0 and 1")
            optimization = optimize_layers(prefill, generation, minimum, maximum,
                                           args.prefill_weight, devices)
    except (OSError, ValueError) as exc:
        print(f"distributed_profile: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "log": str(args.log),
            "benchmark": csv_row,
            "phases": {
                name: {
                    "summary": phase_summary(stages),
                    "stages": [asdict(stage) for stage in stages],
                }
                for name, stages in phases.items()
            },
            "optimization": optimization,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(markdown_report(args.log, phases, csv_row, optimization))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
