#!/usr/bin/env python3
"""
Generate schedulable hard-task specs via UUnifast + WFD.

Usage:
  python task_parameter_generation_uunifast.py \
    --num-tasks 9 \
    --periods 7,5,6,4,9,12,3,10,2 \
    --num-cores 8 \
    --total-util 5.6 \
    --output hard_tasks.json \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import List, Optional


# --- Simulator constants (keep in sync with task_and_user_generator.py / config.py)
FREQUENCY = 0.5
EXEC_TIME_DIVISOR = 1e6
SCALING_MAX = 1.2 * 1.3  # max alpha * max beta
DEFAULT_LAMBDA = 1.0

# Ranges for picking size / cycles (tune as needed)
SIZE_MIN_BOUND = 200
SIZE_MAX_BOUND = 3000
CYCLES_MIN_BOUND = 100
CYCLES_MAX_BOUND = 2000
ABS_MIN_SIZE = 1
ABS_MIN_CYCLES = 1

# min/max spread inside each task spec (for random generation later)
INTRA_TASK_SPREAD = 0.85  # min = max * this


@dataclass
class GeneratedTask:
    period: int
    size_min: int
    size_max: int
    cycles_min: int
    cycles_max: int
    lambda_: float
    utilization: float
    wcet: float
    core: int


def uunifast(n: int, u_total: float, rng: random.Random) -> List[float]:
    """Generate n utilizations summing to u_total (Bini & Buttazzo UUnifast)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if u_total <= 0:
        raise ValueError("u_total must be positive")

    remaining = u_total
    utils: List[float] = []
    for i in range(n - 1):
        exp = 1.0 / (n - i)
        ui = remaining * (rng.random() ** exp)
        utils.append(ui)
        remaining -= ui
    utils.append(remaining)
    return utils


def uunifast_bounded(
    n: int,
    u_total: float,
    rng: random.Random,
    u_max: float = 1.0 - 1e-9,
    u_min: float = 0.0,
) -> Optional[List[float]]:
    """
    UUnifast-style draw with each utilization capped at u_max.

    Plain UUnifast can produce u_i > 1 even when the mean is low; that breaks
    partitioned scheduling where every task must fit on a single core.
    """
    if u_min < 0:
        raise ValueError("u_min must be non-negative")
    if u_total < n * u_min:
        return None
    if u_total > n * u_max:
        return None

    remaining = u_total
    utils: List[float] = []
    for i in range(n - 1):
        left = n - i
        low = max(u_min, remaining - (left - 1) * u_max)
        high = min(u_max, remaining - (left - 1) * u_min)
        if low > high + 1e-12:
            return None

        exp = 1.0 / left
        ui = remaining * (rng.random() ** exp)
        ui = min(max(ui, low), high)
        utils.append(ui)
        remaining -= ui

    if remaining < u_min - 1e-9 or remaining > u_max + 1e-9:
        return None
    utils.append(remaining)
    return utils


def wfd_assign(
    utilizations: List[float],
    num_cores: int,
) -> tuple[List[int], List[float]]:
    """
    Worst Fit Decreasing by utilization.
    Returns (task_index -> core_id, per-core load).
    """
    indexed = sorted(
        enumerate(utilizations),
        key=lambda x: x[1],
        reverse=True,
    )
    core_loads = [0.0] * num_cores
    assignment = [0] * len(utilizations)

    for task_idx, u in indexed:
        core = min(range(num_cores), key=lambda c: core_loads[c])
        assignment[task_idx] = core
        core_loads[core] += u

    return assignment, core_loads


def compute_exec_time(
    size_baseline: float,
    cycles_baseline: float,
    scaling: float,
    sensitivity: float,
    frequency: float = FREQUENCY,
) -> float:
    """
    data_size       = size_baseline * scaling * sensitivity
    cycles_per_bit  = cycles_baseline * scaling * sensitivity
    exec_time       = (data_size * cycles_per_bit) / (frequency * EXEC_TIME_DIVISOR)
    """
    data_size = size_baseline * scaling * sensitivity
    cycles_per_bit = cycles_baseline * scaling * sensitivity
    return (data_size * cycles_per_bit) / (frequency * EXEC_TIME_DIVISOR)


def compute_wcet(
    size_baseline: float,
    cycles_baseline: float,
    lambda_: float = DEFAULT_LAMBDA,
    scaling: float = SCALING_MAX,
    frequency: float = FREQUENCY,
) -> float:
    """WCET at max environment scaling for the given baselines."""
    return compute_exec_time(size_baseline, cycles_baseline, scaling, lambda_, frequency)


def max_baseline_product(target_wcet: float, lambda_: float = DEFAULT_LAMBDA) -> float:
    """Largest size_baseline * cycles_baseline with exec_time <= target_wcet at WCET."""
    unit = compute_exec_time(1.0, 1.0, SCALING_MAX, lambda_)
    return target_wcet / unit


def _effective_bounds(product: float) -> tuple[int, int, int, int]:
    """Lower/upper search bounds for size and cycles baselines."""
    if product >= SIZE_MIN_BOUND * CYCLES_MIN_BOUND:
        return SIZE_MIN_BOUND, SIZE_MAX_BOUND, CYCLES_MIN_BOUND, CYCLES_MAX_BOUND

    lo_size = max(ABS_MIN_SIZE, int(math.ceil(product / CYCLES_MAX_BOUND)))
    lo_cycles = max(ABS_MIN_CYCLES, int(math.ceil(product / SIZE_MAX_BOUND)))
    hi_size = min(SIZE_MAX_BOUND, max(lo_size, int(math.ceil(math.sqrt(product)))))
    hi_cycles = min(CYCLES_MAX_BOUND, max(lo_cycles, int(math.ceil(math.sqrt(product)))))
    return lo_size, hi_size, lo_cycles, hi_cycles


def pick_size_cycles(
    u: float,
    period: float,
    lambda_: float,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    """
    Pick size_max in range, derive cycles_max.
    Retry with different sizes until cycles fall in bounds.
    Falls back to balanced sqrt split.
    """
    target_wcet = u * period
    product = max_baseline_product(target_wcet, lambda_)
    lo_size, hi_size, lo_cycles, hi_cycles = _effective_bounds(product)

    for _ in range(500):
        size_max = rng.randint(lo_size, hi_size)
        cycles_max = int(product // size_max)
        if cycles_max < lo_cycles:
            continue
        if cycles_max > hi_cycles:
            cycles_max = hi_cycles
        if compute_wcet(size_max, cycles_max, lambda_) <= target_wcet + 1e-9:
            spread_size_lo = max(lo_size, ABS_MIN_SIZE)
            spread_cycles_lo = max(lo_cycles, ABS_MIN_CYCLES)
            size_min = max(spread_size_lo, int(round(size_max * INTRA_TASK_SPREAD)))
            cycles_min = max(spread_cycles_lo, int(round(cycles_max * INTRA_TASK_SPREAD)))
            return size_min, size_max, cycles_min, cycles_max

    side = math.sqrt(product)
    size_max = int(max(lo_size, min(hi_size, round(side))))
    cycles_max = int(min(hi_cycles, max(lo_cycles, product // size_max)))
    while cycles_max >= lo_cycles and compute_wcet(size_max, cycles_max, lambda_) > target_wcet + 1e-9:
        cycles_max -= 1

    if cycles_max < lo_cycles:
        raise ValueError(
            f"Cannot fit WCET={target_wcet:.6f} within size/cycles bounds; "
            "try higher total_util per task or widen bounds."
        )

    spread_size_lo = max(lo_size, ABS_MIN_SIZE)
    spread_cycles_lo = max(lo_cycles, ABS_MIN_CYCLES)
    size_min = max(spread_size_lo, int(round(size_max * INTRA_TASK_SPREAD)))
    cycles_min = max(spread_cycles_lo, int(round(cycles_max * INTRA_TASK_SPREAD)))
    return size_min, size_max, cycles_min, cycles_max


def validate_inputs(
    num_tasks: int,
    periods: List[int],
    num_cores: int,
    total_util: float,
) -> None:
    if len(periods) != num_tasks:
        raise ValueError(f"Expected {num_tasks} periods, got {len(periods)}")
    if num_cores <= 0:
        raise ValueError("num_cores must be positive")
    if total_util <= 0:
        raise ValueError("total_util must be positive")
    if total_util > num_cores:
        raise ValueError(
            f"total_util ({total_util}) cannot exceed num_cores ({num_cores})"
        )
    if any(p <= 0 for p in periods):
        raise ValueError("All periods must be positive")


def generate(
    num_tasks: int,
    periods: List[int],
    num_cores: int,
    total_util: float,
    lambda_: float = DEFAULT_LAMBDA,
    max_attempts: int = 10_000,
    seed: Optional[int] = None,
) -> dict:
    validate_inputs(num_tasks, periods, num_cores, total_util)
    rng = random.Random(seed)

    min_wcet = compute_wcet(ABS_MIN_SIZE, ABS_MIN_CYCLES, lambda_)
    # Each task must be representable and get a fair minimum share.
    fair_share = total_util / num_tasks
    u_min = max(min_wcet / min(periods), fair_share * 0.05)

    for attempt in range(1, max_attempts + 1):
        utils = uunifast_bounded(num_tasks, total_util, rng, u_min=u_min)
        if utils is None:
            continue

        if any(u * period < min_wcet for u, period in zip(utils, periods)):
            continue

        assignment, core_loads = wfd_assign(utils, num_cores)

        if not all(load < 1.0 for load in core_loads):
            continue

        tasks: List[GeneratedTask] = []
        valid = True

        for i, (u, period) in enumerate(zip(utils, periods)):
            try:
                size_min, size_max, cycles_min, cycles_max = pick_size_cycles(
                    u, period, lambda_, rng
                )
            except ValueError:
                valid = False
                break
            wcet = compute_wcet(size_max, cycles_max, lambda_)

            if wcet > u * period + 1e-9:
                valid = False
                break

            tasks.append(
                GeneratedTask(
                    period=period,
                    size_min=size_min,
                    size_max=size_max,
                    cycles_min=cycles_min,
                    cycles_max=cycles_max,
                    lambda_=lambda_,
                    utilization=round(u, 6),
                    wcet=round(wcet, 6),
                    core=assignment[i],
                )
            )

        if not valid:
            continue

        return {
            "metadata": {
                "num_tasks": num_tasks,
                "num_cores": num_cores,
                "total_utilization": total_util,
                "frequency": FREQUENCY,
                "exec_time_divisor": EXEC_TIME_DIVISOR,
                "scaling_max": SCALING_MAX,
                "lambda": lambda_,
                "u_min": round(u_min, 9),
                "min_wcet": round(min_wcet, 9),
                "seed": seed,
                "attempts": attempt,
                "core_loads": [round(x, 6) for x in core_loads],
            },
            "tasks": [
                {
                    "period": t.period,
                    "size_min": t.size_min,
                    "size_max": t.size_max,
                    "cycles_min": t.cycles_min,
                    "cycles_max": t.cycles_max,
                    "lambda": t.lambda_,
                    "utilization": t.utilization,
                    "wcet": t.wcet,
                    "core": t.core,
                }
                for t in tasks
            ],
        }

    raise RuntimeError(
        f"Failed to generate feasible task set after {max_attempts} attempts. "
        "Try lower total_util or fewer tasks."
    )


def parse_periods(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hard tasks via UUnifast + WFD")
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--periods", type=str, required=True, help="Comma-separated, e.g. 7,5,6")
    parser.add_argument("--num-cores", type=int, required=True)
    parser.add_argument("--total-util", type=float, required=True)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--output", type=str, default="hard_tasks.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=10_000)
    args = parser.parse_args()

    periods = parse_periods(args.periods)
    result = generate(
        num_tasks=args.num_tasks,
        periods=periods,
        num_cores=args.num_cores,
        total_util=args.total_util,
        lambda_=args.lambda_,
        max_attempts=args.max_attempts,
        seed=args.seed,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {args.output}")
    print(f"Core loads: {result['metadata']['core_loads']}")
    print(f"Attempts:   {result['metadata']['attempts']}")


if __name__ == "__main__":
    main()
