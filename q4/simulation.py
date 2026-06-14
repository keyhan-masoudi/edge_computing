import copy
import os

import numpy as np
import pandas as pd

from config     import RANDOM_SEED, OUTPUT_DIR
from core       import build_platform
from task_generator  import generate_task_set, save_dataset_csv
from algorithms import EDFScheduler, RMScheduler, SEDMScheduler, RLScheduler

LOAD_LEVELS = ["low_load", "medium_load", "overload"]
RESULT_COLS = [
    "load_level", "algorithm", "total_tasks", "missed_deadlines",
    "Miss Ratio (%)", "Avg Response (ms)", "Energy (J)", "Utilization (%)",
]


# ── Public API ────────────────────────────────────────────────────────────────

def run_all(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Run all four schedulers on all three load levels.

    For each combination:
      1. Generate a fresh task set
      2. Deep-copy tasks so each scheduler works on an independent copy.
      3. For RL: train offline on a copy of the task set, then reset
         node state before the greedy evaluation pass.
      4. Collect metrics and append to results list.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_rows = []

    for load in LOAD_LEVELS:
        base_tasks = generate_task_set(load, seed=seed)
        save_dataset_csv(base_tasks, load)

        n_per = sum(1 for t in base_tasks if t.type == "periodic")
        n_spo = sum(1 for t in base_tasks if t.type == "sporadic")
        n_ape = sum(1 for t in base_tasks if t.type == "aperiodic")

        print(f"\n{'═' * 62}")
        print(f"  Load: {load.upper()}  "
              f"({len(base_tasks)} tasks: "
              f"{n_per} periodic, {n_spo} sporadic, {n_ape} aperiodic)")
        print(f"{'─' * 62}")

        schedulers = _build_schedulers()

        for sched in schedulers:
            tasks = copy.deepcopy(base_tasks)

            if isinstance(sched, RLScheduler):
                sched.train(copy.deepcopy(base_tasks))
                sched.reset()

            metrics               = sched.schedule(tasks)
            metrics["load_level"] = load

            _print_row(sched.name, metrics)
            all_rows.append(metrics)

    df = pd.DataFrame(all_rows)
    return df[RESULT_COLS]


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_schedulers() -> list:
    """
    Instantiate all four schedulers.
    """
    return [
        EDFScheduler(build_platform()),
        RMScheduler(build_platform()),
        SEDMScheduler(build_platform()),
        RLScheduler(build_platform()),
    ]



def _print_row(name: str, m: dict) -> None:
    print(f"  [{name:22s}]  "
          f"Miss={m['Miss Ratio (%)']:6.2f}%  "
          f"Resp={m['Avg Response (ms)']:8.2f}ms  "
          f"E={m['Energy (J)']:7.4f}J  "
          f"Util={m['Utilization (%)']:6.2f}%")
