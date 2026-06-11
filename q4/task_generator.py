import os
import random
import numpy as np
import pandas as pd

from config import (RANDOM_SEED, HORIZON, LOAD_PARAMS, PERIODIC_TEMPLATES,
                    OUTPUT_DIR)
from task import Task


# ── Public API ───────────────────────────────────────────────────────────────

def generate_task_set(load_level: str, seed: int = RANDOM_SEED) -> list:
    """
    Build and return a sorted list of Task objects for *load_level*.

    Parameters
    ----------
    load_level : one of 'low_load', 'medium_load', 'overload'
    seed       : RNG seed for reproducibility

    Returns
    -------
    list of Task, sorted by arrival_time ascending
    """
    if load_level not in LOAD_PARAMS:
        raise ValueError(f"Unknown load level '{load_level}'. "
                         f"Choose from {list(LOAD_PARAMS)}")

    Task.reset_counter()
    random.seed(seed)
    np.random.seed(seed)

    n_per, n_spo, n_ape = LOAD_PARAMS[load_level]
    tasks = []

    tasks += _make_periodic(n_per)
    tasks += _make_sporadic(n_spo)
    tasks += _make_aperiodic(n_ape)

    tasks.sort(key=lambda t: t.arrival_time)
    return tasks


def save_dataset_csv(tasks: list, load_level: str,
                     out_dir: str = OUTPUT_DIR) -> str:
    """
    Serialize *tasks* to a CSV file.

    Returns the path of the written file.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = [
        {
            "id":           t.id,
            "type":         t.type,
            "exec_time":    t.exec_time,
            "period":       t.period,
            "deadline":     t.deadline,
            "abs_deadline": t.abs_deadline,
            "energy_cost":  t.energy_cost,
            "comm_latency": t.comm_latency,
            "arrival_time": t.arrival_time,
        }
        for t in tasks
    ]
    path = os.path.join(out_dir, f"dataset_{load_level}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_periodic(n_templates: int) -> list:
    """
    Instantiate *n_templates* periodic task families.
    Each family contributes one Task per period over [0, HORIZON).
    A ±0.3 ms jitter is added to exec_time to avoid perfectly aligned
    worst-case scenarios.
    """
    tasks = []
    for (et, per, dl) in PERIODIC_TEMPLATES[:n_templates]:
        t = 0
        while t < HORIZON:
            jitter = random.uniform(-0.3, 0.3)
            tasks.append(Task(
                task_type    = "periodic",
                exec_time    = et + jitter,
                period       = per,
                deadline     = dl,
                energy_cost  = round(et * 0.25, 2),
                comm_latency = round(random.uniform(0.3, 1.5), 2),
                arrival_time = float(t),
            ))
            t += per
    return tasks


def _make_sporadic(n: int) -> list:
    """
    Generate *n* sporadic tasks with random but bounded arrivals.
    Minimum separation = 2 × exec_time  (bounded sporadic model).
    """
    tasks = []
    for _ in range(n):
        et  = round(random.uniform(4, 18), 1)
        per = et * 2                                      # min inter-arrival
        dl  = round(random.uniform(et * 2.0, et * 5.0), 1)
        arr = round(random.uniform(0, max(0, HORIZON - dl)), 1)
        tasks.append(Task(
            task_type    = "sporadic",
            exec_time    = et,
            period       = per,
            deadline     = dl,
            energy_cost  = round(et * 0.20, 2),
            comm_latency = round(random.uniform(0.5, 3.0), 2),
            arrival_time = arr,
        ))
    return tasks


def _make_aperiodic(n: int) -> list:
    """
    Generate *n* aperiodic (best-effort) tasks.
    Deadlines are loose (3–8 × exec_time) to model soft-deadline jobs.
    """
    tasks = []
    for _ in range(n):
        et  = round(random.uniform(2, 14), 1)
        dl  = round(random.uniform(et * 3.0, et * 8.0), 1)
        arr = round(random.uniform(0, max(0, HORIZON - dl)), 1)
        tasks.append(Task(
            task_type    = "aperiodic",
            exec_time    = et,
            period       = 0,
            deadline     = dl,
            energy_cost  = round(et * 0.15, 2),
            comm_latency = round(random.uniform(0.5, 4.0), 2),
            arrival_time = arr,
        ))
    return tasks