import os
import copy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from config     import ALG_COLORS, TASK_COLORS, OUTPUT_DIR, RANDOM_SEED
from core       import build_platform
from task_generator   import generate_task_set
from algorithms import EDFScheduler, RMScheduler, SEDMScheduler, RLScheduler


# ── Comparison bar chart ─────────────────────────────────────────────────────

def plot_comparison(df: pd.DataFrame,
                    out: str = None) -> None:
    """
    Four-panel bar chart: one panel per KPI, grouped by load level,
    coloured by algorithm.
    
    """
    if out is None:
        out = os.path.join(OUTPUT_DIR, "q4_comparison.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    algorithms  = list(df["algorithm"].unique())
    load_levels = ["low_load", "medium_load", "overload"]
    load_labels = ["Low Load", "Medium Load", "Overload"]
    metric_cfg  = [
        ("Miss Ratio (%)",    "Deadline Miss Ratio (%)",   "lower is better"),
        ("Avg Response (ms)", "Avg Response Time (ms)",    "lower is better"),
        ("Energy (J)",        "Total Energy Consumed (J)", "lower is better"),
        ("Utilization (%)",   "Node Utilization (%)",      "higher = busier"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Edge Real-Time Task Scheduling — Algorithm Comparison\n"
        "EDF  vs  RM  vs  SEDM-Heuristic  vs  RL Q-Learning",
        fontsize=13, fontweight="bold",
    )

    x = np.arange(len(load_levels))
    w = 0.18

    for ax, (col, title, note) in zip(axes.flat, metric_cfg):
        for i, alg in enumerate(algorithms):
            vals = []
            for ll in load_levels:
                row = df[(df["algorithm"] == alg) & (df["load_level"] == ll)]
                vals.append(float(row[col].values[0]) if len(row) else 0.0)

            offset = (i - len(algorithms) / 2 + 0.5) * w
            bars   = ax.bar(
                x + offset, vals, w * 0.90,
                label     = alg,
                color     = ALG_COLORS.get(alg, "#607D8B"),
                alpha     = 0.88,
                edgecolor = "white",
                linewidth = 0.4,
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.1f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="#333",
                )

        ax.set_title(f"{title}\n({note})", fontsize=9.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(load_labels, fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Comparison chart saved: {out}")


# ── Gantt chart ───────────────────────────────────────────────────────────────

def plot_gantt(load_level: str,
               algorithm:  str,
               out:        str = None,
               seed:       int = RANDOM_SEED,
               window_ms:  int = 300) -> None:
    """
    Horizontal bar (Gantt) chart showing task execution on each node
    for the first *window_ms* milliseconds of the simulation.

    Red border around a bar = deadline was missed.
    Bar colour = task type (periodic / sporadic / aperiodic).

    """
    if out is None:
        out = os.path.join(OUTPUT_DIR, f"q4_gantt_{algorithm}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # ── Generate tasks and schedule ──────────────────────────────────────────
    base_tasks = generate_task_set(load_level, seed=seed)
    nodes      = build_platform()

    sched_map = {
        "EDF":             EDFScheduler(nodes),
        "RM":              RMScheduler(nodes),
        "SEDM_Heuristic": SEDMScheduler(nodes),
        "RL_Q-Learning":   RLScheduler(nodes),
    }
    if algorithm not in sched_map:
        raise ValueError(f"Unknown algorithm '{algorithm}'. "
                         f"Choose from {list(sched_map)}")

    sched = sched_map[algorithm]
    if isinstance(sched, RLScheduler):
        sched.train(copy.deepcopy(base_tasks))
        sched.reset()

    tasks = copy.deepcopy(base_tasks)
    sched.schedule(tasks)

    # ── Filter to display window ─────────────────────────────────────────────
    tplot    = [t for t in tasks
                if t.start_time is not None and t.start_time < window_ms]
    node_ids = [n.id for n in nodes]

    # ── Draw ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 6))

    for task in tplot:
        y     = node_ids.index(task.assigned_node)
        color = TASK_COLORS.get(task.type, "#607D8B")
        ec    = "red"  if task.missed else "black"
        lw    = 1.8    if task.missed else 0.4
        dur   = min(task.finish_time, window_ms) - task.start_time

        ax.barh(y, dur, left=task.start_time, height=0.55,
                color=color, edgecolor=ec, linewidth=lw, alpha=0.85)
        if dur > 0.8:
            ax.text(task.start_time + dur / 2, y,
                    f"T{task.id}",
                    va="center", ha="center",
                    fontsize=5.5, color="white", fontweight="bold")

    ax.set_yticks(range(len(node_ids)))
    ax.set_yticklabels([f"Node {nid}" for nid in node_ids])
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_xlim(0, window_ms)
    ax.set_title(
        f"Gantt Chart — {algorithm} | {load_level} (first {window_ms} ms)\n",
        fontsize=10, fontweight="bold",
    )
    patches = [mpatches.Patch(color=c, label=k) for k, c in TASK_COLORS.items()]
    ax.legend(handles=patches, fontsize=8, loc="upper right")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Gantt chart saved: {out}")
