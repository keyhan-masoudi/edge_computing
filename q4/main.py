import os
import pandas as pd

from config        import OUTPUT_DIR
from simulation    import run_all
from visualisation import plot_comparison, plot_gantt


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Run all algorithms on all load levels ──────────────────────────
    results_df = run_all()

    # ── 2. Save results CSV ───────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "q4_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results CSV saved: {csv_path}")

    # ── 3. Print pivot summary tables ─────────────────────────────────────
    _print_pivots(results_df)

    # ── 5. Generate charts ────────────────────────────────────────────────
    print("\n  Generating charts...")

    plot_comparison(results_df,
                    os.path.join(OUTPUT_DIR, "q4_comparison.png"))

    for alg in ["EDF", "RM", "SEDM_Heuristic", "RL_Q-Learning"]:
        plot_gantt(
            load_level = "overload",
            algorithm  = alg,
            out        = os.path.join(OUTPUT_DIR, f"q4_gantt_{alg}.png"),
        )

    print("\n" + "▓" * 62)
    print(f"  SIMULATION COMPLETE — all output files are in '{OUTPUT_DIR}/'")
    print("▓" * 62)



def _print_pivots(df: pd.DataFrame) -> None:
    load_order = ["low_load", "medium_load", "overload"]
    pivots = [
        ("Miss Ratio (%)",    "Deadline Miss Ratio (%)"),
        ("Avg Response (ms)", "Avg Response Time (ms)"),
        ("Energy (J)",        "Total Energy (J)"),
        ("Utilization (%)",   "Node Utilization (%)"),
    ]
    for col, label in pivots:
        print("\n" + "─" * 62)
        print(f"  SUMMARY — {label}")
        print("─" * 62)
        pivot = (df.pivot(index="algorithm",
                          columns="load_level",
                          values=col)
                   .reindex(columns=load_order))
        print(pivot.to_string())


if __name__ == "__main__":
    main()