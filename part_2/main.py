import pandas as pd
import os

from data_generator import generate_dataset
from feature_extractor import build_feature_table
from model import AEStateDetector, evaluate_model      
from passport import create_passport, build_passport_table
from decision_engine import decide_ae_action, run_decision_batch, DEFAULT_NODES 
from zero_trust import (
    zero_trust_check, demo_zero_trust_scenarios,
    get_audit_log, get_break_glass_logs
)
from simulation import run_all_scenarios
from baseline import run_comparison, plot_comparison


def main():
    print("="*70)
    print("Q6 — AE Decision Engine for Industrial Edge Computing")
    print("="*70)

    # ── INITIALIZATION: Create and route to outputs directory ─────────────────
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    os.chdir(out_dir)
    print(f"\n[System] All generated files will be saved to: ./{out_dir}/")

    # ── Part A: Generate dataset ──────────────────────────────────────────────
    print("\n[Part A] Generating synthetic AE dataset...")
    samples = generate_dataset(80)
    
    # We exclude signal_window from the CSV to keep the file readable, 
    # but it STILL exists in the 'samples' memory for Part B to use!
    meta_df = pd.DataFrame([
        {k: v for k, v in s.items() if k != "signal_window"} for s in samples
    ])
    
    # Convert trust_context dict to string so it saves properly in the CSV
    meta_df["trust_context"] = meta_df["trust_context"].apply(str)
    
    meta_df.to_csv("ae_samples_meta.csv", index=False)
    print(f"  > ae_samples_meta.csv saved ({len(meta_df)} samples)")

    # ── Part B: Feature extraction ────────────────────────────────────────────
    print("\n[Part B] Extracting features...")
    ft = build_feature_table(samples)
    ft.to_csv("feature_table.csv", index=False)
    print(f"  > feature_table.csv saved ({len(ft)} rows × {len(ft.columns)} columns)")

    # ── Part C: Train model ───────────────────────────────────────────────────
    print("\n[Part C] Training lightweight AE state detector...")
    detector = AEStateDetector()
    detector.fit(ft)
    evaluate_model(ft)
    preds = detector.predict_batch(ft)
    print(f"  > Prediction distribution:\n{preds['predicted_state'].value_counts().to_string()}")

    # ── Part D: Data passports ─────────────────────────────────────────────────
    print("\n[Part D] Creating Data Passports...")
    passports = [create_passport(s) for s in samples]
    passport_df = build_passport_table(samples)
    passport_df.to_csv("passport_table.csv", index=False)
    print(f"  > passport_table.csv saved")
    print(f"  > audit_required: {passport_df['audit_required'].sum()}/{len(passport_df)}")

    # ── Part E: Decision engine ───────────────────────────────────────────────
    print("\n[Part E] Running edge decision engine...")
    decision_df = run_decision_batch(samples, ft, preds, passports, DEFAULT_NODES)
    decision_df.to_csv("decision_output.csv", index=False)
    print(f"  > decision_output.csv saved")
    print(f"  > decision_type:\n{decision_df['decision_type'].value_counts().to_string()}")

    # ── Part F: Zero Trust ────────────────────────────────────────────────────
    print("\n[Part F] Running Zero Trust scenarios...")
    demo_zero_trust_scenarios(samples, passports)

    # Run ZT checks for every decision in our output to build a comprehensive audit log
    for i, sample in enumerate(samples):
        passport = passports[i]
        row = decision_df.iloc[i]
        zero_trust_check(
            requester="local_edge_01",
            sample_id=sample["sample_id"],
            data_form_requested=row["data_form"],
            passport=passport,
            network_state=sample["network_state"],
        )

    audit_df = get_audit_log()
    if not audit_df.empty:
        audit_df.to_csv("audit_log.csv", index=False)
        print(f"\n  > audit_log.csv saved ({len(audit_df)} entries)")

    bgl = get_break_glass_logs()
    if not bgl.empty:
        bgl.to_csv("break_glass_log.csv", index=False)
        print(f"  > break_glass_log.csv saved ({len(bgl)} entries)")

    # ── Part G: Challenge scenarios ───────────────────────────────────────────
    print("\n[Part G] Running challenge scenarios...")
    scenario_df = run_all_scenarios(samples, ft, detector, passports)
    scenario_df.to_csv("scenario_results.csv", index=False)
    print(f"\n  > scenario_results.csv saved ({len(scenario_df)} rows)")

# ── Part H: Baseline comparison ───────────────────────────────────────────
    print("\n[Part H] Running baseline comparison...")
    for net in ["normal", "degraded", "offline"]:
        print(f"\n  -- Network: {net.upper()} --")
        # Notice we are passing the Master Dataset (samples, ft, preds, passports) here!
        df_cmp = run_comparison(net, samples, ft, preds, passports)
        df_cmp.to_csv(f"comparison_{net}.csv")
        try:
            plot_comparison(df_cmp, net)
        except Exception as e:
            print(f"  (Plot skipped: {e})")

    # ── VERIFICATION ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"PIPELINE COMPLETE - All output files routed to ./{out_dir}/:")
    output_files = [
        "ae_samples_meta.csv", "feature_table.csv", "passport_table.csv",
        "decision_output.csv", "audit_log.csv", "scenario_results.csv",
        "comparison_normal.csv", "comparison_degraded.csv", "comparison_offline.csv",
        "comparison_normal.png", "comparison_degraded.png", "comparison_offline.png"
    ]
    for f in output_files:
        size = os.path.getsize(f) if os.path.exists(f) else 0
        print(f"  {'✓' if size > 0 else '✗'} {f} ({size:,} bytes)")
    print("="*70)

if __name__ == "__main__":
    main()