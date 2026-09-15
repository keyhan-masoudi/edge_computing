import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from data_generator import generate_dataset
from feature_extractor import build_feature_table
from model import AEStateDetector
from passport import create_passport
from decision_engine import decide_ae_action, DEFAULT_NODES

def first_cloud_decision(sample, features, prediction, passport, nodes, context):

    # Force selection of 'cloud' regardless of network state or availability
    if context["network_state"] == "offline":
        selected = "industrial_gateway"
    else:
        selected = "cloud"
    dtype = (
        "local_decision"
        if selected == "industrial_gateway"
        else "cloud_offloading"
    )

    # Privacy violation check remains: non-transferable data sent to cloud
    privacy_violation = (passport.sensitivity_level == "non_transferable")
    nt_violation = privacy_violation

    data_form = "raw" 

    maint = "continue_operation"
    if prediction["predicted_state"] == "critical":
        maint = "safe_shutdown"
    elif prediction["predicted_state"] == "warning":
        maint = "request_operator_check"

    # Audit still required if critical equipment is involved
    audit = passport.audit_required or maint == "safe_shutdown"

    return {
        "selected_node": selected,
        "decision_type": dtype,
        "data_form": data_form,
        "maintenance_action": maint,
        "expected_latency_ms": nodes.get(selected, {}).get(
            "latency_ms",
            15 if selected != "cloud" else 200
        ),
        "audit_required": audit,
        "privacy_violation": privacy_violation,
        "nt_violation": nt_violation,
        "human_review": False,
    }

def first_latency_decision(sample, features, prediction, passport, nodes, context):
    avail = [(n, info) for n, info in nodes.items() if info.get("available", False)]
    avail.sort(key=lambda x: x[1]["latency_ms"])
    selected = avail[0][0] if avail else "sensor"

    privacy_violation = (passport.sensitivity_level == "non_transferable" and selected in ["cloud", "datacenter"])

    if selected in ["sensor", "industrial_gateway"]:
        dtype = "local_decision"
    elif selected == "local_edge":
        dtype = "edge_inference"
    else:
        dtype = "cloud_offloading"

    maint = "continue_operation"
    if prediction["predicted_state"] == "critical":
        maint = "safe_shutdown"
    elif prediction["predicted_state"] == "warning":
        maint = "request_operator_check"

    return {
        "selected_node": selected,
        "decision_type": dtype,
        "data_form": "raw",
        "maintenance_action": maint,
        "expected_latency_ms": nodes.get(selected, {}).get("latency_ms", 5),
        "audit_required": passport.audit_required or maint == "safe_shutdown",
        "privacy_violation": privacy_violation,
        "nt_violation": privacy_violation,
        "human_review": False, 
    }

def our_engine_decision(sample, features, prediction, passport, nodes, context):
    d = decide_ae_action(sample, features, prediction, passport, nodes, context)
    privacy_violation = (d.privacy_risk == "blocked") or (
        passport.sensitivity_level == "non_transferable" and d.selected_node in ["cloud", "datacenter"]
    )
    return {
        "selected_node": d.selected_node,
        "decision_type": d.decision_type,
        "data_form": d.data_form,
        "maintenance_action": d.maintenance_action,
        "expected_latency_ms": d.expected_latency_ms,
        "audit_required": d.audit_required,
        "privacy_violation": privacy_violation,
        "nt_violation": privacy_violation,
        "human_review": d.decision_type == "human-review" or d.decision_type == "human_review",
    }

METHODS = {
    "our_engine": our_engine_decision,
    "first_cloud": first_cloud_decision,
    "first_latency": first_latency_decision,
}

def compute_metrics(results: list, samples: list, predictions: pd.DataFrame, modified_preds: list) -> dict:
    latencies = []

    for r in results:
        total = r["expected_latency_ms"]

        if r["audit_required"]:
            total += 5

        if r["human_review"]:
            total += 50
        latencies.append(total)
        
    avg_latency = round(np.mean(latencies), 2)

    critical_indices = [
        i for i, s in enumerate(samples)
        if s["risk_level"] in ["high", "critical"] or s["fault_label"] in ["crack_growth", "cavitation"]
    ]
    critical_detected = sum(
        1 for i in critical_indices
        if modified_preds[i]["predicted_state"] in ["critical", "warning"]
    )
    critical_detection_rate = round(critical_detected / len(critical_indices) if critical_indices else 0.0, 4)

    false_alarms = sum(
        1 for i, r in enumerate(results)
        if modified_preds[i]["predicted_state"] in ["critical", "warning"]
        and samples[i]["fault_label"] == "normal"
    )

    return {
        "average_latency": avg_latency,
        "critical_detection_rate": critical_detection_rate,
        "false_alarm_count": false_alarms,
        "privacy_violation_count": sum(1 for r in results if r["privacy_violation"]),
        "non_transferable_violation_count": sum(1 for r in results if r["nt_violation"]),
        "cloud_usage_count": sum(1 for r in results if r["selected_node"] in ["cloud", "datacenter"]),
        "human_review_count": sum(1 for r in results if r["human_review"]),
        "audit_required_count": sum(1 for r in results if r["audit_required"]),
    }

# NOTICE: We now pass the exact same dataset via arguments!
def run_comparison(network_state: str, samples: list, ft: pd.DataFrame, preds: pd.DataFrame, passports: list) -> pd.DataFrame:
    nodes = {}
    for n, info in DEFAULT_NODES.items():
        ninfo = info.copy()
        if network_state == "offline" and n in ["cloud", "datacenter"]:
            ninfo["available"] = False
        if network_state == "degraded":
            ninfo["latency_ms"] = info["latency_ms"] * 2
            ninfo["trust"] = max(0.2, info["trust"] - 0.15) 
        nodes[n] = ninfo

    rows = {}
    for method_name, method_fn in METHODS.items():
        results = []
        modified_preds = [] 
        
        for i, sample in enumerate(samples):
            feats = ft.iloc[i].to_dict()
            pred = preds.iloc[i].to_dict()
            
            # Injecting identical realistic noise across all scenarios
            if i % 12 == 0 and sample["fault_label"] == "normal":
                pred["predicted_state"] = "warning"
            if sample["fault_label"] == "unknown_anomaly":
                pred["model_uncertainty"] = 0.85
                pred["confidence"] = 0.15
                
            modified_preds.append(pred)
            
            context = {
                "network_state": network_state,
                "deadline_ms": 50 if sample["risk_level"] == "critical" else 500,
            }
            r = method_fn(sample, feats, pred, passports[i], nodes, context)
            results.append(r)

        metrics = compute_metrics(results, samples, preds, modified_preds)
        rows[method_name] = metrics

    return pd.DataFrame(rows).T

def plot_comparison(df: pd.DataFrame, scenario: str) -> None:
    """Plot comparison bar charts with robust scaling for zero-values."""
    metrics = ["average_latency", "critical_detection_rate", "privacy_violation_count", 
               "non_transferable_violation_count", "cloud_usage_count", "human_review_count"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # Define explicit colors for the three methods: [our_engine, first_cloud, first_latency]
    colors = ["#2196F3", "#FF9800", "#4CAF50"] 

    for ax, metric in zip(axes.flatten(), metrics):
        vals = df[metric].values
        # Map colors to the specific index of the DataFrame
        bars = ax.bar(df.index, vals, color=colors)
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        
        # --- ROBUST SCALING LOGIC ---
        max_val = max(vals)
        if max_val == 0:
            ax.set_ylim(0, 1)  # Force a visible range if data is empty
            offset = 0.05      # Small static offset for text
        else:
            ax.set_ylim(0, max_val * 1.25) # Add 25% headroom
            offset = max_val * 0.04        # Proportional offset
            
        # Add labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, 
                    bar.get_height() + offset,
                    f"{val:.2f}", 
                    ha="center", va="bottom", fontsize=9, fontweight='bold')

    fig.suptitle(f"Baseline Comparison — {scenario.upper()} Network", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"comparison_{scenario}.png", dpi=120, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # Local test block generates ONE dataset and tests all three
    test_samples = generate_dataset(80)
    test_ft = build_feature_table(test_samples)
    test_detector = AEStateDetector()
    test_detector.fit(test_ft)
    test_preds = test_detector.predict_batch(test_ft)
    test_passports = [create_passport(s) for s in test_samples]
    
    print(run_comparison("normal", test_samples, test_ft, test_preds, test_passports))