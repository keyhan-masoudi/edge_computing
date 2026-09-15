from data_generator import generate_dataset
from feature_extractor import build_feature_table

# Make sure this matches whatever you named your file in Part C (e.g., model.py or model_classifier.py)
from model import AEStateDetector 
from passport import create_passport

from dataclasses import dataclass, asdict
from typing import Optional, List
import pandas as pd
import time

# ── Node definitions ──────────────────────────────────────────────────────────
DEFAULT_NODES = {
    "sensor": {"level": 0, "latency_ms": 1, "trust": 0.5, "available": True},
    "industrial_gateway": {"level": 1, "latency_ms": 5, "trust": 0.65, "available": True},
    "local_edge": {"level": 2, "latency_ms": 15, "trust": 0.85, "available": True},
    "datacenter": {"level": 3, "latency_ms": 80, "trust": 0.90, "available": True},
    "cloud": {"level": 4, "latency_ms": 200, "trust": 0.88, "available": True},
}

MAINTENANCE_ACTIONS = [
    "continue_operation",
    "increase_monitoring",
    "request_operator_check",
    "schedule_inspection",
    "reduce_load",
    "safe_shutdown",
    "emergency_stop",
]

@dataclass
class AEDecision:
    sample_id: str
    predicted_state: str         
    confidence: float           
    fault_score: float
    selected_node: str
    data_form: str
    decision_type: str           # local_decision / edge_inference / cloud_offloading /
                                 # blocked / human_review / emergency_break_glass
    maintenance_action: str
    expected_latency_ms: float
    privacy_risk: str            # low / medium / high / blocked
    trust_status: str            # trusted / low_trust / untrusted
    audit_required: bool
    decision_reason: str

    def to_dict(self) -> dict:
        return asdict(self)

def _assess_privacy_risk(passport, selected_node: str) -> str:
    if selected_node in passport.forbidden_nodes:
        return "blocked"
    if not passport.raw_transfer_allowed and selected_node in ["cloud", "datacenter"]:
        return "high"
    if passport.sensitivity_level in ["confidential", "non_transferable"]:
        return "medium"
    return "low"

def _assess_trust(selected_node: str, nodes: dict) -> str:
    t = nodes.get(selected_node, {}).get("trust", 0.5)
    if t >= 0.80:
        return "trusted"
    if t >= 0.55:
        return "low_trust"
    return "untrusted"

def _pick_maintenance_action(predicted_state: str, fault_label: str, network_state: str) -> str:
    if predicted_state == "critical" or fault_label in ["crack_growth", "cavitation"]:
        if network_state == "offline":
            return "emergency_stop"
        return "safe_shutdown"
    if predicted_state == "warning":
        if fault_label == "impact":
            return "reduce_load"
        return "request_operator_check"
    if predicted_state == "unknown":
        return "increase_monitoring"
    return "continue_operation"

def decide_ae_action(sample: dict, features: dict, prediction: dict,
                     passport, nodes: dict, context: dict) -> AEDecision:
    """
    Main edge decision function.
    context keys: network_state, deadline_ms (optional), operator_present (bool)
    """
    sample_id = sample["sample_id"]
    network_state = context.get("network_state", sample.get("network_state", "normal"))
    deadline_ms = context.get("deadline_ms", 500)
    sensitivity = passport.sensitivity_level
    predicted_state = prediction.get("predicted_state", "unknown")
    confidence = prediction.get("confidence", 0.0)
    uncertainty = prediction.get("model_uncertainty", 1.0)
    fault_label = sample.get("fault_label", "unknown")

    reasons = []
    available_nodes = {n: info for n, info in nodes.items() if info.get("available", False)}

    # ── Rule 1: Offline network → cloud/datacenter forbidden ────────────────
    if network_state == "offline":
        available_nodes = {n: info for n, info in available_nodes.items() if n not in ["cloud", "datacenter"]}
        reasons.append("network_offline: cloud/datacenter excluded")

    # ── Rule 2: Non-transferable data → enforce forbidden nodes ─────────────
    if sensitivity == "non_transferable":
        available_nodes = {n: info for n, info in available_nodes.items() if n not in passport.forbidden_nodes}
        reasons.append("non_transferable: forbidden nodes excluded")

    # ── Rule 3: Critical + short deadline → local/edge first ────────────────
    if predicted_state == "critical" and deadline_ms <= 100:
        preferred = ["local_edge", "industrial_gateway", "sensor"]
        for p in preferred:
            if p in available_nodes:
                selected_node = p
                decision_type = "local_decision"
                reasons.append(f"critical+short_deadline → {p}")
                break
        else:
            selected_node = list(available_nodes.keys())[0] if available_nodes else "local_edge"
            decision_type = "local_decision"

    # ── Rule 4: High uncertainty + normal network → escalate ────────────────
    elif uncertainty > 0.60 and network_state == "normal":
        selected_node = "local_edge"
        decision_type = "human_review"
        reasons.append(f"high_uncertainty({uncertainty:.2f}) → human_review on local_edge")

    # ── Rule 5: Low trust destination fallback ──────────────────────────────
    elif all(nodes.get(n, {}).get("trust", 0) < 0.60 for n in available_nodes):
        selected_node = "local_edge"
        decision_type = "human_review"
        reasons.append("all_nodes_low_trust → human_review")

    # ── Rule 6: Normal case — pick lowest-latency available node ────────────
    else:
        candidates = sorted(available_nodes.items(), key=lambda x: x[1]["latency_ms"])
        selected_node = candidates[0][0] if candidates else "local_edge"

        if selected_node in ["sensor", "industrial_gateway"]:
            decision_type = "local_decision"
        elif selected_node == "local_edge":
            decision_type = "edge_inference"
        elif selected_node in ["datacenter", "cloud"]:
            if sensitivity == "non_transferable":
                selected_node = "local_edge"
                decision_type = "edge_inference"
                reasons.append("override: non_transferable cannot go to cloud")
            else:
                decision_type = "cloud_offloading"
        else:
            decision_type = "edge_inference"
            
        reasons.append(f"auto_select: {selected_node} (latency={available_nodes.get(selected_node,{}).get('latency_ms','?')}ms)")

    # Evaluate Trust Status early for data_form downgrades
    trust_status = _assess_trust(selected_node, nodes)

    # ── Determine data_form ─────────────────────────────────────────────────
    if "raw" in passport.allowed_data_forms and passport.raw_transfer_allowed:
        data_form = "raw"
    elif "feature" in passport.allowed_data_forms:
        data_form = "feature"
    elif "summary" in passport.allowed_data_forms:
        data_form = "summary"
    else:
        data_form = "alert_only"

    # Hard Override: If destination trust is low, DO NOT send sensitive raw or feature data
    if trust_status in ["low_trust", "untrusted"] and data_form in ["raw", "feature"]:
        data_form = "summary" if "summary" in passport.allowed_data_forms else "alert_only"
        reasons.append(f"low_trust_destination → downgraded_data_form_to_{data_form}")

    # Fallback: Low confidence and unknown network → blocked
    if confidence < 0.20 and predicted_state == "unknown" and network_state == "offline":
        decision_type = "blocked"
        data_form = "alert_only"
        reasons.append("low_confidence+offline → blocked")

    # ── Maintenance action & Audit ──────────────────────────────────────────
    maintenance_action = _pick_maintenance_action(predicted_state, fault_label, network_state)
    
    audit_required = (
        passport.audit_required or
        maintenance_action in ["safe_shutdown", "emergency_stop"] or
        decision_type in ["emergency_break_glass", "human_review"]
    )

    return AEDecision(
        sample_id=sample_id,
        predicted_state=predicted_state,             # <-- ADDED THIS
        confidence=confidence,                       # <-- ADDED THIS
        fault_score=prediction.get("fault_score"),   # <-- ADDED THIS
        selected_node=selected_node,
        data_form=data_form,
        decision_type=decision_type,
        maintenance_action=maintenance_action,
        expected_latency_ms=nodes.get(selected_node, {}).get("latency_ms", 50.0),
        privacy_risk=_assess_privacy_risk(passport, selected_node),
        trust_status=trust_status,
        audit_required=audit_required,
        decision_reason=" | ".join(reasons) if reasons else "standard_routing",
        )

def run_decision_batch(samples: list, feature_rows: pd.DataFrame,
                       pred_rows: pd.DataFrame, passports: list,
                       nodes: dict, network_override: Optional[str] = None) -> pd.DataFrame:
    """Run decision engine over all samples and return decision_output DataFrame."""
    rows = []
    for i, sample in enumerate(samples):
        feats = feature_rows.iloc[i].to_dict()
        pred = pred_rows.iloc[i].to_dict()
        passport = passports[i]
        ctx = {
            "network_state": network_override or sample["network_state"],
            "deadline_ms": 50 if sample["risk_level"] == "critical" else 500,
            "operator_present": True,
        }
        decision = decide_ae_action(sample, feats, pred, passport, nodes, ctx)
        rows.append(decision.to_dict())
    return pd.DataFrame(rows)

if __name__ == "__main__":
    samples = generate_dataset(80)
    ft = build_feature_table(samples)
    
    detector = AEStateDetector()
    detector.fit(ft)
    preds = detector.predict_batch(ft)
    passports = [create_passport(s) for s in samples]

    df = run_decision_batch(samples, ft, preds, passports, DEFAULT_NODES)
    df.to_csv("decision_output.csv", index=False)
    
    print("\n[Success] decision_output.csv saved.")
    print("\nDecision Types Overview:")
    print(df["decision_type"].value_counts())
    print("\nMaintenance Actions Overview:")
    print(df["maintenance_action"].value_counts())