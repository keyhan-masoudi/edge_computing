import pandas as pd
from data_generator import generate_dataset
from feature_extractor import build_feature_table
from model import AEStateDetector       # Updated to match Part C filename
from passport import create_passport
from decision_engine import decide_ae_action       # Updated to match Part E filename

SCENARIOS = {
    "normal": {
        "description": "Cloud available, some data confidential",
        "network_state": "normal",
        "nodes": {
            "sensor":               {"level": 0, "latency_ms": 1,   "trust": 0.50, "available": True},
            "industrial_gateway":   {"level": 1, "latency_ms": 5,   "trust": 0.65, "available": True},
            "local_edge":           {"level": 2, "latency_ms": 15,  "trust": 0.85, "available": True},
            "datacenter":           {"level": 3, "latency_ms": 80,  "trust": 0.90, "available": True},
            "cloud":                {"level": 4, "latency_ms": 200, "trust": 0.88, "available": True},
        }
    },
    "degraded": {
        "description": "Increased latency, lower trust on some nodes, higher uncertainty",
        "network_state": "degraded",
        "nodes": {
            "sensor":               {"level": 0, "latency_ms": 2,   "trust": 0.45, "available": True},
            "industrial_gateway":   {"level": 1, "latency_ms": 30,  "trust": 0.40, "available": True},
            "local_edge":           {"level": 2, "latency_ms": 80,  "trust": 0.70, "available": True},
            "datacenter":           {"level": 3, "latency_ms": 350, "trust": 0.75, "available": True},
            "cloud":                {"level": 4, "latency_ms": 800, "trust": 0.60, "available": True},
        }
    },
    "offline": {
        "description": "Cloud and datacenter unavailable, decision on gateway or local_edge",
        "network_state": "offline",
        "nodes": {
            "sensor":               {"level": 0, "latency_ms": 1,  "trust": 0.50, "available": True},
            "industrial_gateway":   {"level": 1, "latency_ms": 5,  "trust": 0.65, "available": True},
            "local_edge":           {"level": 2, "latency_ms": 15, "trust": 0.85, "available": True},
            "datacenter":           {"level": 3, "latency_ms": 80, "trust": 0.90, "available": False},  # down
            "cloud":                {"level": 4, "latency_ms": 200,"trust": 0.88, "available": False},  # down
        }
    },
}

def run_scenario(scenario_name: str, config: dict,
                 samples: list, ft: pd.DataFrame,
                 detector: AEStateDetector, passports: list,
                 n_report: int = 8) -> pd.DataFrame:
    """Run all samples under a scenario and return report DataFrame."""
    rows = []
    for i, sample in enumerate(samples):
        feats = ft.iloc[i].to_dict()

        pred = detector.predict_single(feats)
        
        # In degraded scenario: artificially inflate uncertainty to test fallback routing
        if scenario_name == "degraded":
            pred["model_uncertainty"] = min(1.0, pred["model_uncertainty"] + 0.25)
            pred["confidence"] = max(0.0, pred["confidence"] - 0.15)

        passport = passports[i]
        context = {
            "network_state": config["network_state"],
            "deadline_ms": 50 if sample["risk_level"] == "critical" else 500,
        }

        decision = decide_ae_action(sample, feats, pred, passport, config["nodes"], context)

        rows.append({
            "scenario": scenario_name,
            "sample_id": sample["sample_id"],
            "fault_label": sample["fault_label"],
            "risk_level": sample["risk_level"],
            "sensitivity_level": sample["sensitivity_level"],
            "predicted_state": pred["predicted_state"],
            "confidence": pred["confidence"],
            "model_uncertainty": pred["model_uncertainty"],
            "selected_node": decision.selected_node,
            "decision_type": decision.decision_type,
            "data_form": decision.data_form,
            "maintenance_action": decision.maintenance_action,
            "expected_latency_ms": decision.expected_latency_ms,
            "privacy_risk": decision.privacy_risk,
            "trust_status": decision.trust_status,
            "audit_required": decision.audit_required,
            "decision_reason": decision.decision_reason,
        })

    df = pd.DataFrame(rows)
    return df

def run_all_scenarios(samples: list, ft: pd.DataFrame,
                      detector: AEStateDetector, passports: list) -> pd.DataFrame:
    all_results = []
    for name, config in SCENARIOS.items():
        print(f"\n{'='*70}")
        print(f"SCENARIO: {name.upper()} — {config['description']}")
        print('='*70)
        
        df = run_scenario(name, config, samples, ft, detector, passports)
        all_results.append(df)

        # Report first 5 samples per scenario as requested in Part G
        print(df[["sample_id", "fault_label", "predicted_state",
                   "selected_node", "decision_type", "maintenance_action",
                   "expected_latency_ms", "audit_required"]].head(5).to_string(index=False))

        print(f"\n  > decision_type distribution for {name}:")
        print(df["decision_type"].value_counts().to_string())

    return pd.concat(all_results, ignore_index=True)

if __name__ == "__main__":
    # Test the Simulation end-to-end
    samples = generate_dataset(80)
    ft = build_feature_table(samples)
    
    detector = AEStateDetector()
    detector.fit(ft)
    passports = [create_passport(s) for s in samples]

    results = run_all_scenarios(samples, ft, detector, passports)
    results.to_csv("scenario_results.csv", index=False)
    print("\n[Success] scenario_results.csv saved. Ready for LaTeX integration.")