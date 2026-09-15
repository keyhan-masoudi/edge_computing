import numpy as np
import pandas as pd
import uuid
import random

SAMPLING_RATE = 1_000_000  # 1 MHz typical for AE
WINDOW_SIZE = 1024          # samples per window
FAULT_LABELS = ["normal", "friction", "impact", "leakage", "crack_growth",
                "cavitation", "unknown_anomaly"]
NETWORK_STATES = ["normal", "degraded", "offline"]
SENSITIVITY_LEVELS = ["low_risk", "internal", "confidential", "non_transferable"]
RISK_LEVELS = ["low", "medium", "high", "critical"]
EQUIPMENT_TYPES = ["pump", "compressor", "turbine", "gearbox", "bearing",
                   "industrial_valve", "pressure_vessel"]


def generate_ae_window(fault_label: str, risk_level: str) -> np.ndarray:
    """Generate a realistic synthetic AE signal window based on fault type."""
    t = np.linspace(0, WINDOW_SIZE / SAMPLING_RATE, WINDOW_SIZE)
    noise = np.random.normal(0, 0.05, WINDOW_SIZE)

    if fault_label == "normal":
        signal = 0.1 * np.sin(2 * np.pi * 50_000 * t) + noise * 0.3

    elif fault_label == "friction":
        signal = (0.3 * np.sin(2 * np.pi * 200_000 * t) +
                  0.1 * np.sin(2 * np.pi * 400_000 * t) + noise)

    elif fault_label == "impact":
        signal = noise.copy()
        for _ in range(random.randint(3, 8)):
            idx = random.randint(0, WINDOW_SIZE - 50)
            burst_len = random.randint(10, 50)
            signal[idx:idx + burst_len] += np.random.uniform(1.5, 3.0) * np.exp(
                -np.linspace(0, 5, burst_len))

    elif fault_label == "leakage":
        signal = np.random.normal(0, 0.4, WINDOW_SIZE)
        signal *= np.linspace(0.5, 1.5, WINDOW_SIZE)

    elif fault_label == "crack_growth":
        signal = noise.copy()
        n_bursts = random.randint(5, 15)
        positions = sorted(random.sample(range(0, WINDOW_SIZE - 30), n_bursts))
        for i, pos in enumerate(positions):
            amp = 0.5 + i * 0.1
            signal[pos:pos + 20] += amp * np.exp(-np.linspace(0, 4, 20))

    elif fault_label == "cavitation":
        signal = 0.05 * np.sin(2 * np.pi * 300_000 * t) + noise * 0.2
        for _ in range(random.randint(10, 30)):
            idx = random.randint(0, WINDOW_SIZE - 10)
            signal[idx:idx + 10] += np.random.uniform(0.8, 2.0) * np.random.randn(10)

    elif fault_label == "unknown_anomaly":
        signal = (np.random.normal(0, 0.3, WINDOW_SIZE) *
                  (1 + 0.5 * np.sin(2 * np.pi * 10_000 * t)))
        signal += 0.2 * np.sin(2 * np.pi * np.random.uniform(50_000, 500_000) * t)
    else:
        signal = noise

    risk_scale = {"low": 0.5, "medium": 1.0, "high": 1.5, "critical": 2.0}
    signal *= risk_scale.get(risk_level, 1.0)
    return signal.astype(np.float32)


def assign_trust_context(fault_label: str, network_state: str, sensitivity: str) -> dict:
    """Assign a realistic trust context for the sample."""
    if sensitivity == "non_transferable":
        trust_score = random.uniform(0.3, 0.6)
    elif sensitivity == "confidential":
        trust_score = random.uniform(0.5, 0.75)
    else:
        trust_score = random.uniform(0.6, 1.0)

    return {
        "trust_score": round(trust_score, 3),
        "network_trust": "low" if network_state == "offline" else
                         ("medium" if network_state == "degraded" else "high"),
        "node_authenticated": random.choice([True, True, True, False]),
    }


def generate_dataset(n_samples: int = 80) -> list:
    """Generate synthetic AE dataset satisfying all hard constraints."""
    # Ensure minimum sample size per prompt 
    samples = []

    non_transferable_count = 0
    high_critical_count = 0
    degraded_offline_count = 0
    unknown_anomaly_count = 0

    for i in range(n_samples):
        sample_id = f"AE_{i:04d}_{uuid.uuid4().hex[:6]}"
        samples_left = n_samples - i

        # Constraint 1: >= 20 non_transferable
        if non_transferable_count < 20:
            if (20 - non_transferable_count) >= samples_left:
                sensitivity_level = "non_transferable"
            else:
                sensitivity_level = random.choice(["non_transferable"] * 4 + ["low_risk", "internal", "confidential"])
        else:
            sensitivity_level = random.choice(["low_risk", "internal", "confidential"])
            
        if sensitivity_level == "non_transferable":
            non_transferable_count += 1

        # Constraint 2: >= 20 high or critical
        if high_critical_count < 20:
            if (20 - high_critical_count) >= samples_left:
                risk_level = random.choice(["high", "critical"])
            else:
                risk_level = random.choices(RISK_LEVELS, weights=[0.2, 0.2, 0.3, 0.3])[0]
        else:
            risk_level = random.choices(RISK_LEVELS, weights=[0.4, 0.4, 0.1, 0.1])[0]

        if risk_level in ["high", "critical"]:
            high_critical_count += 1

        # Constraint 3: >= 10 degraded or offline
        if degraded_offline_count < 10:
            if (10 - degraded_offline_count) >= samples_left:
                network_state = random.choice(["degraded", "offline"])
            else:
                network_state = random.choices(NETWORK_STATES, weights=[0.4, 0.3, 0.3])[0]
        else:
            network_state = random.choices(NETWORK_STATES, weights=[0.8, 0.1, 0.1])[0]

        if network_state in ["degraded", "offline"]:
            degraded_offline_count += 1

        # Constraint 4: >= 5 unknown_anomaly
        if unknown_anomaly_count < 5:
            if (5 - unknown_anomaly_count) >= samples_left:
                fault_label = "unknown_anomaly"
            else:
                fault_label = random.choices(FAULT_LABELS, weights=[0.1]*6 + [0.4])[0]
        else:
            fault_label = random.choices(FAULT_LABELS[:-1])[0] # Pick from everything except unknown

        if fault_label == "unknown_anomaly":
            unknown_anomaly_count += 1

        energy_level = round(random.uniform(0.1, 5.0), 3)
        if risk_level == "critical":
            energy_level = round(random.uniform(3.0, 5.0), 3)

        signal = generate_ae_window(fault_label, risk_level)
        trust_context = assign_trust_context(fault_label, network_state, sensitivity_level)

        # EXACT 10 FIELDS REQUIRED BY PROMPT
        sample = {
            "sample_id": sample_id,
            "equipment_type": random.choice(EQUIPMENT_TYPES),
            "signal_window": signal,
            "sampling_rate": SAMPLING_RATE,
            "fault_label": fault_label,
            "risk_level": risk_level,
            "network_state": network_state,
            "sensitivity_level": sensitivity_level,
            "energy_level": energy_level,
            "trust_context": trust_context,
        }
        samples.append(sample)

    print(f"[DataGen] Generated {len(samples)} samples successfully.")
    print(f"  > non_transferable: {sum(s['sensitivity_level']=='non_transferable' for s in samples)} (Target: >= 20)")
    print(f"  > high/critical risk: {sum(s['risk_level'] in ['high','critical'] for s in samples)} (Target: >= 20)")
    print(f"  > degraded/offline: {sum(s['network_state'] in ['degraded','offline'] for s in samples)} (Target: >= 10)")
    print(f"  > unknown_anomaly: {sum(s['fault_label']=='unknown_anomaly' for s in samples)} (Target: >= 5)")
    return samples

if __name__ == "__main__":
    samples = generate_dataset(80)
    
    print("\n[Verification] Checking keys of the first generated sample:")
    print(list(samples[0].keys()))
    

    meta = pd.DataFrame([
        {k: v for k, v in s.items() if k != "signal_window"} for s in samples
    ])
    
    meta["trust_context"] = meta["trust_context"].apply(str)
    
    meta.to_csv("ae_samples_meta.csv", index=False)
    
    print("\n[Success] Sample metadata saved to ae_samples_meta.csv")