import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy
from scipy.signal import welch
from data_generator import generate_dataset

SAMPLING_RATE = 1_000_000  # 1 MHz typical for AE


def extract_features(signal: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    """
    Time-domain features (8):
      - rms              : Root Mean Square — overall signal energy level; rises with friction, leakage
      - peak_amplitude   : Maximum absolute value — indicator of high-energy impact events
      - crest_factor     : Peak / RMS — high values indicate transient impact or crack growth
      - energy           : Sum of squares — total signal power in the window
      - zero_crossing_rate: Frequency of sign changes — related to dominant frequency content
      - kurtosis_val     : 4th statistical moment — very sensitive to impulsive faults (impact, crack)
      - skewness_val     : 3rd moment — asymmetry; asymmetric bursts indicate non-stationary damage
      - signal_entropy   : Information entropy of amplitude histogram — disorder indicator

    Frequency-domain features (3):
      - dominant_frequency: Frequency with maximum PSD — shifts with fault type
      - spectral_centroid : Weighted mean frequency — higher in friction/cavitation
      - band_power_ratio  : Power in 100-400 kHz band vs total — AE active band

    Transient event features (1):
      - burst_count       : Number of bursts above adaptive threshold — count of AE events
    """
    signal = signal.astype(np.float64)
    n = len(signal)

    # ── Time-domain ──────────────────────────────────────────────
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak_amplitude = float(np.max(np.abs(signal)))
    crest_factor = float(peak_amplitude / (rms + 1e-12))
    energy = float(np.sum(signal ** 2))
    zcr = float(np.sum(np.diff(np.sign(signal)) != 0) / n)
    kurtosis_val = float(kurtosis(signal))
    skewness_val = float(skew(signal))

    # Amplitude histogram entropy (32 bins)
    hist, _ = np.histogram(signal, bins=32, density=True)
    hist = hist + 1e-12  # avoid log(0)
    hist /= hist.sum()
    sig_entropy = float(entropy(hist))

    # ── Frequency-domain ─────────────────────────────────────────
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, n))
    dominant_freq = float(freqs[np.argmax(psd)])
    spectral_centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))

    # Band power 100 kHz – 400 kHz (typical AE band)
    band_mask = (freqs >= 100_000) & (freqs <= 400_000)
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    band_power = float(_trapz(psd[band_mask], freqs[band_mask])) if band_mask.any() else 0.0
    total_power = float(_trapz(psd, freqs)) + 1e-12
    band_power_ratio = band_power / total_power

    # ── Transient bursts ─────────────────────────────────────────
    threshold = rms * 3.0  # 3x RMS adaptive threshold
    above = (np.abs(signal) > threshold).astype(int)
    burst_count = int(np.sum(np.diff(above) == 1))  # rising edges = burst starts

    return {
        "rms": round(rms, 6),
        "peak_amplitude": round(peak_amplitude, 6),
        "crest_factor": round(crest_factor, 4),
        "energy": round(energy, 6),
        "zero_crossing_rate": round(zcr, 6),
        "kurtosis_val": round(kurtosis_val, 4),
        "skewness_val": round(skewness_val, 4),
        "signal_entropy": round(sig_entropy, 4),
        "dominant_frequency": round(dominant_freq, 2),
        "spectral_centroid": round(spectral_centroid, 2),
        "band_power_ratio": round(band_power_ratio, 6),
        "burst_count": burst_count,
    }


def build_feature_table(samples: list) -> pd.DataFrame:
    """Extract features for all samples and return a DataFrame."""
    rows = []
    for s in samples:
        feats = extract_features(s["signal_window"], s["sampling_rate"])
        
        # Flatten trust_context for the dataframe
        trust = s.get("trust_context", {})
        
        row = {
            "sample_id": s["sample_id"],
            "equipment_type": s["equipment_type"],
            "fault_label": s["fault_label"],
            "risk_level": s["risk_level"],
            "network_state": s["network_state"],
            "sensitivity_level": s["sensitivity_level"],
            "energy_level": s["energy_level"],
            "trust_score": trust.get("trust_score", 1.0),
            "network_trust": trust.get("network_trust", "high"),
            "node_authenticated": trust.get("node_authenticated", True)
        }
        row.update(feats)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    samples = generate_dataset(80)
    df = build_feature_table(samples)
    df.to_csv("feature_table.csv", index=False)
    print("\n[Success] feature_table.csv saved successfully.")
    print("Preview of extracted features:")
    print(df[["sample_id", "fault_label", "rms", "burst_count", "spectral_centroid"]].head())