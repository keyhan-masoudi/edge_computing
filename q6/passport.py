from dataclasses import dataclass
from typing import List
import pandas as pd

ALLOWED_DATA_FORMS = ["raw", "feature", "summary", "alert_only", "blocked"]

RETENTION_POLICIES = {
    "non_transferable": "local_only_7d",
    "confidential": "local_30d_no_cloud",
    "internal": "edge_90d",
    "low_risk": "cloud_1y",
}

@dataclass
class DataPassport:
    sample_id: str
    sensitivity_level: str              # low_risk / internal / confidential / non_transferable
    allowed_data_forms: List[str]       # subset of ALLOWED_DATA_FORMS
    forbidden_nodes: List[str]          # node names forbidden from receiving this data
    raw_transfer_allowed: bool
    feature_transfer_allowed: bool
    retention_policy: str
    audit_required: bool

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "sensitivity_level": self.sensitivity_level,
            "allowed_data_forms": "|".join(self.allowed_data_forms),
            "forbidden_nodes": "|".join(self.forbidden_nodes),
            "raw_transfer_allowed": self.raw_transfer_allowed,
            "feature_transfer_allowed": self.feature_transfer_allowed,
            "retention_policy": self.retention_policy,
            "audit_required": self.audit_required,
        }

def create_passport(sample: dict) -> DataPassport:
    """
    Create a DataPassport based on sample metadata.

    Rules:
    - non_transferable: raw signal must NOT leave to datacenter or cloud.
      Only feature/summary/alert_only allowed externally.
    - confidential: feature or summary preferred over raw.
    - critical equipment: audit_required = True.
    - If cloud model is more accurate but data forbidden, local/edge inference only.
    """
    sensitivity = sample["sensitivity_level"]
    risk_level = sample["risk_level"]
    fault_label = sample["fault_label"]

    # Determine allowed forms based on strict network and privacy policies
    if sensitivity == "non_transferable":
        allowed_forms = ["feature", "summary", "alert_only"]
        raw_transfer = False
        feature_transfer = True
        forbidden_nodes = ["cloud", "datacenter"]
    elif sensitivity == "confidential":
        allowed_forms = ["feature", "summary", "alert_only"]
        raw_transfer = False
        feature_transfer = True
        forbidden_nodes = ["cloud"]
    elif sensitivity == "internal":
        allowed_forms = ["feature", "summary"]
        raw_transfer = False
        feature_transfer = True
        forbidden_nodes = []
    else:  # low_risk
        allowed_forms = ["raw", "feature", "summary"]
        raw_transfer = True
        feature_transfer = True
        forbidden_nodes = []

    # Audit required if critical risk or dangerous fault
    audit_required = (
        risk_level in ["high", "critical"] or
        fault_label in ["crack_growth", "cavitation", "impact"] or
        sensitivity == "non_transferable"
    )

    return DataPassport(
        sample_id=sample["sample_id"],
        sensitivity_level=sensitivity,
        allowed_data_forms=allowed_forms,
        forbidden_nodes=forbidden_nodes,
        raw_transfer_allowed=raw_transfer,
        feature_transfer_allowed=feature_transfer,
        retention_policy=RETENTION_POLICIES.get(sensitivity, "edge_90d"),
        audit_required=audit_required,
    )

def build_passport_table(samples: list) -> pd.DataFrame:
    """Build passport table for all samples."""
    rows = [create_passport(s).to_dict() for s in samples]
    return pd.DataFrame(rows)

if __name__ == "__main__":
    from data_generator import generate_dataset
    
    samples = generate_dataset(80)
    df = build_passport_table(samples)
    
    print("\n[Success] Data Passport Preview:")
    print(df[["sample_id", "sensitivity_level", "forbidden_nodes", "audit_required"]].head(10).to_string())
    print(f"\nStats:")
    print(f"  > audit_required: {df['audit_required'].sum()}/{len(df)}")
    print(f"  > non_transferable: {(df['sensitivity_level'] == 'non_transferable').sum()}/{len(df)}")