import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List
import pandas as pd

# ── Node registry ─────────────────────────────────────────────────────────────
NODE_REGISTRY = {
    "industrial_gateway_01": {
        "valid_identity": True,
        "credential_expiry": datetime.now() + timedelta(days=30),
        "trust_score": 0.45,   # deliberately low for scenario 1
        "role": "gateway",
        "permissions": ["feature", "alert_only"],  # NOT raw
        "break_glass_allowed": False,
        "network_zone": "industrial_dmz",
    },
    "cloud_provider_a": {
        "valid_identity": True,
        "credential_expiry": datetime.now() + timedelta(days=365),
        "trust_score": 0.88,
        "role": "cloud",
        "permissions": ["feature", "summary"],
        "break_glass_allowed": False,
        "network_zone": "public_cloud",
    },
    "operator_unit_1": {
        "valid_identity": True,
        "credential_expiry": datetime.now() + timedelta(hours=8),
        "trust_score": 0.75,
        "role": "operator",
        "permissions": ["feature", "summary", "alert_only"],
        "break_glass_allowed": True,
        "network_zone": "plant_floor",
    },
    "maintenance_team_a": {
        "valid_identity": True,
        "credential_expiry": datetime.now() + timedelta(hours=2),
        "trust_score": 0.80,
        "role": "maintenance",
        "permissions": ["feature", "summary"],
        "break_glass_allowed": True,
        "network_zone": "plant_floor",
    },
    "local_edge_01": {
        "valid_identity": True,
        "credential_expiry": datetime.now() + timedelta(days=365),
        "trust_score": 0.92,
        "role": "edge",
        "permissions": ["raw", "feature", "summary", "alert_only"],
        "break_glass_allowed": False,
        "network_zone": "trusted_edge",
    },
}

@dataclass
class ZeroTrustResult:
    request_id: str
    requester: str
    sample_id: str
    data_form_requested: str
    granted: bool
    granted_data_form: Optional[str]
    reason: str
    risk_level: str
    requires_audit: bool
    break_glass_used: bool
    timestamp: str

@dataclass
class BreakGlassLog:
    sample_id: str
    requester: str
    reason: str
    data_form: str
    risk_level: str
    timestamp: str
    expiration_time: str
    requires_review: bool

_break_glass_logs: List[BreakGlassLog] = []
_audit_log_entries: List[dict] = []

def _is_credential_valid(node_id: str) -> bool:
    info = NODE_REGISTRY.get(node_id)
    if not info:
        return False
    return info["credential_expiry"] > datetime.now()

def _has_permission(node_id: str, data_form: str) -> bool:
    info = NODE_REGISTRY.get(node_id)
    if not info:
        return False
    return data_form in info["permissions"]

def zero_trust_check(
    requester: str,
    sample_id: str,
    data_form_requested: str,
    passport,
    network_state: str = "normal",
    emergency_reason: Optional[str] = None,
) -> ZeroTrustResult:
    """
    Evaluate a data access request under Zero Trust principles.
    All parameters are evaluated; no implicit trust is assumed.
    """
    request_id = f"ZT-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now().isoformat()
    node_info = NODE_REGISTRY.get(requester)
    break_glass_used = False

    # ── Check 1: Valid identity ──────────────────────────────────────────────
    if not node_info or not node_info.get("valid_identity"):
        result = ZeroTrustResult(
            request_id=request_id, requester=requester, sample_id=sample_id,
            data_form_requested=data_form_requested, granted=False,
            granted_data_form=None,
            reason="DENIED: unknown or invalid identity",
            risk_level="critical", requires_audit=True, break_glass_used=False,
            timestamp=timestamp,
        )
        _audit_log_entries.append(asdict(result))
        return result

    # ── Check 2: Expired credential ──────────────────────────────────────────
    if not _is_credential_valid(requester):
        result = ZeroTrustResult(
            request_id=request_id, requester=requester, sample_id=sample_id,
            data_form_requested=data_form_requested, granted=False,
            granted_data_form=None,
            reason="DENIED: credential expired",
            risk_level="high", requires_audit=True, break_glass_used=False,
            timestamp=timestamp,
        )
        _audit_log_entries.append(asdict(result))
        return result

    # ── Check 3: Offline → external nodes blocked ────────────────────────────
    if network_state == "offline" and node_info["network_zone"] in ["public_cloud", "industrial_dmz"]:
        result = ZeroTrustResult(
            request_id=request_id, requester=requester, sample_id=sample_id,
            data_form_requested=data_form_requested, granted=False,
            granted_data_form=None,
            reason="DENIED: network offline, external node blocked",
            risk_level="medium", requires_audit=False, break_glass_used=False,
            timestamp=timestamp,
        )
        _audit_log_entries.append(asdict(result))
        return result

    trust_score = node_info["trust_score"]

    # ── Check 4: Data permission (least privilege) ───────────────────────────
    if not _has_permission(requester, data_form_requested):
        # Scenario 1: gateway requests raw but not permitted
        if requester == "industrial_gateway_01" and data_form_requested == "raw":
            result = ZeroTrustResult(
                request_id=request_id, requester=requester, sample_id=sample_id,
                data_form_requested=data_form_requested, granted=False,
                granted_data_form=None,
                reason=(f"DENIED: industrial_gateway has low trust ({trust_score}) "
                        "and is not permitted to receive raw AE data. "
                        "Downgrade to feature data only."),
                risk_level="high", requires_audit=True, break_glass_used=False,
                timestamp=timestamp,
            )
            _audit_log_entries.append(asdict(result))
            return result

        result = ZeroTrustResult(
            request_id=request_id, requester=requester, sample_id=sample_id,
            data_form_requested=data_form_requested, granted=False,
            granted_data_form=None,
            reason=f"DENIED: {requester} lacks permission for {data_form_requested}",
            risk_level="medium", requires_audit=True, break_glass_used=False,
            timestamp=timestamp,
        )
        _audit_log_entries.append(asdict(result))
        return result

    # ── Check 5: Passport transfer restrictions ──────────────────────────────
    if data_form_requested == "raw" and not passport.raw_transfer_allowed:
        # Scenario 2: cloud requests raw but non-transferable
        if requester == "cloud_provider_a":
            result = ZeroTrustResult(
                request_id=request_id, requester=requester, sample_id=sample_id,
                data_form_requested=data_form_requested, granted=False,
                granted_data_form=None,
                reason=("DENIED: cloud has better model but AE data is non-transferable. "
                        "Inference must be performed locally or at on-site edge."),
                risk_level="critical", requires_audit=True, break_glass_used=False,
                timestamp=timestamp,
            )
            _audit_log_entries.append(asdict(result))
            return result

        result = ZeroTrustResult(
            request_id=request_id, requester=requester, sample_id=sample_id,
            data_form_requested=data_form_requested, granted=False,
            granted_data_form=None,
            reason=f"DENIED: passport forbids raw transfer (sensitivity={passport.sensitivity_level})",
            risk_level="high", requires_audit=True, break_glass_used=False,
            timestamp=timestamp,
        )
        _audit_log_entries.append(asdict(result))
        return result

    if requester in passport.forbidden_nodes:
        result = ZeroTrustResult(
            request_id=request_id, requester=requester, sample_id=sample_id,
            data_form_requested=data_form_requested, granted=False,
            granted_data_form=None,
            reason=f"DENIED: {requester} is in passport forbidden_nodes",
            risk_level="critical", requires_audit=True, break_glass_used=False,
            timestamp=timestamp,
        )
        _audit_log_entries.append(asdict(result))
        return result

    # ── Check 6: Low trust score ─────────────────────────────────────────────
    if trust_score < 0.60:
        # Emergency break-glass override
        if node_info.get("break_glass_allowed") and emergency_reason:
            bg_log = BreakGlassLog(
                sample_id=sample_id,
                requester=requester,
                reason=emergency_reason,
                data_form=data_form_requested,
                risk_level="critical",
                timestamp=timestamp,
                expiration_time=(datetime.now() + timedelta(hours=1)).isoformat(),
                requires_review=True,
            )
            _break_glass_logs.append(bg_log)
            break_glass_used = True
        else:
            result = ZeroTrustResult(
                request_id=request_id, requester=requester, sample_id=sample_id,
                data_form_requested=data_form_requested, granted=False,
                granted_data_form=None,
                reason=f"DENIED: trust_score={trust_score} below threshold 0.60",
                risk_level="high", requires_audit=True, break_glass_used=False,
                timestamp=timestamp,
            )
            _audit_log_entries.append(asdict(result))
            return result

    # ── Scenario 3: Emergency operator break-glass access ───────────────────
    if emergency_reason and not break_glass_used and node_info.get("break_glass_allowed"):
        bg_log = BreakGlassLog(
            sample_id=sample_id,
            requester=requester,
            reason=emergency_reason,
            data_form=data_form_requested,
            risk_level="high",
            timestamp=timestamp,
            expiration_time=(datetime.now() + timedelta(hours=1)).isoformat(),
            requires_review=True,
        )
        _break_glass_logs.append(bg_log)
        break_glass_used = True

    # ── Granted ─────────────────────────────────────────────────────────────
    granted_form = data_form_requested
    reason = "GRANTED"
    if break_glass_used:
        reason = f"GRANTED via break-glass emergency override. Reason: {emergency_reason}"

    result = ZeroTrustResult(
        request_id=request_id, requester=requester, sample_id=sample_id,
        data_form_requested=data_form_requested, granted=True,
        granted_data_form=granted_form,
        reason=reason,
        risk_level="low" if trust_score >= 0.80 else "medium",
        requires_audit=break_glass_used or passport.audit_required,
        break_glass_used=break_glass_used,
        timestamp=timestamp,
    )
    _audit_log_entries.append(asdict(result))
    return result

def get_break_glass_logs() -> pd.DataFrame:
    if not _break_glass_logs:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "sample_id": b.sample_id, "requester": b.requester,
            "reason": b.reason, "data_form": b.data_form,
            "risk_level": b.risk_level, "timestamp": b.timestamp,
            "expiration_time": b.expiration_time,
            "requires_review": b.requires_review,
        }
        for b in _break_glass_logs
    ])

def get_audit_log() -> pd.DataFrame:
    if not _audit_log_entries:
        return pd.DataFrame()
    return pd.DataFrame(_audit_log_entries)

def demo_zero_trust_scenarios(samples: list, passports: list) -> None:
    """Run the three required Zero Trust scenarios."""
    print("\n" + "="*70)
    print("ZERO TRUST SCENARIOS")
    print("="*70)

    # Pick a non-transferable sample for scenarios 1 & 2
    nt_sample = next(
        (s for s in samples if s["sensitivity_level"] == "non_transferable"), samples[0]
    )
    nt_passport = passports[samples.index(nt_sample)]

    # Scenario 1: Gateway with low trust requests raw AE data
    print("\n[Scenario 1] Industrial gateway (low trust) requests raw AE data:")
    r1 = zero_trust_check("industrial_gateway_01", nt_sample["sample_id"],
                           "raw", nt_passport, "normal")
    print(f"  Granted: {r1.granted} | Reason: {r1.reason}")

    # Scenario 2: Cloud wants to run better model, but data is non-transferable
    print("\n[Scenario 2] Cloud requests raw AE (better model, but data forbidden):")
    r2 = zero_trust_check("cloud_provider_a", nt_sample["sample_id"],
                           "raw", nt_passport, "normal")
    print(f"  Granted: {r2.granted} | Reason: {r2.reason}")

    # Scenario 3: Operator needs emergency access
    print("\n[Scenario 3] Operator/maintenance requests emergency (break-glass) access:")
    r3 = zero_trust_check("operator_unit_1", nt_sample["sample_id"],
                           "feature", nt_passport, "normal",
                           emergency_reason="critical_failure_imminent_pump_01")
    print(f"  Granted: {r3.granted} | Break-glass: {r3.break_glass_used}")
    print(f"  Reason: {r3.reason}")

    bgl = get_break_glass_logs()
    if not bgl.empty:
        print("\n[Break-Glass Log]")
        print(bgl.to_string())

if __name__ == "__main__":
    from data_generator import generate_dataset
    from passport import create_passport

    samples = generate_dataset(80)
    passports = [create_passport(s) for s in samples]
    demo_zero_trust_scenarios(samples, passports)

    audit = get_audit_log()
    if not audit.empty:
        audit.to_csv("audit_log.csv", index=False)
        print(f"\n[Success] audit_log.csv saved ({len(audit)} entries)")