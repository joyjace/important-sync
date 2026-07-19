#!/usr/bin/env python3
"""
Build DQN dataset by merging CSI and ACK data with raw IQ features and reward.

Supports six reward modes:
    1. Delay minimization: reward = -service_ms (minimize latency)
    2. PDR maximization: reward = delivered (0/1 per packet)
    3. Fast-latency: reward highest for packets in the very low-delay region
    4. Log-delay: reward = -log(service_ms), continuous and threshold-free
    5. Robust-delay: reward = -log1p(clipped_service_ms / scale), capped for tail robustness
    6. Utility: bounded log-goodput reward with explicit loss and optional tail-delay penalties

Usage:
    # Delay-optimized dataset
    python build_dqn_dataset.py --csi-csv csi_data.csv --ack-csv ack_data.csv \\
                                 --output rl_dqn_dataset_delay.csv --reward-mode delay

    # PDR-optimized dataset
    python build_dqn_dataset.py --csi-csv csi_data.csv --ack-csv ack_data.csv \\
                                 --output rl_dqn_dataset_pdr.csv --reward-mode pdr

    # Fast-latency dataset
    python build_dqn_dataset.py --csi-csv csi_data.csv --ack-csv ack_data.csv \\
                                 --output rl_dqn_dataset_fast_latency.csv \\
                                 --reward-mode fast_latency

    # Continuous low-latency dataset
    python build_dqn_dataset.py --csi-csv csi_data.csv --ack-csv ack_data.csv \\
                                 --output rl_dqn_dataset_log_delay.csv \\
                                 --reward-mode log_delay

    # Robust low-latency dataset with heavy-tail suppression
    python build_dqn_dataset.py --csi-csv csi_data.csv --ack-csv ack_data.csv \
                                 --output rl_dqn_dataset_robust_delay.csv \
                                 --reward-mode robust_delay \
                                 --robust-cap-quantile 0.99 \
                                 --robust-scale-ms 0.25

Output CSV columns:
    seq: packet sequence ID
    state_seq: sequence ID of the CSI state used for this decision (causal mode)
    state_mcs_index: MCS that produced the CSI state (causal mode)
    state_age_packets: ACK-decision rows since the CSI state was observed (causal mode)
    state_packet_gap: target sequence gap from state_seq (causal mode)
    state_missing_packets: missing-packet estimate known before the target decision
    state_is_stale: 1 when the decision state is older than the immediate next packet
    mcs_index: MCS 0-7
    delivered: binary delivery outcome
    service_ms: service time in milliseconds
    reward: per-packet reward (depends on reward_mode)
    iq_raw: legacy amplitude vector (57 HT20 bins, padded later by trainers)
    iq_features: extracted IQ statistics (mean, std, p10, p50, p90 amplitudes)
    iq_active_amplitudes/iq_phase_diff_*: optional versioned full-CSI transform
    rssi, snr, fft_gain, agc_gain, channel, sig_len: CSI features
"""

import argparse
import hashlib
import json
import csv
import platform
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from .csi_link_v7c import (
        CONTRACT_SHA256 as LINK_V7C_CONTRACT_SHA256,
        COMPLEX_BIN_COUNT as LINK_V7C_COMPLEX_BIN_COUNT,
        INPUT_SCALAR_COUNT as LINK_V7C_INPUT_SCALAR_COUNT,
        SCHEMA_ID as LINK_V7C_CONTRACT_ID,
        contract_manifest_dict as link_v7c_contract_manifest,
        contract_sidecar as link_v7c_contract_sidecar,
        extract_features as extract_link_v7c_features,
        parse_compensated_iq_json,
    )
except ImportError:
    from csi_link_v7c import (
        CONTRACT_SHA256 as LINK_V7C_CONTRACT_SHA256,
        COMPLEX_BIN_COUNT as LINK_V7C_COMPLEX_BIN_COUNT,
        INPUT_SCALAR_COUNT as LINK_V7C_INPUT_SCALAR_COUNT,
        SCHEMA_ID as LINK_V7C_CONTRACT_ID,
        contract_manifest_dict as link_v7c_contract_manifest,
        contract_sidecar as link_v7c_contract_sidecar,
        extract_features as extract_link_v7c_features,
        parse_compensated_iq_json,
    )


LEGACY_CSI_FEATURE_CONTRACT = "legacy_amplitude_v1"
CSI_FEATURE_CONTRACTS = (LEGACY_CSI_FEATURE_CONTRACT, LINK_V7C_CONTRACT_ID)
LINK_V7C_CAPTURE_FORMATS = frozenset({"b64", "c5c6"})


CSI_STATE_COLUMNS = [
    "rssi",
    "noise_floor",
    "fft_gain",
    "agc_gain",
    "channel",
    "sig_len",
    "data_json",
]

# Optional capture fields that make a CSI feature row auditable.  They are not
# required for legacy datasets, but the link_v7c contract validates and retains
# them so training and firmware cannot silently disagree about the wire format.
CSI_PROVENANCE_COLUMNS = [
    "host_time",
    "format",
    "mac",
    "rate",
    "local_timestamp",
    "rx_state",
    "data_len",
    "first_word",
    "iq_pairs",
    "b64_version",
    "gain_compensation_exact",
]

B64_EXACT_GAIN_VERSION = 2


def carried_csi_columns(csi_df):
    """Return state plus available capture-provenance columns in stable order."""
    return [
        *CSI_STATE_COLUMNS,
        *(column for column in CSI_PROVENANCE_COLUMNS if column in csi_df.columns),
    ]


def extract_iq_features(iq_json_str):
    """
    Extract statistical features from raw IQ data JSON string.
    
    Returns dict with keys:
        - iq_mean, iq_std, iq_p10, iq_p50, iq_p90: amplitude statistics
        - iq_raw: numpy array of 234 values (amplitudes)
    """
    try:
        if pd.isna(iq_json_str) or iq_json_str == "":
            # Return NaN-filled dict
            return {
                "iq_mean": np.nan,
                "iq_std": np.nan,
                "iq_p10": np.nan,
                "iq_p50": np.nan,
                "iq_p90": np.nan,
                "iq_raw": np.full(234, np.nan),
            }
        
        # Parse IQ data (interleaved I, Q pairs)
        iq_values = json.loads(iq_json_str)
        iq_array = np.array(iq_values, dtype=np.float32)
        
        # Compute amplitudes from I/Q pairs: sqrt(I^2 + Q^2)
        iq_pairs = iq_array.reshape(-1, 2)  # Shape: (117, 2)
        amplitudes = np.sqrt(np.sum(iq_pairs**2, axis=1))  # Shape: (117,)
        
        return {
            "iq_mean": float(np.mean(amplitudes)),
            "iq_std": float(np.std(amplitudes)),
            "iq_p10": float(np.percentile(amplitudes, 10)),
            "iq_p50": float(np.percentile(amplitudes, 50)),
            "iq_p90": float(np.percentile(amplitudes, 90)),
            "iq_raw": amplitudes,  # Raw amplitudes for DQN input
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {
            "iq_mean": np.nan,
            "iq_std": np.nan,
            "iq_p10": np.nan,
            "iq_p50": np.nan,
            "iq_p90": np.nan,
            "iq_raw": np.full(117, np.nan),
        }


def _required_integral_capture_value(row, column):
    """Read a required integer-valued capture field with a useful error."""
    if column not in row.index or pd.isna(row[column]) or row[column] == "":
        raise ValueError(f"link_v7c capture is missing required {column!r}")
    try:
        numeric = float(row[column])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"link_v7c capture has non-numeric {column!r}") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"link_v7c capture has non-integral {column!r}: {row[column]!r}")
    return int(numeric)


def extract_link_v7c_row_features(row):
    """Validate one normalized HT20 capture row and compute link_v7c features."""
    raw_iq = validate_link_v7c_capture_row(row)
    features = extract_link_v7c_features(raw_iq)
    legacy_features = extract_iq_features(row.get("data_json", ""))
    legacy_amplitudes = np.asarray(legacy_features["iq_raw"], dtype=np.float32)
    amplitudes = np.asarray(features.amplitudes, dtype=np.float32)
    phase_real = np.asarray(features.phase_real, dtype=np.float32)
    phase_imag = np.asarray(features.phase_imag, dtype=np.float32)

    return {
        # Preserve the exact legacy 57-bin amplitude view and summary columns.
        # Existing amplitude-only trainers can still build a valid baseline;
        # the new contract lives only in explicitly named v7 columns.
        "iq_mean": legacy_features["iq_mean"],
        "iq_std": legacy_features["iq_std"],
        "iq_p10": legacy_features["iq_p10"],
        "iq_p50": legacy_features["iq_p50"],
        "iq_p90": legacy_features["iq_p90"],
        "iq_raw": legacy_amplitudes,
        "iq_active_amplitudes": amplitudes,
        "iq_phase_diff_real": phase_real,
        "iq_phase_diff_imag": phase_imag,
        "iq_phase_valid_fraction": float(features.valid_fraction),
        "iq_phase_coherence": float(features.coherence),
    }


def validate_link_v7c_capture_row(row):
    """Return strict compensated int16[114] input for one usable capture row."""
    if "format" not in row.index or pd.isna(row["format"]) or not str(row["format"]).strip():
        raise ValueError("link_v7c capture is missing required 'format'")
    capture_format = str(row["format"]).strip()
    if capture_format not in LINK_V7C_CAPTURE_FORMATS:
        raise ValueError(
            f"link_v7c capture format {capture_format!r} is not one of "
            f"{sorted(LINK_V7C_CAPTURE_FORMATS)}"
        )

    data_len = _required_integral_capture_value(row, "data_len")
    if data_len != LINK_V7C_INPUT_SCALAR_COUNT:
        raise ValueError(
            f"link_v7c_ht20_v1 requires exactly {LINK_V7C_INPUT_SCALAR_COUNT} "
            f"compensated IQ scalars ({LINK_V7C_COMPLEX_BIN_COUNT} complex bins), "
            f"got data_len={data_len}"
        )

    first_word = _required_integral_capture_value(row, "first_word")
    if first_word != 0:
        raise ValueError(
            "link_v7c_ht20_v1 rejects CSI with first_word_invalid set; "
            f"got first_word={first_word}"
        )

    if "iq_pairs" in row.index:
        iq_pairs = _required_integral_capture_value(row, "iq_pairs")
        if iq_pairs != LINK_V7C_COMPLEX_BIN_COUNT:
            raise ValueError(
                "link_v7c_ht20_v1 requires "
                f"iq_pairs={LINK_V7C_COMPLEX_BIN_COUNT}, got {iq_pairs}"
            )

    return parse_compensated_iq_json(row.get("data_json", ""))


def filter_link_v7c_capture_rows(csi_df):
    """Quarantine malformed source rows before causal state alignment.

    Older serial captures can contain an isolated partially parsed line. A
    strict full-CSI artifact must never transform that line, but losing the
    entire scenario would unnecessarily require recollection. Filtering before
    alignment makes the next decision use the latest earlier *valid* CSI state.
    """
    valid_indices = []
    rejection_reasons = {}
    for index, row in csi_df.iterrows():
        try:
            validate_link_v7c_capture_row(row)
        except (TypeError, ValueError) as exc:
            reason = str(exc)
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        else:
            valid_indices.append(index)

    report = {
        "source_row_count": int(len(csi_df)),
        "retained_row_count": int(len(valid_indices)),
        "rejected_row_count": int(len(csi_df) - len(valid_indices)),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }
    if not valid_indices:
        reasons = "; ".join(
            f"{reason} ({count} row(s))"
            for reason, count in report["rejection_reasons"].items()
        )
        raise ValueError(f"No valid {LINK_V7C_CONTRACT_ID} capture rows: {reasons}")
    return csi_df.loc[valid_indices].copy(), report


def sha256_file(path):
    """Return a streaming SHA-256 for an input artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_contract_sidecar_path(output_csv):
    """Place the contract next to the exact CSV name, including its suffix."""
    return Path(f"{Path(output_csv)}.feature_contract.json")


def validate_feature_contract_record(
    record,
    *,
    expected_id=None,
    expected_sha256=None,
):
    """Verify that a sidecar's embedded contract matches its claimed digest."""
    if not isinstance(record, dict):
        raise ValueError("feature contract must be a JSON object")
    schema_id = record.get("feature_contract_id")
    claimed_digest = record.get("feature_contract_sha256")
    contract = record.get("contract")
    if not isinstance(schema_id, str) or not isinstance(claimed_digest, str):
        raise ValueError("feature contract is missing its schema ID or SHA-256")
    if not isinstance(contract, dict):
        raise ValueError("feature contract is missing its canonical contract document")
    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    actual_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if claimed_digest != actual_digest:
        raise ValueError(
            f"feature contract digest mismatch: claimed {claimed_digest}, got {actual_digest}"
        )
    if contract.get("schema_id") != schema_id:
        raise ValueError("feature contract schema ID disagrees with its contract document")
    contract_input_count = contract.get("input", {}).get("scalar_count")
    contract_feature_count = contract.get("output", {}).get("feature_count")
    if record.get("input_scalar_count") != contract_input_count:
        raise ValueError("feature contract input count disagrees with its contract document")
    if record.get("feature_count") != contract_feature_count:
        raise ValueError("feature contract output count disagrees with its contract document")
    if "feature_names" in record and record["feature_names"] != contract.get("output", {}).get("feature_names"):
        raise ValueError("feature names disagree with the canonical contract document")
    if expected_id is not None and schema_id != expected_id:
        raise ValueError(f"expected feature contract {expected_id}, got {schema_id}")
    if expected_sha256 is not None and claimed_digest != expected_sha256:
        raise ValueError(
            f"expected feature contract digest {expected_sha256}, got {claimed_digest}"
        )
    return schema_id, claimed_digest


def _json_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def metadata_csv_value(value):
    """Keep scalar metadata scalar; serialize structured values canonically."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _capture_value_counts(csi_df, column):
    if column not in csi_df.columns:
        return None
    counts = csi_df[column].value_counts(dropna=False)
    result = {}
    for value, count in counts.items():
        key_value = _json_scalar(value)
        key = "<missing>" if key_value is None else str(key_value)
        result[key] = int(count)
    return dict(sorted(result.items()))


def capture_provenance_summary(csi_df):
    """Summarize fields that determine whether capture and live input match."""
    summary = {
        column: _capture_value_counts(csi_df, column)
        for column in (
            "format",
            "data_len",
            "first_word",
            "iq_pairs",
            "mac",
            "rx_state",
            "b64_version",
            "gain_compensation_exact",
        )
        if column in csi_df.columns
    }
    if "local_timestamp" in csi_df.columns:
        timestamp = pd.to_numeric(csi_df["local_timestamp"], errors="coerce").dropna()
        summary["local_timestamp"] = {
            "min": _json_scalar(timestamp.min()) if not timestamp.empty else None,
            "max": _json_scalar(timestamp.max()) if not timestamp.empty else None,
            "note": "wrapping device microsecond counter; not wall-clock time",
        }
    return summary


def _exact_gain_flag(value):
    """Return True only for an explicit CSV boolean/1 exactness marker."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) == 1
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value) and float(value) == 1.0)
    return str(value).strip().lower() in {"1", "true"}


def gain_compensation_qualification(csi_df):
    """Classify whether every compact B64 row has exact gain reconstruction.

    ``c5c6`` rows already contain the firmware-compensated int16 values and do
    not need a wire gain. Historical B64 v1 remains valid training input, but
    its six-decimal gain cannot establish byte-for-byte live-input parity.
    """
    if "format" not in csi_df.columns:
        return {
            "exact": False,
            "b64_frame_count": 0,
            "exact_b64_frame_count": 0,
            "inexact_or_unknown_b64_frame_count": 0,
            "device_compensated_frame_count": 0,
            "required_b64_version": B64_EXACT_GAIN_VERSION,
            "reason": "capture format provenance is missing",
        }

    formats = csi_df["format"].fillna("").astype(str).str.strip().str.lower()
    b64_mask = formats.eq("b64")
    b64_count = int(b64_mask.sum())
    device_compensated_count = int(formats.eq("c5c6").sum())
    if b64_count == 0:
        return {
            "exact": True,
            "b64_frame_count": 0,
            "exact_b64_frame_count": 0,
            "inexact_or_unknown_b64_frame_count": 0,
            "device_compensated_frame_count": device_compensated_count,
            "required_b64_version": B64_EXACT_GAIN_VERSION,
            "reason": None,
        }

    if "b64_version" in csi_df.columns:
        versions = pd.to_numeric(csi_df.loc[b64_mask, "b64_version"], errors="coerce")
        version_exact = versions.eq(B64_EXACT_GAIN_VERSION)
    else:
        version_exact = pd.Series(False, index=csi_df.index[b64_mask], dtype=bool)

    if "gain_compensation_exact" in csi_df.columns:
        exact_flags = csi_df.loc[b64_mask, "gain_compensation_exact"].map(_exact_gain_flag)
    else:
        exact_flags = pd.Series(False, index=csi_df.index[b64_mask], dtype=bool)

    exact_count = int((version_exact & exact_flags).sum())
    inexact_count = b64_count - exact_count
    reason = None
    if inexact_count:
        reason = (
            f"{inexact_count} B64 capture frame(s) lack exact gain compensation; "
            f"deployment requires b64_version={B64_EXACT_GAIN_VERSION} and "
            "gain_compensation_exact=1"
        )
    return {
        "exact": inexact_count == 0,
        "b64_frame_count": b64_count,
        "exact_b64_frame_count": exact_count,
        "inexact_or_unknown_b64_frame_count": inexact_count,
        "device_compensated_frame_count": device_compensated_count,
        "required_b64_version": B64_EXACT_GAIN_VERSION,
        "reason": reason,
    }


def combine_parent_qualifications(parents):
    """Conservatively combine qualification records for a merged artifact."""
    qualifications = [parent.get("qualification", {}) for parent in parents]
    causal_alignment = bool(qualifications) and all(
        item.get("causal_alignment") is True for item in qualifications
    )
    gain_exact = bool(qualifications) and all(
        item.get("gain_compensation_exact") is True for item in qualifications
    )
    parents_qualified = bool(qualifications) and all(
        item.get("deployment_candidate") is True for item in qualifications
    )

    blocking_reasons = []
    if not causal_alignment:
        blocking_reasons.append("one or more parent datasets are not causally aligned")
    if not gain_exact:
        blocking_reasons.append(
            "one or more parent datasets lack exact gain-compensation provenance"
        )
    if not parents_qualified:
        blocking_reasons.append(
            "one or more parent datasets are not deployment candidates"
        )
    for item in qualifications:
        for reason in item.get("blocking_reasons", []):
            parent_reason = f"parent dataset: {reason}"
            if parent_reason not in blocking_reasons:
                blocking_reasons.append(parent_reason)

    deployment_candidate = (
        causal_alignment and gain_exact and parents_qualified and not blocking_reasons
    )
    if deployment_candidate:
        status = "candidate"
    elif not causal_alignment:
        status = "noncausal"
    else:
        status = "blocked"
    return {
        "causal_alignment": causal_alignment,
        "gain_compensation_exact": gain_exact,
        "deployment_candidate": deployment_candidate,
        "status": status,
        "blocking_reasons": blocking_reasons,
    }


def _source_artifact(path, row_count=None, expected_sha256=None):
    path_obj = Path(path)
    actual_sha256 = sha256_file(path_obj)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"source artifact changed while the dataset was being built: {path_obj}")
    artifact = {
        "path": str(path_obj.resolve()),
        "sha256": actual_sha256,
    }
    if row_count is not None:
        artifact["row_count"] = int(row_count)
    return artifact


def write_link_v7c_dataset_sidecar(
    *,
    output_csv,
    output_df,
    csi_csv,
    csi_df,
    source_csi_df,
    capture_filter_report,
    ack_csv,
    ack_df,
    metadata_json,
    metadata,
    dedup_strategy,
    reward_manifest,
    state_alignment,
    stale_ages,
    stale_augment_source,
    source_hashes,
):
    """Write an auditable contract/lineage record next to a v7 dataset."""
    output_path = Path(output_csv)
    sources = {
        "csi_csv": _source_artifact(
            csi_csv,
            len(source_csi_df),
            source_hashes.get("csi_csv"),
        ),
        "ack_csv": _source_artifact(
            ack_csv,
            len(ack_df),
            source_hashes.get("ack_csv"),
        ),
    }
    if metadata_json:
        sources["metadata_json"] = _source_artifact(
            metadata_json,
            expected_sha256=source_hashes.get("metadata_json"),
        )

    feature_contract = link_v7c_contract_sidecar(include_feature_names=True)
    feature_contract["contract"] = link_v7c_contract_manifest()
    validate_feature_contract_record(
        feature_contract,
        expected_id=LINK_V7C_CONTRACT_ID,
        expected_sha256=LINK_V7C_CONTRACT_SHA256,
    )
    causal_alignment = state_alignment == "previous_csi"
    gain_qualification = gain_compensation_qualification(csi_df)
    blocking_reasons = []
    if not causal_alignment:
        blocking_reasons.append(
            "same_packet alignment uses CSI unavailable when its action was selected"
        )
    if gain_qualification["reason"]:
        blocking_reasons.append(gain_qualification["reason"])
    deployment_candidate = not blocking_reasons
    if deployment_candidate:
        qualification_status = "candidate"
    elif not causal_alignment:
        qualification_status = "noncausal"
    else:
        qualification_status = "blocked"
    sidecar = {
        "manifest_schema": "dqn_csi_feature_contract/v1",
        "artifact": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "row_count": int(len(output_df)),
            "columns": list(output_df.columns),
        },
        "feature_contract": feature_contract,
        "build": {
            "dedup_strategy": dedup_strategy,
            "reward": reward_manifest,
            "state_alignment": state_alignment,
            "stale_augment_ages": list(stale_ages),
            "stale_augment_source": stale_augment_source,
        },
        "capture_validation": {
            "accepted_formats": sorted(LINK_V7C_CAPTURE_FORMATS),
            "data_len": LINK_V7C_INPUT_SCALAR_COUNT,
            "iq_pairs": LINK_V7C_COMPLEX_BIN_COUNT,
            "first_word_invalid": 0,
            "scalar_order_per_bin": ["imaginary", "real"],
            "gain_compensation": {
                "device_compensated_formats": ["c5c6"],
                "exact_b64_version": B64_EXACT_GAIN_VERSION,
                "exact_b64_marker": "gain_compensation_exact=1",
                "legacy_b64_training_allowed": True,
            },
            "source_filter": capture_filter_report,
        },
        "sources": sources,
        "scenario_metadata": metadata,
        "capture_provenance": capture_provenance_summary(csi_df),
        "source_capture_provenance": capture_provenance_summary(source_csi_df),
        "qualification": {
            "causal_alignment": causal_alignment,
            "gain_compensation_exact": gain_qualification["exact"],
            "gain_compensation": gain_qualification,
            "deployment_candidate": deployment_candidate,
            "status": qualification_status,
            "blocking_reasons": blocking_reasons,
        },
        "producer": {
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(__file__),
            },
            "transform": {
                "path": str(Path(__file__).with_name("csi_link_v7c.py").resolve()),
                "sha256": sha256_file(Path(__file__).with_name("csi_link_v7c.py")),
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
    }

    sidecar_path = feature_contract_sidecar_path(output_path)
    temporary_path = sidecar_path.with_name(f"{sidecar_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(sidecar, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary_path.replace(sidecar_path)
    print(f"  Feature contract: {sidecar_path}")


def compute_snr(rssi, noise_floor):
    """Compute SNR = RSSI - noise_floor (dB)."""
    if pd.notna(rssi) and pd.notna(noise_floor):
        return float(rssi - noise_floor)
    return np.nan


def parse_stale_augment_ages(value):
    """Parse a comma-separated list of positive stale ages."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).split(",")
    ages = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        age = int(text)
        if age < 1:
            raise ValueError("stale augmentation ages must be positive")
        ages.append(age)
    return sorted(set(ages))


def deduplicate_csi(csi_df, dedup_strategy):
    """Return at most one CSI observation for each packet sequence ID."""
    if "seq_or_id" in csi_df.columns:
        csi_df = csi_df.rename(columns={"seq_or_id": "seq"})

    required = {"seq", *CSI_STATE_COLUMNS}
    missing = sorted(required - set(csi_df.columns))
    if missing:
        raise ValueError(f"CSI CSV missing required columns: {missing}")

    csi_df = csi_df.copy()
    if dedup_strategy == "keep_first":
        return csi_df.drop_duplicates(subset=["seq"], keep="first").reset_index(drop=True)
    if dedup_strategy == "keep_latest":
        if "host_time" not in csi_df.columns:
            raise ValueError("keep_latest requires host_time in the CSI CSV")
        return (
            csi_df.sort_values("host_time")
            .drop_duplicates(subset=["seq"], keep="last")
            .reset_index(drop=True)
        )
    if dedup_strategy == "aggregate":
        aggregation = {
            "rssi": "mean",
            "noise_floor": "mean",
            "fft_gain": "mean",
            "agc_gain": "mean",
            "channel": "first",
            "sig_len": "mean",
            "data_json": "first",
        }
        for column in CSI_PROVENANCE_COLUMNS:
            if column in csi_df.columns:
                aggregation[column] = "first"
        return csi_df.groupby("seq", as_index=False).agg(aggregation)

    raise ValueError(f"Unknown dedup strategy: {dedup_strategy}")


def align_packet_decisions(csi_df, ack_df, dedup_strategy, state_alignment):
    """
    Align each action/reward row with its model state.

    ``same_packet`` preserves the original behavior. ``previous_csi`` uses the
    latest CSI observation from a strictly earlier ACK row, which matches the
    information available when selecting the target packet's MCS.
    """
    ack_columns = ["seq", "mcs_index", "delivered", "service_us"]
    missing_ack = sorted(set(ack_columns) - set(ack_df.columns))
    if missing_ack:
        raise ValueError(f"ACK CSV missing required columns: {missing_ack}")

    csi_dedup = deduplicate_csi(csi_df, dedup_strategy)
    print(f"  CSI rows after dedup: {len(csi_dedup)}")

    ack_timeline = ack_df[ack_columns].copy()
    ack_timeline["_ack_order"] = np.arange(len(ack_timeline), dtype=np.int64)

    if state_alignment == "same_packet":
        merged = pd.merge(
            csi_dedup,
            ack_timeline,
            on="seq",
            how="inner",
            sort=False,
        )
        print(f"  Same-packet matched rows: {len(merged)}")
        return merged.reset_index(drop=True)

    if state_alignment != "previous_csi":
        raise ValueError(f"Unknown state alignment: {state_alignment}")

    duplicate_ack_seq = ack_timeline["seq"].duplicated(keep=False)
    if duplicate_ack_seq.any():
        examples = ack_timeline.loc[duplicate_ack_seq, "seq"].head(5).tolist()
        raise ValueError(
            "previous_csi alignment requires unique ACK sequence IDs within a run; "
            f"duplicates include {examples}"
        )

    state_columns = carried_csi_columns(csi_dedup)
    csi_for_merge = csi_dedup[["seq", *state_columns]].copy()
    csi_for_merge["_target_has_csi"] = True
    aligned = ack_timeline.merge(
        csi_for_merge,
        on="seq",
        how="left",
        sort=False,
        validate="one_to_one",
    ).sort_values("_ack_order").reset_index(drop=True)

    has_csi = aligned["_target_has_csi"].notna().to_numpy()
    row_positions = np.arange(len(aligned), dtype=np.int64)
    observation_positions = pd.Series(
        np.where(has_csi, row_positions, np.nan),
        dtype=float,
    )
    source_positions = observation_positions.ffill().shift(1)
    usable = source_positions.notna().to_numpy()

    causal = aligned.loc[usable].copy().reset_index(drop=True)
    source_positions_int = source_positions.loc[usable].astype(np.int64).to_numpy()
    observed_csi = aligned.loc[:, state_columns]
    for column in state_columns:
        causal[column] = observed_csi.iloc[source_positions_int][column].to_numpy()

    causal["state_seq"] = aligned.iloc[source_positions_int]["seq"].to_numpy()
    causal["state_mcs_index"] = (
        aligned.iloc[source_positions_int]["mcs_index"].to_numpy(dtype=np.int64)
    )
    causal["state_age_packets"] = (
        causal["_ack_order"].to_numpy() - source_positions_int
    ).astype(np.int64)
    causal["target_has_csi"] = has_csi[usable].astype(np.int8)

    target_seq_numeric = pd.to_numeric(causal["seq"], errors="coerce")
    state_seq_numeric = pd.to_numeric(causal["state_seq"], errors="coerce")
    sequence_gap = target_seq_numeric - state_seq_numeric
    usable_sequence_gap = sequence_gap.notna() & (sequence_gap >= 1)
    causal["state_packet_gap"] = np.where(
        usable_sequence_gap.to_numpy(),
        sequence_gap.fillna(0).to_numpy(dtype=np.float64),
        causal["state_age_packets"].to_numpy(dtype=np.float64),
    ).astype(np.int64)
    causal["state_missing_packets"] = np.maximum(
        causal["state_packet_gap"].to_numpy(dtype=np.int64) - 1,
        causal["state_age_packets"].to_numpy(dtype=np.int64) - 1,
    ).clip(min=0).astype(np.int64)
    causal["state_is_stale"] = (
        (causal["state_age_packets"] > 1)
        | (causal["state_missing_packets"] > 0)
    ).astype(np.int8)

    warmup_dropped = int((~usable).sum())
    matched_targets = int(has_csi.sum())
    retained_losses = int((causal["delivered"] == 0).sum())
    print(f"  Target ACK rows with CSI: {matched_targets}/{len(aligned)}")
    print(f"  Causal rows after prior-CSI warmup: {len(causal)}")
    print(f"  Warmup rows without earlier CSI dropped: {warmup_dropped}")
    print(f"  Undelivered ACK outcomes retained: {retained_losses}")

    if not (causal["state_age_packets"] >= 1).all():
        raise AssertionError("Causal state must come from a strictly earlier ACK row")

    return causal


def add_synthetic_stale_rows(output_df, stale_ages, stale_source):
    """
    Duplicate observed rows into no-new-CSI states with larger age/gap values.

    This does not invent positive counterfactual rewards. By default it expands
    only loss/no-CSI rows, teaching the DQN that repeating the logged action
    under increasingly stale CSI is risky. Recovery behavior should come from
    real rows where lower-MCS packets succeed after stale/loss periods.
    """
    stale_ages = parse_stale_augment_ages(stale_ages)
    if not stale_ages:
        return output_df
    if "state_age_packets" not in output_df.columns:
        raise ValueError("--stale-augment-ages requires --state-alignment previous_csi")
    if stale_source not in {"loss_only", "loss_or_missing", "all"}:
        raise ValueError(f"Unknown stale augmentation source: {stale_source}")

    augmented = output_df.copy()
    augmented["synthetic_stale"] = 0
    augmented["row_kind"] = "observed"

    if stale_source == "loss_only":
        eligible = augmented["delivered"].astype(int) == 0
    elif stale_source == "loss_or_missing":
        eligible = augmented["delivered"].astype(int) == 0
        if "target_has_csi" in augmented.columns:
            eligible = eligible | (augmented["target_has_csi"].astype(int) == 0)
    else:
        eligible = pd.Series(True, index=augmented.index)

    synthetic_frames = []
    for age in stale_ages:
        base = augmented.loc[
            eligible & (augmented["state_age_packets"].astype(int) < age)
        ].copy()
        if base.empty:
            continue
        base["state_age_packets"] = age
        if "state_packet_gap" in base.columns:
            base["state_packet_gap"] = np.maximum(
                base["state_packet_gap"].astype(int).to_numpy(),
                age,
            )
        else:
            base["state_packet_gap"] = age
        base["state_missing_packets"] = np.maximum(
            base["state_packet_gap"].astype(int).to_numpy() - 1,
            age - 1,
        ).clip(min=0).astype(np.int64)
        base["state_is_stale"] = 1
        base["synthetic_stale"] = 1
        base["row_kind"] = f"stale_age_{age}"
        synthetic_frames.append(base)

    if not synthetic_frames:
        return augmented

    return pd.concat([augmented, *synthetic_frames], ignore_index=True)


def build_dataset(
    csi_csv,
    ack_csv,
    output_csv,
    dedup_strategy="keep_first",
    reward_mode="delay",
    metadata_json=None,
    fast_target_ms=0.25,
    fast_ok_ms=0.50,
    robust_cap_quantile=0.99,
    robust_scale_ms=0.25,
    robust_loss_penalty_ms=5.0,
    utility_payload_bytes=128,
    utility_goodput_quantile=0.95,
    utility_loss_reward=-1.0,
    utility_tail_target_ms=0.0,
    utility_tail_weight=0.0,
    state_alignment="same_packet",
    stale_augment_ages=None,
    stale_augment_source="loss_only",
    csi_feature_contract=LEGACY_CSI_FEATURE_CONTRACT,
):
    """
    Merge CSI and ACK data, extract features, compute reward.
    
    Args:
        csi_csv: Path to CSI CSV file
        ack_csv: Path to ACK CSV file
        output_csv: Path to output DQN dataset
        dedup_strategy: 'keep_first', 'keep_latest', or 'aggregate'
        reward_mode: 'delay' (minimize -service_ms), 'pdr' (maximize delivery, +0/+1),
                     'fast_latency' (maximize very-low-latency packets),
                     'log_delay' (continuous threshold-free low-delay objective),
                     'robust_delay' (log-scaled clipped delay objective), or
                     'utility' (bounded log-goodput with loss/tail penalties)
        metadata_json: Path to JSON file with scenario metadata (optional)
                      Expected format: {"distance_m": 5.0, "obstacle_type": "none", 
                                        "movement_type": "static", "channel_condition": "LOS", ...}
        fast_target_ms: Full-reward service-time threshold for fast_latency
        fast_ok_ms: Partial-reward service-time threshold for fast_latency
        state_alignment: 'same_packet' for legacy behavior or 'previous_csi'
                         for a deployable causal decision state
        stale_augment_ages: optional comma-separated/list of no-new-CSI ages
        stale_augment_source: 'loss_only', 'loss_or_missing', or 'all'
        csi_feature_contract: legacy amplitude extraction or the versioned
                              compact HT20 full-CSI transform
    """
    print(f"[DQN Dataset Builder]")
    print(f"  Loading CSI from: {csi_csv}")
    print(f"  Loading ACK from: {ack_csv}")
    print(f"  Dedup strategy: {dedup_strategy}")
    print(f"  Reward mode: {reward_mode}")
    print(f"  State alignment: {state_alignment}")
    if csi_feature_contract not in CSI_FEATURE_CONTRACTS:
        raise ValueError(
            f"Unknown CSI feature contract {csi_feature_contract!r}; "
            f"choose one of {CSI_FEATURE_CONTRACTS}"
        )
    print(f"  CSI feature contract: {csi_feature_contract}")
    stale_ages = parse_stale_augment_ages(stale_augment_ages)
    if stale_ages:
        print(f"  Synthetic stale ages: {stale_ages} ({stale_augment_source})")
    source_hashes = {}
    
    # Load metadata if provided
    metadata = {}
    if metadata_json:
        try:
            if csi_feature_contract == LINK_V7C_CONTRACT_ID:
                source_hashes["metadata_json"] = sha256_file(metadata_json)
            with open(metadata_json, 'r', encoding='utf-8') as f:
                metadata_payload = json.load(f)
            if (
                csi_feature_contract == LINK_V7C_CONTRACT_ID
                and sha256_file(metadata_json) != source_hashes["metadata_json"]
            ):
                raise ValueError("metadata changed while it was being read")
            if isinstance(metadata_payload, dict):
                metadata = metadata_payload
            elif (
                isinstance(metadata_payload, list)
                and len(metadata_payload) == 1
                and isinstance(metadata_payload[0], dict)
            ):
                # Scenario capture metadata is currently stored as a singleton
                # JSON array.  Accept it without losing the object fields.
                metadata = metadata_payload[0]
            else:
                raise ValueError("metadata must be an object or a singleton array of one object")
            print(f"  Metadata loaded from: {metadata_json}")
            print(f"    Scenario: {metadata.get('scenario_id', metadata.get('scenario', 'N/A'))}")
            print(f"    Distance: {metadata.get('distance_m', 'N/A')} m")
            print(f"    Location: {metadata.get('location', 'N/A')}")
            print(f"    Motion: {metadata.get('motion', metadata.get('movement_type', 'N/A'))}")
            print(f"    Environment: {metadata.get('environment', metadata.get('channel_condition', 'N/A'))}")
        except Exception as e:
            if csi_feature_contract == LINK_V7C_CONTRACT_ID:
                raise ValueError(f"Could not load scenario metadata {metadata_json}: {e}") from e
            print(f"  WARNING: Could not load metadata: {e}")
    
    # Load CSI data
    try:
        if csi_feature_contract == LINK_V7C_CONTRACT_ID:
            source_hashes["csi_csv"] = sha256_file(csi_csv)
        csi_df = pd.read_csv(csi_csv)
        if (
            csi_feature_contract == LINK_V7C_CONTRACT_ID
            and sha256_file(csi_csv) != source_hashes["csi_csv"]
        ):
            raise ValueError("CSI CSV changed while it was being read")
        print(f"  CSI rows: {len(csi_df)}")
    except Exception as e:
        raise ValueError(f"Could not load CSI CSV {csi_csv}: {e}") from e
    
    # Load ACK data
    try:
        if csi_feature_contract == LINK_V7C_CONTRACT_ID:
            source_hashes["ack_csv"] = sha256_file(ack_csv)
        ack_df = pd.read_csv(ack_csv)
        if (
            csi_feature_contract == LINK_V7C_CONTRACT_ID
            and sha256_file(ack_csv) != source_hashes["ack_csv"]
        ):
            raise ValueError("ACK CSV changed while it was being read")
        print(f"  ACK rows: {len(ack_df)}")
    except Exception as e:
        raise ValueError(f"Could not load ACK CSV {ack_csv}: {e}") from e

    if csi_feature_contract == LINK_V7C_CONTRACT_ID and dedup_strategy == "aggregate":
        raise ValueError(
            "link_v7c_ht20_v1 does not support aggregate CSI deduplication: "
            "averaged RF metadata plus one raw CSI frame is not one observation"
        )

    source_csi_df = csi_df
    capture_filter_report = {
        "source_row_count": int(len(csi_df)),
        "retained_row_count": int(len(csi_df)),
        "rejected_row_count": 0,
        "rejection_reasons": {},
    }
    if csi_feature_contract == LINK_V7C_CONTRACT_ID:
        csi_df, capture_filter_report = filter_link_v7c_capture_rows(csi_df)
        rejected_rows = capture_filter_report["rejected_row_count"]
        if rejected_rows:
            print(
                f"  Quarantined invalid CSI rows: {rejected_rows}/"
                f"{capture_filter_report['source_row_count']}"
            )
            for reason, count in capture_filter_report["rejection_reasons"].items():
                print(f"    {count}x {reason}")
    
    merged_dedup = align_packet_decisions(
        csi_df,
        ack_df,
        dedup_strategy=dedup_strategy,
        state_alignment=state_alignment,
    )
    if merged_dedup.empty:
        raise ValueError("No CSI states could be aligned with ACK decision rows")
    
    # Extract features
    print("  Extracting IQ features...")
    iq_features_list = []
    iq_raw_list = []
    iq_cache = {}
    state_key_column = "state_seq" if state_alignment == "previous_csi" else "seq"
    
    for idx, row in merged_dedup.iterrows():
        state_key = row[state_key_column]
        if state_key not in iq_cache:
            try:
                if csi_feature_contract == LINK_V7C_CONTRACT_ID:
                    iq_cache[state_key] = extract_link_v7c_row_features(row)
                else:
                    iq_cache[state_key] = extract_iq_features(row.get("data_json", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"CSI state {state_key!r} violates {csi_feature_contract}: {exc}") from exc
        iq_feats = iq_cache[state_key]
        iq_features_list.append(iq_feats)
        iq_raw_list.append(iq_feats["iq_raw"])
        
        if (idx + 1) % 1000 == 0:
            print(f"    Processed {idx + 1}/{len(merged_dedup)} packets...")
    print(f"  Unique CSI states extracted: {len(iq_cache)}")
    
    # Create feature dataframe
    iq_df = pd.DataFrame(iq_features_list)
    
    # Compute SNR
    merged_dedup.loc[:, "snr"] = merged_dedup.apply(
        lambda row: compute_snr(row["rssi"], row["noise_floor"]), axis=1
    )
    
    # Compute reward based on reward_mode
    merged_dedup.loc[:, "service_ms"] = merged_dedup["service_us"].fillna(0) / 1000.0
    reward_manifest = {"mode": reward_mode, "parameters": {}, "derived": {}}

    if reward_mode == "delay":
        # Minimize service time: reward = -service_ms
        merged_dedup.loc[:, "reward"] = -merged_dedup["service_ms"]
        reward_description = "Delay minimization (-service_ms)"
        reward_manifest["parameters"] = {"formula": "-service_ms"}
    elif reward_mode == "pdr":
        # Maximize packet delivery: reward = delivered (0 or 1)
        merged_dedup.loc[:, "reward"] = merged_dedup["delivered"].astype(float)
        reward_description = "PDR maximization (delivered 0/1)"
        reward_manifest["parameters"] = {"formula": "delivered"}
    elif reward_mode == "fast_latency":
        if fast_target_ms <= 0 or fast_ok_ms <= 0:
            raise ValueError("fast latency thresholds must be positive")
        if fast_target_ms > fast_ok_ms:
            raise ValueError("fast_target_ms must be <= fast_ok_ms")

        service_ms = merged_dedup["service_ms"]
        delivered = merged_dedup["delivered"].astype(bool)
        reward = np.full(len(merged_dedup), -1.0, dtype=np.float32)
        reward[delivered & (service_ms <= fast_ok_ms)] = 0.5
        reward[delivered & (service_ms <= fast_target_ms)] = 1.0
        merged_dedup.loc[:, "reward"] = reward
        reward_manifest["parameters"] = {
            "target_ms": float(fast_target_ms),
            "ok_ms": float(fast_ok_ms),
            "slow_or_lost_reward": -1.0,
        }
        reward_description = (
            f"Fast latency (+1 <= {fast_target_ms:.3f} ms, "
            f"+0.5 <= {fast_ok_ms:.3f} ms, -1 slow/lost)"
        )
    elif reward_mode == "log_delay":
        service_ms = merged_dedup["service_ms"].clip(lower=1e-6)
        reward = -np.log(service_ms).astype(np.float32)
        reward[~merged_dedup["delivered"].astype(bool)] = -10.0
        merged_dedup.loc[:, "reward"] = reward
        reward_description = "Log delay (-log(service_ms), lost=-10)"
        reward_manifest["parameters"] = {
            "minimum_service_ms": 1e-6,
            "loss_reward": -10.0,
        }
    elif reward_mode == "robust_delay":
        if not 0.0 < robust_cap_quantile <= 1.0:
            raise ValueError("robust_cap_quantile must be in (0, 1]")
        if robust_scale_ms <= 0.0:
            raise ValueError("robust_scale_ms must be positive")
        if robust_loss_penalty_ms < 0.0:
            raise ValueError("robust_loss_penalty_ms must be non-negative")

        service_ms = merged_dedup["service_ms"].clip(lower=0.0)
        cap_ms = max(float(service_ms.quantile(robust_cap_quantile)), 1e-6)
        clipped_service = service_ms.clip(upper=cap_ms)
        reward = -np.log1p(clipped_service / robust_scale_ms).astype(np.float32)
        if (~merged_dedup["delivered"].astype(bool)).any():
            loss_cost = cap_ms + robust_loss_penalty_ms
            # Keep dtype stable (float32) for boolean masked assignment across pandas versions.
            reward[~merged_dedup["delivered"].astype(bool)] = np.float32(-np.log1p(loss_cost / robust_scale_ms))
        merged_dedup.loc[:, "reward"] = reward
        reward_manifest["parameters"] = {
            "cap_quantile": float(robust_cap_quantile),
            "scale_ms": float(robust_scale_ms),
            "loss_penalty_ms": float(robust_loss_penalty_ms),
        }
        reward_manifest["derived"] = {"cap_ms": float(cap_ms)}
        reward_description = (
            "Robust delay (-log1p(min(service_ms, cap_ms)/scale_ms), "
            f"cap_q={robust_cap_quantile:.3f}, scale={robust_scale_ms:.3f} ms, "
            f"loss_penalty={robust_loss_penalty_ms:.3f} ms)"
        )
    elif reward_mode == "utility":
        if utility_payload_bytes <= 0:
            raise ValueError("utility_payload_bytes must be positive")
        if not 0.0 < utility_goodput_quantile <= 1.0:
            raise ValueError("utility_goodput_quantile must be in (0, 1]")
        if utility_tail_target_ms < 0.0:
            raise ValueError("utility_tail_target_ms must be non-negative")
        if utility_tail_weight < 0.0:
            raise ValueError("utility_tail_weight must be non-negative")

        service_ms = merged_dedup["service_ms"].clip(lower=1e-6).astype(float)
        delivered = merged_dedup["delivered"].astype(bool)
        payload_bits = float(utility_payload_bytes * 8)

        # bits/ms is numerically equal to kbit/s. The log keeps the reward
        # stable while preserving action ordering by useful delivered goodput.
        goodput_kbps = np.where(delivered, payload_bits / service_ms, 0.0)
        utility_score = np.log1p(goodput_kbps).astype(np.float32)

        delivered_scores = utility_score[delivered.to_numpy()]
        if len(delivered_scores):
            scale = max(float(np.quantile(delivered_scores, utility_goodput_quantile)), 1e-6)
        else:
            scale = 1.0

        reward = (2.0 * np.clip(utility_score / scale, 0.0, 1.0) - 1.0).astype(np.float32)
        reward[~delivered.to_numpy()] = np.float32(utility_loss_reward)

        if utility_tail_target_ms > 0.0 and utility_tail_weight > 0.0:
            tail_excess = np.maximum((service_ms.to_numpy() - utility_tail_target_ms) / utility_tail_target_ms, 0.0)
            tail_penalty = utility_tail_weight * np.clip(tail_excess, 0.0, 1.0)
            reward = (reward - tail_penalty).astype(np.float32)

        reward = np.clip(reward, -1.0, 1.0).astype(np.float32)
        merged_dedup.loc[:, "reward"] = reward
        reward_manifest["parameters"] = {
            "payload_bytes": int(utility_payload_bytes),
            "goodput_quantile": float(utility_goodput_quantile),
            "loss_reward": float(utility_loss_reward),
            "tail_target_ms": float(utility_tail_target_ms),
            "tail_weight": float(utility_tail_weight),
        }
        reward_manifest["derived"] = {"log_goodput_scale": float(scale)}
        reward_description = (
            "Utility (bounded log1p(goodput_kbps), "
            f"payload={utility_payload_bytes} bytes, q={utility_goodput_quantile:.3f}, "
            f"loss_reward={utility_loss_reward:.3f}, "
            f"tail_target={utility_tail_target_ms:.3f} ms, tail_weight={utility_tail_weight:.3f})"
        )
    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")
    
    print(f"  Reward mode: {reward_description}")
    
    # Store raw IQ data for DQN input
    merged_dedup.loc[:, "iq_raw"] = iq_raw_list
    iq_raw_serialized = [json.dumps(np.asarray(arr, dtype=np.float32).tolist()) for arr in iq_raw_list]
    
    # Combine all features
    output_df = pd.DataFrame({
        "seq": merged_dedup["seq"].values,
        "mcs_index": merged_dedup["mcs_index"].values,
        "delivered": merged_dedup["delivered"].values,
        "service_ms": merged_dedup["service_ms"].values,
        "reward": merged_dedup["reward"].values,
        "rssi": merged_dedup["rssi"].values,
        "snr": merged_dedup["snr"].values,
        "fft_gain": merged_dedup["fft_gain"].values,
        "agc_gain": merged_dedup["agc_gain"].values,
        "channel": merged_dedup["channel"].values,
        "sig_len": merged_dedup["sig_len"].values,
        "iq_mean": iq_df["iq_mean"].values,
        "iq_std": iq_df["iq_std"].values,
        "iq_p10": iq_df["iq_p10"].values,
        "iq_p50": iq_df["iq_p50"].values,
        "iq_p90": iq_df["iq_p90"].values,
        "iq_raw": iq_raw_serialized,
    })

    if csi_feature_contract == LINK_V7C_CONTRACT_ID:
        output_df["iq_active_amplitudes"] = [
            json.dumps(np.asarray(value, dtype=np.float32).tolist())
            for value in iq_df["iq_active_amplitudes"].values
        ]
        output_df["iq_phase_diff_real"] = [
            json.dumps(np.asarray(value, dtype=np.float32).tolist())
            for value in iq_df["iq_phase_diff_real"].values
        ]
        output_df["iq_phase_diff_imag"] = [
            json.dumps(np.asarray(value, dtype=np.float32).tolist())
            for value in iq_df["iq_phase_diff_imag"].values
        ]
        output_df["iq_phase_valid_fraction"] = iq_df["iq_phase_valid_fraction"].values
        output_df["iq_phase_coherence"] = iq_df["iq_phase_coherence"].values
        # These fields describe the CSI state, not necessarily the target ACK.
        # The prefix keeps that distinction explicit in previous_csi datasets.
        for column in CSI_PROVENANCE_COLUMNS:
            if column in merged_dedup.columns:
                output_df[f"csi_{column}"] = merged_dedup[column].values

    if state_alignment == "previous_csi":
        output_df.insert(1, "state_seq", merged_dedup["state_seq"].values)
        output_df.insert(2, "state_mcs_index", merged_dedup["state_mcs_index"].values)
        output_df.insert(3, "state_age_packets", merged_dedup["state_age_packets"].values)
        output_df.insert(4, "state_packet_gap", merged_dedup["state_packet_gap"].values)
        output_df.insert(5, "state_missing_packets", merged_dedup["state_missing_packets"].values)
        output_df.insert(6, "state_is_stale", merged_dedup["state_is_stale"].values)
        output_df.insert(7, "target_has_csi", merged_dedup["target_has_csi"].values)

        output_df = add_synthetic_stale_rows(
            output_df,
            stale_ages,
            stale_augment_source,
        )
    
    # Add metadata columns if provided
    if metadata:
        for key, value in metadata.items():
            output_df[f"meta_{key}"] = metadata_csv_value(value)
    
    # Save to CSV
    print(f"  Writing output to: {output_csv}")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    if csi_feature_contract == LINK_V7C_CONTRACT_ID:
        write_link_v7c_dataset_sidecar(
            output_csv=output_csv,
            output_df=output_df,
            csi_csv=csi_csv,
            csi_df=csi_df,
            source_csi_df=source_csi_df,
            capture_filter_report=capture_filter_report,
            ack_csv=ack_csv,
            ack_df=ack_df,
            metadata_json=metadata_json,
            metadata=metadata,
            dedup_strategy=dedup_strategy,
            reward_manifest=reward_manifest,
            state_alignment=state_alignment,
            stale_ages=stale_ages,
            stale_augment_source=stale_augment_source,
            source_hashes=source_hashes,
        )
    
    print(f"  ✅ Dataset complete: {len(output_df)} rows, {len(output_df.columns)} columns")
    print(f"  Feature columns: seq, mcs_index, delivered, service_ms, reward")
    print(f"                   rssi, snr, fft_gain, agc_gain, channel, sig_len")
    print(f"                   iq_mean, iq_std, iq_p10, iq_p50, iq_p90, iq_raw")
    if csi_feature_contract == LINK_V7C_CONTRACT_ID:
        print(
            "                   iq_active_amplitudes, iq_phase_diff_real, "
            "iq_phase_diff_imag, iq_phase_valid_fraction, "
            "iq_phase_coherence"
        )
    print(f"  Reward mode: {reward_mode}")
    print(f"  State alignment: {state_alignment}")
    print(f"  CSI feature contract: {csi_feature_contract}")
    if state_alignment == "previous_csi":
        print(
            "  Causal state age (packets): "
            f"median={output_df['state_age_packets'].median():.1f}, "
            f"p95={output_df['state_age_packets'].quantile(0.95):.1f}, "
            f"max={output_df['state_age_packets'].max()}"
        )
        print(
            "  Causal missing-packet estimate: "
            f"median={output_df['state_missing_packets'].median():.1f}, "
            f"p95={output_df['state_missing_packets'].quantile(0.95):.1f}, "
            f"max={output_df['state_missing_packets'].max()}"
        )
        print("  Source-packet MCS: state_mcs_index (one-hot encoded during training)")
        print(f"  Retained losses: {int((output_df['delivered'] == 0).sum())}")
        if "synthetic_stale" in output_df.columns:
            print(f"  Synthetic stale rows: {int(output_df['synthetic_stale'].sum())}")
    if reward_mode == "delay":
        print(f"  Reward range: [{output_df['reward'].min():.3f}, {output_df['reward'].max():.3f}] (negative milliseconds)")
    elif reward_mode == "log_delay":
        print(f"  Reward range: [{output_df['reward'].min():.3f}, {output_df['reward'].max():.3f}] (-log milliseconds)")
    elif reward_mode == "robust_delay":
        print(f"  Reward range: [{output_df['reward'].min():.3f}, {output_df['reward'].max():.3f}] (robust log-scaled delay)")
    elif reward_mode == "utility":
        print(f"  Reward range: [{output_df['reward'].min():.3f}, {output_df['reward'].max():.3f}] (bounded utility)")
        print(f"  Mean reward by delivery: {output_df.groupby('delivered')['reward'].mean().to_dict()}")
    elif reward_mode in {"pdr", "fast_latency"}:
        pdr_stats = output_df['reward'].value_counts().to_dict()
        print(f"  Reward distribution: {pdr_stats}")
    print(f"  Service time range: [{output_df['service_ms'].min():.3f}, {output_df['service_ms'].max():.3f}] ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build DQN dataset from CSI and ACK logs"
    )
    parser.add_argument(
        "--csi-csv", default="csi_data.csv",
        help="Path to CSI CSV file"
    )
    parser.add_argument(
        "--ack-csv", default="ack_data.csv",
        help="Path to ACK CSV file"
    )
    parser.add_argument(
        "--output", default="rl_dqn_dataset.csv",
        help="Path to output DQN dataset"
    )
    parser.add_argument(
        "--dedup", choices=["keep_first", "keep_latest", "aggregate"],
        default="keep_first",
        help="Deduplication strategy for duplicate sequence IDs"
    )
    parser.add_argument(
        "--reward-mode", choices=["delay", "pdr", "fast_latency", "log_delay", "robust_delay", "utility"],
        default="delay",
        help=(
            "Reward function: 'delay' = -service_ms, 'pdr' = delivered 0/1, "
            "'fast_latency' = +1 for very low service time, "
            "'log_delay' = -log(service_ms), "
            "'robust_delay' = -log1p(min(service_ms, q)/scale), "
            "'utility' = bounded log-goodput with loss/tail penalties"
        )
    )
    parser.add_argument(
        "--fast-target-ms",
        type=float,
        default=0.25,
        help="Full-reward service-time threshold for --reward-mode fast_latency"
    )
    parser.add_argument(
        "--fast-ok-ms",
        type=float,
        default=0.50,
        help="Partial-reward service-time threshold for --reward-mode fast_latency"
    )
    parser.add_argument(
        "--robust-cap-quantile",
        type=float,
        default=0.99,
        help="Cap service_ms at this quantile for --reward-mode robust_delay"
    )
    parser.add_argument(
        "--robust-scale-ms",
        type=float,
        default=0.25,
        help="Scale term in -log1p(service_ms/scale) for --reward-mode robust_delay"
    )
    parser.add_argument(
        "--robust-loss-penalty-ms",
        type=float,
        default=5.0,
        help="Extra cost added past the cap for undelivered packets in robust_delay mode"
    )
    parser.add_argument(
        "--utility-payload-bytes",
        type=int,
        default=128,
        help="Payload bytes used to compute goodput for --reward-mode utility"
    )
    parser.add_argument(
        "--utility-goodput-quantile",
        type=float,
        default=0.95,
        help="Delivered log-goodput quantile mapped to reward +1 in utility mode"
    )
    parser.add_argument(
        "--utility-loss-reward",
        type=float,
        default=-1.0,
        help="Reward assigned to undelivered packets in utility mode before clipping"
    )
    parser.add_argument(
        "--utility-tail-target-ms",
        type=float,
        default=0.0,
        help="Optional tail-delay target for utility mode; 0 disables the tail penalty"
    )
    parser.add_argument(
        "--utility-tail-weight",
        type=float,
        default=0.0,
        help="Optional tail-delay penalty weight for utility mode"
    )
    parser.add_argument(
        "--metadata-json", default=None,
        help="Path to JSON file with scenario metadata (distance_m, obstacle_type, movement_type, etc.)"
    )
    parser.add_argument(
        "--csi-feature-contract",
        choices=CSI_FEATURE_CONTRACTS,
        default=LEGACY_CSI_FEATURE_CONTRACT,
        help=(
            "CSI transform to write. The default preserves legacy amplitude datasets; "
            "link_v7c_ht20_v1 enables the strict 166-value full-CSI contract and sidecar."
        ),
    )
    parser.add_argument(
        "--state-alignment",
        choices=["same_packet", "previous_csi"],
        default="same_packet",
        help=(
            "State/action alignment: same_packet preserves legacy datasets; "
            "previous_csi uses the latest strictly earlier CSI and retains ACK losses"
        ),
    )
    parser.add_argument(
        "--stale-augment-ages",
        default=None,
        help=(
            "Comma-separated ages for synthetic no-new-CSI rows, e.g. '2,4,8,16'. "
            "Requires --state-alignment previous_csi."
        ),
    )
    parser.add_argument(
        "--stale-augment-source",
        choices=["loss_only", "loss_or_missing", "all"],
        default="loss_only",
        help="Rows to duplicate for synthetic stale-state augmentation",
    )
    
    args = parser.parse_args()
    
    build_dataset(
        args.csi_csv,
        args.ack_csv,
        args.output,
        dedup_strategy=args.dedup,
        reward_mode=args.reward_mode,
        metadata_json=args.metadata_json,
        fast_target_ms=args.fast_target_ms,
        fast_ok_ms=args.fast_ok_ms,
        robust_cap_quantile=args.robust_cap_quantile,
        robust_scale_ms=args.robust_scale_ms,
        robust_loss_penalty_ms=args.robust_loss_penalty_ms,
        utility_payload_bytes=args.utility_payload_bytes,
        utility_goodput_quantile=args.utility_goodput_quantile,
        utility_loss_reward=args.utility_loss_reward,
        utility_tail_target_ms=args.utility_tail_target_ms,
        utility_tail_weight=args.utility_tail_weight,
        state_alignment=args.state_alignment,
        stale_augment_ages=args.stale_augment_ages,
        stale_augment_source=args.stale_augment_source,
        csi_feature_contract=args.csi_feature_contract,
    )
