#!/usr/bin/env python3
"""Score all MCS actions with an auditable two-head bandit checkpoint.

The prediction CSV is compatible with ``shared/evaluate_policy_delay.py`` and
retains the ordered identity of every source row used in evaluation.  A fair
exact-fresh checkpoint automatically applies its training population contract;
checkpoints without population metadata remain usable with an all-row default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DQN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DQN_ROOT / "dqn_model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dqn import (  # noqa: E402
    build_state_vector,
    feature_contract_metadata,
    validate_dataset_feature_contract,
)
from train_reward_model import normalize, parse_iq_raw  # noqa: E402
from train_bandit_model import (  # noqa: E402
    ACTION_DIM,
    FAIR_CHECKPOINT_SCHEMA,
    MODEL_TYPE,
    BanditOutcomeNetwork,
    score_all_actions,
)
DEFAULT_PASSTHROUGH_COLUMNS = (
    "v33_row_id",
    "source_file",
    "source_scenario",
    "meta_angle_deg",
    "meta_capture_group",
    "meta_repeat_idx",
)
MIN_PREDICTED_DELAY_MS = 1e-3
MAX_PREDICTED_DELAY_MS = 1e4


def firmware_float32_policy_outputs(
    p_deliver: np.ndarray,
    log_delay_mu: np.ndarray,
    utility_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the generated C utility arithmetic for deployable checkpoints."""

    p = np.asarray(p_deliver, dtype=np.float32)
    mu = np.asarray(log_delay_mu, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        delay_ms = np.exp(mu, dtype=np.float32)
    delay_ms = np.clip(
        delay_ms,
        np.float32(MIN_PREDICTED_DELAY_MS),
        np.float32(MAX_PREDICTED_DELAY_MS),
    ).astype(np.float32)
    objective = str(utility_params["objective"])
    payload_bits = np.float32(float(utility_params["payload_bytes"]) * 8.0)
    if objective == "utility":
        scale = np.float32(utility_params["utility_scale"])
        success = (
            np.float32(2.0)
            * np.clip(
                np.log1p(payload_bits / delay_ms, dtype=np.float32) / scale,
                np.float32(0.0),
                np.float32(1.0),
            )
            - np.float32(1.0)
        ).astype(np.float32)
        tail_target = np.float32(utility_params["tail_target_ms"])
        tail_weight = np.float32(utility_params["tail_weight"])
        if tail_target > 0.0 and tail_weight > 0.0:
            excess = np.maximum(
                (delay_ms - tail_target) / tail_target, np.float32(0.0)
            )
            success = (
                success
                - tail_weight
                * np.clip(excess, np.float32(0.0), np.float32(1.0))
            ).astype(np.float32)
        success = np.clip(
            success, np.float32(-1.0), np.float32(1.0)
        ).astype(np.float32)
        loss_reward = np.float32(utility_params["loss_reward"])
        scores = (
            p * success + (np.float32(1.0) - p) * loss_reward
        ).astype(np.float32)
    elif objective == "goodput":
        scores = (p * payload_bits / delay_ms).astype(np.float32)
    elif objective == "delay":
        scores = (
            -delay_ms / np.maximum(p, np.float32(1e-3))
        ).astype(np.float32)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return delay_ms, scores


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    """Publish a complete CSV without exposing a partially written file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ordered_identity_sha256(frame: pd.DataFrame) -> str:
    """Fingerprint the ordered population consumed by the policy evaluator."""

    columns = ["dataset_row_index"] + [
        column for column in DEFAULT_PASSTHROUGH_COLUMNS if column in frame.columns
    ]
    digest = hashlib.sha256()
    digest.update(json.dumps(columns, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")
    for values in frame[columns].itertuples(index=False, name=None):
        digest.update(
            json.dumps(
                [None if pd.isna(value) else value for value in values],
                default=str,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def load_bandit_checkpoint(path: Path, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Bandit checkpoint must be a mapping")
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"Checkpoint is not a {MODEL_TYPE} model: {checkpoint.get('model_type')}"
        )
    # Checkpoints trained before the schema rename stored the compact
    # 57-amplitude layout as "link_v3".  Keep that historical artifact usable.
    if checkpoint.get("state_schema") == "link_v3":
        checkpoint["state_schema"] = "link_v3c"
    for key, expected in feature_contract_metadata(checkpoint.get("state_schema")).items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"Checkpoint {key}={checkpoint.get(key)!r} does not match {expected!r}"
            )

    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint.get("action_dim", ACTION_DIM))
    action_feature_dim = int(checkpoint.get("action_feature_dim", 9))
    if action_dim != ACTION_DIM:
        raise ValueError(f"Bandit checkpoint must contain {ACTION_DIM} actions")
    if action_feature_dim != ACTION_DIM + 1:
        raise ValueError("Bandit checkpoint has an unsupported action feature layout")

    state_mean = np.asarray(checkpoint.get("state_mean"), dtype=np.float32)
    state_std = np.asarray(checkpoint.get("state_std"), dtype=np.float32)
    if state_mean.shape != (state_dim,) or state_std.shape != (state_dim,):
        raise ValueError("Checkpoint must contain complete state normalization")
    if not np.all(np.isfinite(state_mean)) or not np.all(np.isfinite(state_std)):
        raise ValueError("Checkpoint normalization contains non-finite values")
    if np.any(state_std <= 0.0):
        raise ValueError("Checkpoint state standard deviations must be positive")

    model = BanditOutcomeNetwork(
        state_dim=state_dim,
        action_feature_dim=action_feature_dim,
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for name, tensor in model.state_dict().items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Checkpoint tensor {name} contains non-finite values")
    return model, checkpoint


def checkpoint_filter_contract(checkpoint: dict) -> tuple[bool, bool]:
    """Return mandatory (observed-only, exact-fresh-only) inference filters."""

    exact_fresh_only = bool(
        checkpoint.get("training_exact_fresh_only", False)
        or checkpoint.get("exact_fresh_only", False)
        or checkpoint.get("checkpoint_schema") == FAIR_CHECKPOINT_SCHEMA
    )
    observed_only = bool(checkpoint.get("observed_only", False) or exact_fresh_only)
    return observed_only, exact_fresh_only


def filter_dataset(
    frame: pd.DataFrame,
    *,
    delivered_only: bool,
    observed_only: bool,
    fresh_state_only: bool,
    exact_fresh_only: bool,
) -> pd.DataFrame:
    """Filter without reordering and retain each row's source-file position."""

    work = frame.copy()
    work["_source_row_index"] = np.arange(len(work), dtype=np.int64)

    if delivered_only:
        if "delivered" not in work.columns:
            raise ValueError("--delivered-only requires a delivered column")
        work = work[pd.to_numeric(work["delivered"], errors="coerce") == 1]

    if observed_only:
        synthetic = pd.to_numeric(
            work.get("synthetic_stale", pd.Series(0, index=work.index)),
            errors="coerce",
        ).fillna(0)
        augmented = pd.to_numeric(
            work.get("model_augmented", pd.Series(0, index=work.index)),
            errors="coerce",
        ).fillna(0)
        work = work[(synthetic == 0) & (augmented == 0)]

    if exact_fresh_only:
        exact_values = {
            "state_age_packets": 1,
            "state_packet_gap": 1,
            "state_missing_packets": 0,
            "state_is_stale": 0,
        }
        missing = sorted(set(exact_values) - set(work.columns))
        if missing:
            raise ValueError(
                "Exact-fresh checkpoint requires dataset columns: "
                + ", ".join(missing)
            )
        keep = np.ones(len(work), dtype=bool)
        for column, expected in exact_values.items():
            values = pd.to_numeric(work[column], errors="coerce").to_numpy()
            keep &= values == expected
        work = work[keep]
    elif fresh_state_only:
        if "state_age_packets" not in work.columns:
            raise ValueError("--fresh-state-only requires state_age_packets")
        ages = pd.to_numeric(work["state_age_packets"], errors="coerce")
        work = work[ages == 1]

    work = work.reset_index(drop=True)
    if work.empty:
        raise ValueError("No rows remain after prediction filters")
    return work


def build_states(frame: pd.DataFrame, checkpoint: dict) -> np.ndarray:
    dataset_is_causal = "state_age_packets" in frame.columns
    model_is_causal = checkpoint["state_context_feature"] == "state_age_packets"
    if dataset_is_causal != model_is_causal:
        raise ValueError("Model and dataset causal-state contracts do not match")
    if checkpoint["include_state_mcs"] and "state_mcs_index" not in frame.columns:
        raise ValueError("Checkpoint requires state_mcs_index, but the dataset omits it")
    if "iq_raw" not in frame.columns:
        raise ValueError("Prediction dataset is missing iq_raw")

    work = frame.copy()
    work["iq_raw_parsed"] = work["iq_raw"].apply(parse_iq_raw)
    states = np.asarray(
        [
            build_state_vector(
                row,
                checkpoint["state_context_feature"],
                checkpoint["include_state_mcs"],
                checkpoint["state_schema"],
            )
            for row in work.itertuples(index=False)
        ],
        dtype=np.float32,
    )
    expected_shape = (len(work), int(checkpoint["state_dim"]))
    if states.shape != expected_shape:
        raise ValueError(
            f"Dataset produced state shape {states.shape}; checkpoint expects "
            f"{expected_shape}"
        )
    if not np.all(np.isfinite(states)):
        raise ValueError("Prediction state matrix contains non-finite values")
    normalized = normalize(
        states,
        np.asarray(checkpoint["state_mean"], dtype=np.float32),
        np.asarray(checkpoint["state_std"], dtype=np.float32),
    ).astype(np.float32)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Normalized prediction state matrix contains non-finite values")
    return normalized


def _numeric_series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.full(len(frame), default), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _json_utility_params(values: dict) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values.items():
        if isinstance(value, (np.integer, int)):
            result[str(key)] = int(value)
        elif isinstance(value, (np.floating, float)):
            converted = float(value)
            if not np.isfinite(converted):
                raise ValueError(f"Bandit utility parameter {key} is non-finite")
            result[str(key)] = converted
        else:
            result[str(key)] = value
    return result


def predict_bandit_model(
    model_path: str | Path,
    dataset_csv: str | Path,
    output_csv: str | Path,
    *,
    objective: str = "checkpoint",
    batch_size: int = 4096,
    device: str = "cpu",
    delivered_only: bool = False,
    observed_only: bool = False,
    fresh_state_only: bool = False,
    metadata_out: str | Path | None = None,
) -> dict[str, object]:
    model_path = Path(model_path).resolve()
    dataset_csv = Path(dataset_csv).resolve()
    output_csv = Path(output_csv).resolve()
    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive")

    print("[Bandit Model Prediction]")
    print(f"  Model: {model_path}")
    print(f"  Dataset: {dataset_csv}")
    model, checkpoint = load_bandit_checkpoint(model_path, device)

    utility_params = dict(checkpoint["utility_params"])
    if objective != "checkpoint":
        utility_params["objective"] = objective
    utility_params = _json_utility_params(utility_params)
    print(f"  Objective: {utility_params['objective']}")

    checkpoint_observed_only, checkpoint_exact_fresh_only = (
        checkpoint_filter_contract(checkpoint)
    )
    effective_observed_only = observed_only or checkpoint_observed_only
    effective_exact_fresh_only = checkpoint_exact_fresh_only
    effective_fresh_state_only = fresh_state_only or effective_exact_fresh_only

    validate_dataset_feature_contract(dataset_csv, checkpoint["state_schema"])
    source_rows = pd.read_csv(dataset_csv)
    frame = filter_dataset(
        source_rows,
        delivered_only=delivered_only,
        observed_only=effective_observed_only,
        fresh_state_only=effective_fresh_state_only,
        exact_fresh_only=effective_exact_fresh_only,
    )
    print(f"  Rows: {len(frame)} / {len(source_rows)}")

    states = build_states(frame, checkpoint)
    p_all, mu_all = score_all_actions(
        model,
        states,
        device,
        batch_size=batch_size,
    )
    expected_shape = (len(frame), ACTION_DIM)
    if p_all.shape != expected_shape or mu_all.shape != expected_shape:
        raise ValueError(
            f"Bandit inference returned {p_all.shape} and {mu_all.shape}; "
            f"expected {expected_shape}"
        )
    if not np.all(np.isfinite(p_all)) or not np.all(np.isfinite(mu_all)):
        raise FloatingPointError("Bandit inference produced non-finite head outputs")
    if np.any((p_all < 0.0) | (p_all > 1.0)):
        raise FloatingPointError("Bandit inference produced invalid probabilities")

    # Use the generated firmware's float32 arithmetic and clamp order so
    # qualification selects the same action whenever network outputs match.
    delay_ms_all, utilities = firmware_float32_policy_outputs(
        p_all,
        mu_all,
        utility_params,
    )
    if not np.all(np.isfinite(utilities)) or not np.all(np.isfinite(delay_ms_all)):
        raise FloatingPointError("Bandit policy derivation produced non-finite outputs")

    predicted = np.argmax(utilities, axis=1).astype(np.int64)
    predicted_values = utilities[np.arange(len(frame)), predicted]
    predictions = pd.DataFrame(
        {
            "dataset_row_index": frame["_source_row_index"].to_numpy(dtype=np.int64),
            "seq": _numeric_series(frame, "seq", -1).to_numpy(),
            "actual_mcs": _numeric_series(frame, "mcs_index", -1).to_numpy(),
            "predicted_mcs": predicted,
            "predicted_utility": predicted_values,
            "predicted_value": predicted_values,
            "predicted_target_column": "expected_utility",
            "predicted_objective": str(utility_params["objective"]),
            "actual_reward": _numeric_series(frame, "reward", np.nan).to_numpy(),
            "service_ms": _numeric_series(frame, "service_ms", np.nan).to_numpy(),
            "delivered": _numeric_series(frame, "delivered", -1).to_numpy(),
            "rssi": _numeric_series(frame, "rssi", np.nan).to_numpy(),
            "snr": _numeric_series(frame, "snr", np.nan).to_numpy(),
            "state_mcs_index": _numeric_series(
                frame, "state_mcs_index", -1
            ).to_numpy(),
            "state_age_packets": _numeric_series(
                frame, "state_age_packets", 0
            ).to_numpy(),
            "state_missing_packets": _numeric_series(
                frame, "state_missing_packets", 0
            ).to_numpy(),
            "state_is_stale": _numeric_series(
                frame, "state_is_stale", 0
            ).to_numpy(),
        }
    )
    for action in range(ACTION_DIM):
        predictions[f"p_deliver_mcs{action}"] = p_all[:, action]
        predictions[f"pred_delay_ms_mcs{action}"] = delay_ms_all[:, action]
        predictions[f"utility_mcs{action}"] = utilities[:, action]
    for column in DEFAULT_PASSTHROUGH_COLUMNS:
        if column in frame.columns:
            predictions[column] = frame[column].to_numpy()

    generated_columns = [
        "predicted_mcs",
        "predicted_utility",
        "predicted_value",
        *[f"p_deliver_mcs{action}" for action in range(ACTION_DIM)],
        *[f"pred_delay_ms_mcs{action}" for action in range(ACTION_DIM)],
        *[f"utility_mcs{action}" for action in range(ACTION_DIM)],
    ]
    generated_values = predictions[generated_columns].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(generated_values)):
        raise FloatingPointError("Bandit prediction table contains non-finite outputs")

    identity_sha = ordered_identity_sha256(predictions)
    atomic_write_dataframe(output_csv, predictions)
    metadata = {
        "schema": "bandit_prediction/v1",
        "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "dataset": {
            "path": str(dataset_csv),
            "sha256": sha256_file(dataset_csv),
            "source_rows": int(len(source_rows)),
            "evaluated_rows": int(len(frame)),
        },
        "output": {"path": str(output_csv), "sha256": sha256_file(output_csv)},
        "checkpoint_schema": checkpoint.get("checkpoint_schema"),
        "state_schema": checkpoint["state_schema"],
        "objective_contract": checkpoint.get("objective_contract"),
        "policy_objective_request": objective,
        "objective": utility_params["objective"],
        "policy_numeric_contract": "generated_firmware_float32/v1",
        "utility_params": utility_params,
        "ordered_identity_sha256": identity_sha,
        "passthrough_columns": [
            column for column in DEFAULT_PASSTHROUGH_COLUMNS if column in frame.columns
        ],
        "filters": {
            "delivered_only": bool(delivered_only),
            "observed_only": bool(effective_observed_only),
            "fresh_state_only": bool(effective_fresh_state_only),
            "exact_fresh_only": bool(effective_exact_fresh_only),
            "requested_observed_only": bool(observed_only),
            "requested_fresh_state_only": bool(fresh_state_only),
            "checkpoint_observed_only": bool(checkpoint_observed_only),
            "checkpoint_exact_fresh_only": bool(checkpoint_exact_fresh_only),
        },
        "predicted_delay_clamp_ms": {
            "minimum": MIN_PREDICTED_DELAY_MS,
            "maximum": MAX_PREDICTED_DELAY_MS,
        },
        "batch_size": int(batch_size),
    }
    metadata_path = (
        Path(metadata_out).resolve()
        if metadata_out is not None
        else output_csv.with_suffix(output_csv.suffix + ".prediction.json")
    )
    atomic_write_json(metadata_path, metadata)

    counts = predictions["predicted_mcs"].value_counts().sort_index()
    print("  Recommended MCS distribution:")
    for mcs_idx, count in counts.items():
        print(
            f"    MCS{int(mcs_idx)}: {int(count)} "
            f"({count / len(predictions) * 100:.1f}%)"
        )
    print("  Mean per-action predictions:")
    for action in range(ACTION_DIM):
        print(
            f"    MCS{action}: p_deliver={p_all[:, action].mean():.3f} "
            f"delay={delay_ms_all[:, action].mean():.3f}ms "
            f"utility={utilities[:, action].mean():.4f}"
        )
    print(f"  Predictions: {output_csv}")
    print(f"  Metadata: {metadata_path}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--objective",
        choices=["checkpoint", "utility", "goodput", "delay"],
        default="checkpoint",
        help="Policy objective; 'checkpoint' reuses the training-time settings",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--delivered-only", action="store_true")
    parser.add_argument(
        "--observed-only",
        action="store_true",
        help="Exclude synthetic/model-augmented rows (mandatory for fair checkpoints)",
    )
    parser.add_argument(
        "--fresh-state-only",
        action="store_true",
        help="Keep state_age_packets==1 (full exact-fresh contract is automatic)",
    )
    parser.add_argument("--metadata-out", type=Path, default=None)
    args = parser.parse_args()

    predict_bandit_model(
        args.model,
        args.dataset,
        args.output,
        objective=args.objective,
        batch_size=args.batch_size,
        device=args.device,
        delivered_only=args.delivered_only,
        observed_only=args.observed_only,
        fresh_state_only=args.fresh_state_only,
        metadata_out=args.metadata_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
