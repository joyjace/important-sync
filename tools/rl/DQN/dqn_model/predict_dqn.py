#!/usr/bin/env python3
"""Batch DQN/Q-network predictions with auditable v3.3 row identity."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_dqn import (  # noqa: E402
    QNetwork,
    build_state_vector,
    feature_contract_metadata,
    state_feature_names,
    validate_dataset_feature_contract,
)


DEFAULT_PASSTHROUGH_COLUMNS = (
    "v33_row_id",
    "source_file",
    "source_scenario",
    "meta_angle_deg",
    "meta_capture_group",
    "meta_repeat_idx",
)


def mcs_to_symbol(mcs_idx: int) -> str:
    """Convert an MCS index to its ESP-IDF rate symbol."""

    symbols = [f"WIFI_PHY_RATE_MCS{index}_LGI" for index in range(8)]
    return symbols[int(mcs_idx)] if 0 <= int(mcs_idx) < 8 else "UNKNOWN"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_dataframe(path: Path, frame: pd.DataFrame) -> None:
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
    """Fingerprint the ordered population used by the policy evaluator."""

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


class DqnPredictionModel:
    """Strict, inference-only view of a serialized Q-network."""

    def __init__(self, checkpoint: dict[str, object], device: str = "cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        self.device = device
        self.checkpoint_schema = checkpoint.get("checkpoint_schema")
        self.state_dim = int(checkpoint["state_dim"])
        self.action_dim = int(checkpoint["action_dim"])
        self.hidden_dim = int(checkpoint.get("hidden_dim", 128))
        self.hidden_layers = int(checkpoint.get("hidden_layers", 2))
        self.dropout = float(checkpoint.get("dropout", 0.0))
        self.layer_norm = bool(checkpoint.get("layer_norm", False))
        self.state_schema = str(checkpoint.get("state_schema", "legacy_v1"))
        self.state_context_feature = str(
            checkpoint.get("state_context_feature", "sig_len")
        )
        self.include_state_mcs = bool(
            checkpoint.get(
                "include_state_mcs", self.state_dim in {136, 140, 143, 262}
            )
        )
        self.objective_contract = str(
            checkpoint.get("objective_contract", "discounted_return")
        )

        if self.action_dim != 8:
            raise ValueError(f"Expected eight MCS actions, got {self.action_dim}")
        expected_names = state_feature_names(
            self.state_schema, self.include_state_mcs
        )
        if self.state_dim != len(expected_names):
            raise ValueError(
                f"Checkpoint state_dim={self.state_dim} does not match "
                f"{self.state_schema} ({len(expected_names)})"
            )
        stored_names = checkpoint.get("state_feature_names")
        if stored_names is not None and list(stored_names) != expected_names:
            raise ValueError("Checkpoint state_feature_names are out of order or invalid")
        for key, expected in feature_contract_metadata(self.state_schema).items():
            if checkpoint.get(key) != expected:
                raise ValueError(
                    f"Checkpoint {key}={checkpoint.get(key)!r} does not match {expected!r}"
                )

        self.state_mean = np.asarray(checkpoint.get("state_mean"), dtype=np.float32)
        self.state_std = np.asarray(checkpoint.get("state_std"), dtype=np.float32)
        if self.state_mean.shape != (self.state_dim,) or self.state_std.shape != (
            self.state_dim,
        ):
            raise ValueError("Checkpoint must contain complete state normalization")
        if not np.all(np.isfinite(self.state_mean)) or not np.all(
            np.isfinite(self.state_std)
        ):
            raise ValueError("Checkpoint normalization contains non-finite values")
        if np.any(self.state_std <= 0.0):
            raise ValueError("Checkpoint state standard deviations must be positive")

        self.network = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        ).to(device)
        self.network.load_state_dict(checkpoint["q_net_state"])
        self.network.eval()
        for name, parameter in self.network.state_dict().items():
            if not torch.isfinite(parameter).all():
                raise ValueError(f"Checkpoint tensor {name} contains non-finite values")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DqnPredictionModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("DQN checkpoint must be a mapping")
        return cls(checkpoint, device=device)

    def score_states(self, states: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("Prediction batch size must be positive")
        states = np.asarray(states, dtype=np.float32)
        if states.shape != (len(states), self.state_dim):
            raise ValueError(
                f"Expected an Nx{self.state_dim} state matrix, got {states.shape}"
            )
        if not np.all(np.isfinite(states)):
            raise ValueError("Prediction state matrix contains non-finite values")

        scores = np.empty((len(states), self.action_dim), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(states), batch_size):
                stop = min(start + batch_size, len(states))
                normalized = (
                    (states[start:stop] - self.state_mean) / self.state_std
                ).astype(np.float32)
                batch = torch.from_numpy(normalized).to(self.device)
                scores[start:stop] = self.network(batch).cpu().numpy()
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("DQN inference produced non-finite Q values")
        return scores


def filter_dataset(
    frame: pd.DataFrame,
    *,
    delivered_only: bool,
    observed_only: bool,
    fresh_state_only: bool,
) -> pd.DataFrame:
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
    if fresh_state_only:
        if "state_age_packets" not in work.columns:
            raise ValueError("--fresh-state-only requires state_age_packets")
        ages = pd.to_numeric(work["state_age_packets"], errors="coerce")
        work = work[ages == 1]
    work = work.reset_index(drop=True)
    if work.empty:
        raise ValueError("No rows remain after prediction filters")
    return work


def build_states(frame: pd.DataFrame, model: DqnPredictionModel) -> np.ndarray:
    dataset_is_causal = "state_age_packets" in frame.columns
    model_is_causal = model.state_context_feature == "state_age_packets"
    if dataset_is_causal != model_is_causal:
        raise ValueError("Model and dataset causal-state contracts do not match")
    if model.include_state_mcs and "state_mcs_index" not in frame.columns:
        raise ValueError("Checkpoint requires state_mcs_index, but the dataset omits it")
    if "iq_raw" not in frame.columns and model.state_schema != "link_v7c":
        raise ValueError("Prediction dataset is missing iq_raw")

    required_numeric = {
        "rssi",
        "snr",
        "fft_gain",
        "agc_gain",
        "channel",
        "sig_len",
    }
    if model.state_context_feature == "state_age_packets":
        required_numeric.add("state_age_packets")
    if model.state_schema != "legacy_v1":
        required_numeric.update(
            {
                "state_age_packets",
                "state_packet_gap",
                "state_missing_packets",
                "state_is_stale",
            }
        )
    if model.state_schema != "link_v7c":
        required_numeric.update(
            {"iq_mean", "iq_std", "iq_p10", "iq_p50", "iq_p90"}
        )
    if model.state_schema in {"link_v3", "link_v4"}:
        required_numeric.update(
            {
                "state_prev_delivered",
                "state_consecutive_losses",
                "state_recent_loss_rate_8",
            }
        )
    if model.state_schema == "link_v6":
        required_numeric.update(
            {"phase_mean", "phase_std", "phase_p10", "phase_p50", "phase_p90"}
        )
    if model.state_schema == "link_v7c":
        required_numeric.update(
            {"iq_phase_valid_fraction", "iq_phase_coherence"}
        )
    if model.include_state_mcs:
        required_numeric.add("state_mcs_index")

    missing = sorted(required_numeric - set(frame.columns))
    if missing:
        raise ValueError(
            "Prediction dataset is missing model input columns: " + ", ".join(missing)
        )
    for column in sorted(required_numeric):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Prediction input column {column} is non-numeric or non-finite")

    if model.state_schema != "link_v7c":
        for index, value in enumerate(frame["iq_raw"]):
            try:
                if isinstance(value, str):
                    raw = value.strip("[]").replace("\n", " ").replace(",", " ")
                    parsed = np.fromstring(raw, sep=" ", dtype=np.float32)
                else:
                    parsed = np.asarray(value, dtype=np.float32).reshape(-1)
            except Exception as exc:
                raise ValueError(f"Prediction row {index} has invalid iq_raw") from exc
            if parsed.size < 1 or parsed.size > 117 or not np.all(np.isfinite(parsed)):
                raise ValueError(
                    f"Prediction row {index} has invalid iq_raw length or values"
                )

    states = np.empty((len(frame), model.state_dim), dtype=np.float32)
    for index, row in enumerate(frame.itertuples(index=False)):
        state = build_state_vector(
            row,
            model.state_context_feature,
            model.include_state_mcs,
            model.state_schema,
        )
        if state.shape != (model.state_dim,):
            raise ValueError(
                f"Row {index} produced state shape {state.shape}; "
                f"expected {(model.state_dim,)}"
            )
        states[index] = state
    if not np.all(np.isfinite(states)):
        raise ValueError("Prediction state matrix contains non-finite values")
    return states


def _numeric_series(
    frame: pd.DataFrame, column: str, default: float
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.full(len(frame), default), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def predict_mcs(
    model_path: str | Path,
    dataset_csv: str | Path,
    output_csv: str | Path,
    device: str = "cpu",
    batch_size: int = 4096,
    delivered_only: bool = False,
    observed_only: bool = False,
    fresh_state_only: bool = False,
    metadata_out: str | Path | None = None,
) -> dict[str, object]:
    model_path = Path(model_path).resolve()
    dataset_csv = Path(dataset_csv).resolve()
    output_csv = Path(output_csv).resolve()

    print("[DQN/Q-network prediction]")
    print(f"  Model: {model_path}")
    model = DqnPredictionModel.load(model_path, device=device)
    validate_dataset_feature_contract(dataset_csv, model.state_schema)
    source_rows = pd.read_csv(dataset_csv)
    frame = filter_dataset(
        source_rows,
        delivered_only=delivered_only,
        observed_only=observed_only,
        fresh_state_only=fresh_state_only,
    )
    print(f"  Rows: {len(frame)} / {len(source_rows)}")

    states = build_states(frame, model)
    scores = model.score_states(states, batch_size=batch_size)
    best_actions = np.argmax(scores, axis=1).astype(np.int64)
    best_values = scores[np.arange(len(scores)), best_actions]
    ranking = np.argsort(scores, axis=1)[:, ::-1]

    predictions = pd.DataFrame(
        {
            "dataset_row_index": frame["_source_row_index"].to_numpy(dtype=np.int64),
            "seq": _numeric_series(frame, "seq", -1).to_numpy(),
            "actual_mcs": _numeric_series(frame, "mcs_index", -1).to_numpy(),
            "predicted_mcs": best_actions,
            "predicted_q_value": best_values,
            "predicted_value": best_values,
            "predicted_target_column": (
                "reward"
                if model.objective_contract.startswith("immediate_logged_packet_utility")
                else "discounted_return"
            ),
            "predicted_objective": "maximize",
            "actual_reward": _numeric_series(frame, "reward", np.nan).to_numpy(),
            "service_ms": _numeric_series(frame, "service_ms", np.nan).to_numpy(),
            "rssi": _numeric_series(frame, "rssi", np.nan).to_numpy(),
            "snr": _numeric_series(frame, "snr", np.nan).to_numpy(),
            "delivered": _numeric_series(frame, "delivered", -1).to_numpy(),
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
    for action in range(model.action_dim):
        predictions[f"q_mcs{action}"] = scores[:, action]
    predictions["top3_mcs"] = [
        [int(action) for action in values]
        for values in ranking[:, :3]
    ]
    predictions["top3_q_values"] = [
        [float(scores[index, action]) for action in values]
        for index, values in enumerate(ranking[:, :3])
    ]
    for column in DEFAULT_PASSTHROUGH_COLUMNS:
        if column in frame.columns:
            predictions[column] = frame[column].to_numpy()

    identity_sha = ordered_identity_sha256(predictions)
    atomic_write_dataframe(output_csv, predictions)
    metadata = {
        "schema": "dqn_prediction/v1",
        "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "dataset": {
            "path": str(dataset_csv),
            "sha256": sha256_file(dataset_csv),
            "source_rows": int(len(source_rows)),
            "evaluated_rows": int(len(frame)),
        },
        "output": {"path": str(output_csv), "sha256": sha256_file(output_csv)},
        "checkpoint_schema": model.checkpoint_schema,
        "state_schema": model.state_schema,
        "objective_contract": model.objective_contract,
        "ordered_identity_sha256": identity_sha,
        "passthrough_columns": [
            column for column in DEFAULT_PASSTHROUGH_COLUMNS if column in frame.columns
        ],
        "filters": {
            "delivered_only": bool(delivered_only),
            "observed_only": bool(observed_only),
            "fresh_state_only": bool(fresh_state_only),
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
    rendered = ", ".join(
        f"MCS{int(action)}={int(count)}" for action, count in counts.items()
    )
    print(f"  Predictions: {output_csv}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Selection counts: {rendered}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="DQN checkpoint")
    parser.add_argument("--dataset", required=True, help="Prepared dataset CSV")
    parser.add_argument("--output", required=True, help="Prediction CSV")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--delivered-only", action="store_true")
    parser.add_argument("--observed-only", action="store_true")
    parser.add_argument("--fresh-state-only", action="store_true")
    parser.add_argument("--metadata-out", default=None)
    args = parser.parse_args()
    predict_mcs(
        args.model,
        args.dataset,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
        delivered_only=args.delivered_only,
        observed_only=args.observed_only,
        fresh_state_only=args.fresh_state_only,
        metadata_out=args.metadata_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
