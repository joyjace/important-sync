#!/usr/bin/env python3
"""
Train a supervised action-conditioned outcome model for MCS selection.

The model learns a scalar outcome from [state, action] pairs:

    outcome_hat = f(state, action)

This supports two practical modes:
    1. Reward modeling: target_column=reward, objective=maximize
    2. Delay modeling:  target_column=service_ms, objective=minimize

For packet-wise MCS selection, direct delay modeling is often easier to debug than
offline DQN because the task is contextual bandit style rather than sequential RL.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DQN_ROOT = Path(__file__).resolve().parents[1]
DQN_MODEL_DIR = DQN_ROOT / "dqn_model"
sys.path.insert(0, str(DQN_MODEL_DIR))

from train_dqn import (
    build_state_vector,
    feature_contract_metadata,
    resolve_state_schema_for_columns,
    state_feature_names,
    validate_dataset_feature_contract,
)
from csi_link_v7c import (
    PHASE_DERIVED_FEATURE_START as LINK_V7C_PHASE_DERIVED_START,
    PHASE_DERIVED_FEATURE_STOP as LINK_V7C_PHASE_DERIVED_STOP,
)


class ActionRewardNetwork(nn.Module):
    """MLP that predicts scalar outcome from [state, action_features]."""

    def __init__(self, state_dim: int = 128, action_feature_dim: int = 9, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_feature_dim = action_feature_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action_features], dim=1)
        return self.net(x).squeeze(1)


def parse_iq_raw(value):
    if isinstance(value, str):
        try:
            raw = value.strip("[]").replace("\n", " ").replace(",", " ")
            arr = np.fromstring(raw, sep=" ")
            if len(arr) == 0:
                return np.zeros(117, dtype=np.float32)
            arr = np.asarray(arr, dtype=np.float32)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return np.zeros(117, dtype=np.float32)
    if isinstance(value, (list, np.ndarray)):
        arr = np.asarray(value, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.zeros(117, dtype=np.float32)


def action_features(actions: np.ndarray, action_dim: int = 8) -> np.ndarray:
    actions = actions.astype(np.int64)
    one_hot = np.zeros((len(actions), action_dim), dtype=np.float32)
    one_hot[np.arange(len(actions)), actions] = 1.0
    action_scalar = (actions.astype(np.float32) / max(action_dim - 1, 1)).reshape(-1, 1)
    return np.concatenate([one_hot, action_scalar], axis=1)


def normalize(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (states - mean) / std


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of an artifact without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def selected_row_indices_sha256(df: pd.DataFrame) -> str:
    """Fingerprint the source rows selected from one immutable input artifact."""

    indices = np.sort(
        df["_trainer_source_row_index"].to_numpy(dtype=np.int64, copy=True)
    )
    digest = hashlib.sha256()
    digest.update(np.asarray([len(indices)], dtype="<i8").tobytes())
    digest.update(indices.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def artifact_provenance(path: str | Path) -> dict[str, object]:
    artifact = Path(path).resolve()
    result: dict[str, object] = {
        "path": str(artifact),
        "sha256": sha256_file(artifact),
        "size_bytes": int(artifact.stat().st_size),
    }
    sidecar = Path(f"{artifact}.feature_contract.json")
    if sidecar.exists():
        result["feature_contract_path"] = str(sidecar)
        result["feature_contract_sha256"] = sha256_file(sidecar)
    return result


def atomic_write_text(path: str | Path, value: str) -> None:
    """Publish a text artifact without leaving a partial final file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def elementwise_regression_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    loss_type: str,
) -> np.ndarray:
    errors = np.asarray(predictions, dtype=np.float64) - np.asarray(
        targets, dtype=np.float64
    )
    if loss_type == "mse":
        return np.square(errors)
    absolute = np.abs(errors)
    return np.where(absolute < 1.0, 0.5 * np.square(errors), absolute - 0.5)


def observed_prediction_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    actions: np.ndarray,
    loss_type: str,
    groups: np.ndarray | None = None,
) -> dict[str, object]:
    """Metrics for the observed/logged action only.

    These metrics measure reward regression, not whether the argmax policy chose
    the optimal counterfactual action.
    """

    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)
    errors = predictions - targets
    losses = elementwise_regression_loss(predictions, targets, loss_type)

    def summarize(mask: np.ndarray) -> dict[str, float | int]:
        selected_errors = errors[mask]
        selected_losses = losses[mask]
        return {
            "rows": int(np.sum(mask)),
            "loss": float(np.mean(selected_losses)),
            "mae": float(np.mean(np.abs(selected_errors))),
            "rmse": float(np.sqrt(np.mean(np.square(selected_errors)))),
            "bias": float(np.mean(selected_errors)),
        }

    result: dict[str, object] = summarize(np.ones(len(targets), dtype=bool))
    result["per_mcs"] = {
        str(action): summarize(actions == action)
        for action in sorted(int(value) for value in np.unique(actions))
    }

    if groups is not None:
        groups = np.asarray(groups, dtype=object)
        per_group = {}
        for group in pd.unique(groups):
            mask = groups == group
            per_group[str(group)] = summarize(mask)
        result["group_count"] = int(len(per_group))
        result["macro_run_loss"] = float(
            np.mean([metrics["loss"] for metrics in per_group.values()])
        )
        result["macro_run_mae"] = float(
            np.mean([metrics["mae"] for metrics in per_group.values()])
        )
        result["per_group"] = per_group
    else:
        result["group_count"] = 0
        result["macro_run_loss"] = None
        result["macro_run_mae"] = None
    return result


def prediction_batches(
    model: ActionRewardNetwork,
    states: np.ndarray,
    action_feature_values: np.ndarray,
    device: str,
    batch_size: int = 4096,
) -> np.ndarray:
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            stop = start + batch_size
            state_tensor = torch.from_numpy(states[start:stop]).float().to(device)
            action_tensor = (
                torch.from_numpy(action_feature_values[start:stop]).float().to(device)
            )
            predictions.append(model(state_tensor, action_tensor).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)


def infer_run_groups(
    df: pd.DataFrame,
    requested_column: str | None,
) -> tuple[np.ndarray | None, str | None]:
    """Choose a complete-run grouping for macro validation metrics."""

    if requested_column:
        if requested_column not in df.columns:
            raise ValueError(
                f"--eval-group-column {requested_column!r} is absent from the evaluation dataset"
            )
        return df[requested_column].fillna("<missing>").astype(str).to_numpy(), requested_column
    if "source_file" in df.columns:
        return df["source_file"].fillna("<missing>").astype(str).to_numpy(), "source_file"
    if {"meta_scenario_id", "meta_repeat_idx"}.issubset(df.columns):
        groups = (
            df["meta_scenario_id"].fillna("<missing>").astype(str)
            + "::repeat="
            + df["meta_repeat_idx"].fillna("<missing>").astype(str)
        )
        return groups.to_numpy(), "meta_scenario_id+meta_repeat_idx"
    if "meta_scenario_id" in df.columns:
        return (
            df["meta_scenario_id"].fillna("<missing>").astype(str).to_numpy(),
            "meta_scenario_id",
        )
    return None, None


def score_all_actions(model: ActionRewardNetwork, states: np.ndarray, action_dim: int, device: str, batch_size: int = 4096) -> np.ndarray:
    all_scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            state_batch = states[start : start + batch_size]
            repeated_states = np.repeat(state_batch, action_dim, axis=0)
            actions = np.tile(np.arange(action_dim, dtype=np.int64), len(state_batch))
            action_feats = action_features(actions, action_dim)

            state_tensor = torch.from_numpy(repeated_states).float().to(device)
            action_tensor = torch.from_numpy(action_feats).float().to(device)
            scores = model(state_tensor, action_tensor).cpu().numpy()
            all_scores.append(scores.reshape(len(state_batch), action_dim))
    return np.vstack(all_scores)


def select_actions(scores: np.ndarray, objective: str) -> np.ndarray:
    if objective == "minimize":
        return np.argmin(scores, axis=1)
    return np.argmax(scores, axis=1)


def train_reward_model(
    dataset_csv,
    output_model,
    epochs=30,
    batch_size=256,
    eval_split=0.2,
    hidden_dim=128,
    lr=1e-3,
    weight_decay=1e-4,
    loss_type="huber",
    target_column="reward",
    objective="maximize",
    clip_target_min=None,
    clip_target_max=None,
    target_scale=1.0,
    max_train_rows=None,
    checkpoint_metric="eval_loss",
    policy_eval_every=1,
    device="cpu",
    seed=42,
    delivered_only=False,
    observed_only=False,
    fresh_state_only=False,
    ignore_state_mcs=False,
    state_schema="auto",
    phase_dropout=0.0,
    phase_consistency_weight=0.0,
    early_stopping_patience=None,
    early_stopping_min_epochs=1,
    early_stopping_min_delta=0.0,
    train_monitor_rows=8192,
    eval_group_column=None,
    eval_dataset=None,
    batch_trace_every=0,
):
    start = time.perf_counter()
    print("[Action-Conditioned Outcome Model Training]", flush=True)
    print(f"  Dataset: {dataset_csv}", flush=True)
    if eval_dataset is not None:
        print(f"  External evaluation dataset: {eval_dataset}", flush=True)
    print(f"  Output: {output_model}", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"  Seed: {seed}", flush=True)
    print(f"  Target column: {target_column} ({objective})", flush=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1 when enabled")
    if early_stopping_min_epochs < 1:
        raise ValueError("early_stopping_min_epochs must be at least 1")
    if early_stopping_min_epochs > epochs:
        raise ValueError("early_stopping_min_epochs cannot exceed epochs")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    if train_monitor_rows < 1:
        raise ValueError("train_monitor_rows must be at least 1")
    if int(batch_trace_every) != batch_trace_every or batch_trace_every < 0:
        raise ValueError("batch_trace_every must be a non-negative integer")
    batch_trace_every = int(batch_trace_every)

    def load_and_filter(path, label):
        frame = pd.read_csv(path)
        original_rows = len(frame)
        frame["_trainer_source_row_index"] = np.arange(
            original_rows, dtype=np.int64
        )
        print(f"  {label} rows: {original_rows}", flush=True)
        required = {"mcs_index", target_column, "iq_raw"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} dataset missing required columns: {missing}")

        if delivered_only:
            if "delivered" not in frame.columns:
                raise ValueError(
                    f"--delivered-only requires a 'delivered' column in the {label} dataset"
                )
            before = len(frame)
            frame = frame[frame["delivered"] == 1].reset_index(drop=True)
            fraction = len(frame) / before * 100.0 if before else 0.0
            print(
                f"  {label} --delivered-only: kept {len(frame)} / {before} rows "
                f"({fraction:.1f}%)",
                flush=True,
            )

        if observed_only:
            before = len(frame)
            synthetic = pd.to_numeric(
                frame.get("synthetic_stale", pd.Series(0, index=frame.index)),
                errors="coerce",
            ).fillna(0)
            model_augmented = pd.to_numeric(
                frame.get("model_augmented", pd.Series(0, index=frame.index)),
                errors="coerce",
            ).fillna(0)
            frame = frame[(synthetic == 0) & (model_augmented == 0)].reset_index(
                drop=True
            )
            print(
                f"  {label} --observed-only: kept {len(frame)} / {before} rows",
                flush=True,
            )

        if fresh_state_only:
            if "state_age_packets" not in frame.columns:
                raise ValueError(
                    f"--fresh-state-only requires state_age_packets in the {label} dataset"
                )
            before = len(frame)
            ages = pd.to_numeric(frame["state_age_packets"], errors="coerce")
            frame = frame[ages == 1].reset_index(drop=True)
            print(
                f"  {label} --fresh-state-only: kept {len(frame)} / {before} rows",
                flush=True,
            )

        if frame.empty:
            raise ValueError(f"No {label} rows remain after reward-model filters")
        frame["iq_raw_parsed"] = frame["iq_raw"].apply(parse_iq_raw)
        return frame, original_rows

    source_df, source_original_rows = load_and_filter(dataset_csv, "Training input")
    if eval_dataset is not None:
        external_eval_df, eval_original_rows = load_and_filter(
            eval_dataset, "External evaluation input"
        )
        train_df = source_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        eval_df = external_eval_df.reset_index(drop=True)
        split_mode = "external_dataset"
        print(
            "  Evaluation split: external dataset (no shuffled training rows in evaluation)",
            flush=True,
        )
    else:
        # Backward-compatible optimization split. Qualification should use an
        # external dataset whose complete runs/scenarios were held out upstream.
        source_df = source_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split_idx = int(len(source_df) * (1 - eval_split))
        train_df = source_df.iloc[:split_idx].reset_index(drop=True)
        eval_df = source_df.iloc[split_idx:].reset_index(drop=True)
        eval_original_rows = source_original_rows
        split_mode = "shuffled_row_split"
        if train_df.empty or eval_df.empty:
            raise ValueError("eval_split must leave at least one train and evaluation row")

    if max_train_rows is not None and max_train_rows < len(train_df):
        train_df = train_df.sample(n=max_train_rows, random_state=seed).reset_index(drop=True)
        print(f"  Sampled train rows: {len(train_df)}", flush=True)

    stable_row_id_overlap_count = None
    if (
        eval_dataset is not None
        and "v33_row_id" in train_df.columns
        and "v33_row_id" in eval_df.columns
    ):
        for label, frame in (("Training", train_df), ("Evaluation", eval_df)):
            if frame["v33_row_id"].isna().any() or frame["v33_row_id"].duplicated().any():
                raise ValueError(
                    f"{label} v33_row_id values must be present and unique"
                )
        train_row_ids = set(train_df["v33_row_id"].astype(str))
        eval_row_ids = set(eval_df["v33_row_id"].astype(str))
        stable_row_id_overlap_count = len(train_row_ids & eval_row_ids)
        del train_row_ids, eval_row_ids
        if stable_row_id_overlap_count:
            raise ValueError(
                "Training and external evaluation datasets overlap by "
                f"{stable_row_id_overlap_count} stable v33_row_id value(s)"
            )

    # The selected frames are independent copies. Release the full inputs before
    # feature precomputation so large external qualification sets do not double
    # peak dataframe memory.
    del source_df
    if eval_dataset is not None:
        del external_eval_df

    print(f"  Train rows: {len(train_df)}, Eval rows: {len(eval_df)}", flush=True)
    print("  Hashing input artifacts for provenance...", flush=True)
    train_artifact = artifact_provenance(dataset_csv)
    if eval_dataset is not None:
        eval_artifact = artifact_provenance(eval_dataset)
        if train_artifact["sha256"] == eval_artifact["sha256"]:
            raise ValueError(
                "--eval-dataset has the same SHA-256 as the training dataset; "
                "provide an independently grouped holdout"
            )
    else:
        eval_artifact = train_artifact
    eval_groups, resolved_eval_group_column = infer_run_groups(
        eval_df, eval_group_column
    )
    provenance = {
        "split_mode": split_mode,
        "seed": int(seed),
        "training_artifact": train_artifact,
        "evaluation_artifact": eval_artifact,
        "training_input_original_rows": int(source_original_rows),
        "evaluation_input_original_rows": int(eval_original_rows),
        "selected_train_rows": int(len(train_df)),
        "selected_eval_rows": int(len(eval_df)),
        "selected_train_source_rows_sha256": selected_row_indices_sha256(train_df),
        "selected_eval_source_rows_sha256": selected_row_indices_sha256(eval_df),
        "eval_split": None if eval_dataset is not None else float(eval_split),
        "eval_group_column": resolved_eval_group_column,
        "stable_row_id_overlap_count": stable_row_id_overlap_count,
        "filters": {
            "delivered_only": bool(delivered_only),
            "observed_only": bool(observed_only),
            "fresh_state_only": bool(fresh_state_only),
        },
    }
    if resolved_eval_group_column is not None:
        print(
            f"  Evaluation macro grouping: {resolved_eval_group_column} "
            f"({len(pd.unique(eval_groups))} groups)",
            flush=True,
        )

    precompute_start = time.perf_counter()
    train_rows = list(train_df.itertuples(index=False))
    eval_rows = list(eval_df.itertuples(index=False))
    state_context_feature = (
        "state_age_packets" if "state_age_packets" in train_df.columns else "sig_len"
    )
    include_state_mcs = "state_mcs_index" in train_df.columns and not ignore_state_mcs
    state_schema = resolve_state_schema_for_columns(train_df.columns, state_schema)
    validate_dataset_feature_contract(dataset_csv, state_schema)
    if eval_dataset is not None:
        validate_dataset_feature_contract(eval_dataset, state_schema)
    if not 0.0 <= phase_dropout < 1.0:
        raise ValueError("phase_dropout must be in [0, 1)")
    if phase_consistency_weight < 0.0:
        raise ValueError("phase_consistency_weight must be non-negative")
    if (phase_dropout > 0.0 or phase_consistency_weight > 0.0) and state_schema != "link_v7c":
        raise ValueError("Phase regularization is supported only for state_schema=link_v7c")
    train_states = np.array(
        [
            build_state_vector(row, state_context_feature, include_state_mcs, state_schema)
            for row in train_rows
        ],
        dtype=np.float32,
    )
    eval_states = np.array(
        [
            build_state_vector(row, state_context_feature, include_state_mcs, state_schema)
            for row in eval_rows
        ],
        dtype=np.float32,
    )
    state_dim = int(train_states.shape[1])
    if len(eval_states) and eval_states.shape[1] != state_dim:
        raise ValueError(
            f"Train/eval state dimension mismatch: {state_dim} vs {eval_states.shape[1]}"
        )
    for label, values in (("training", train_states), ("evaluation", eval_states)):
        non_finite = int(np.size(values) - np.isfinite(values).sum())
        if non_finite:
            raise ValueError(
                f"{label.capitalize()} state matrix contains {non_finite} non-finite value(s)"
            )
    print(f"  State context feature: {state_context_feature}", flush=True)
    print(f"  State schema: {state_schema}", flush=True)
    print(
        f"  Source MCS conditioning: {'enabled (one-hot)' if include_state_mcs else 'disabled'}",
        flush=True,
    )
    if ignore_state_mcs and "state_mcs_index" in train_df.columns:
        print("  Source MCS intentionally ignored for counterfactual action scoring", flush=True)
    print(f"  State dimensions: {state_dim}", flush=True)
    if state_schema == "link_v7c":
        print(
            f"  Phase regularization: dropout={phase_dropout:.3f}, "
            f"consistency_weight={phase_consistency_weight:.3f}",
            flush=True,
        )
    print(f"  Precomputed states in {time.perf_counter() - precompute_start:.1f}s", flush=True)

    state_mean = train_states.mean(axis=0).astype(np.float32)
    state_std = train_states.std(axis=0).astype(np.float32)
    state_std = np.where(state_std < 1e-6, 1.0, state_std).astype(np.float32)
    train_states = normalize(train_states, state_mean, state_std).astype(np.float32)
    eval_states = normalize(eval_states, state_mean, state_std).astype(np.float32)

    def validated_actions(frame: pd.DataFrame, label: str) -> np.ndarray:
        numeric = pd.to_numeric(frame["mcs_index"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{label} mcs_index contains non-finite values")
        if not np.all(numeric == np.floor(numeric)) or not np.all(
            (numeric >= 0) & (numeric < 8)
        ):
            invalid = numeric[(numeric != np.floor(numeric)) | (numeric < 0) | (numeric >= 8)]
            raise ValueError(
                f"{label} mcs_index must contain integers in [0, 7]; "
                f"examples={invalid[:5].tolist()}"
            )
        return numeric.astype(np.int64)

    def validated_targets(frame: pd.DataFrame, label: str) -> np.ndarray:
        numeric = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(
            dtype=np.float32
        )
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{label} {target_column} contains non-finite values")
        return numeric

    train_actions = validated_actions(train_df, "Training")
    eval_actions = validated_actions(eval_df, "Evaluation")
    train_targets = validated_targets(train_df, "Training")
    eval_targets = validated_targets(eval_df, "Evaluation")

    if clip_target_min is not None:
        train_targets = np.maximum(train_targets, clip_target_min).astype(np.float32)
        eval_targets = np.maximum(eval_targets, clip_target_min).astype(np.float32)
    if clip_target_max is not None:
        train_targets = np.minimum(train_targets, clip_target_max).astype(np.float32)
        eval_targets = np.minimum(eval_targets, clip_target_max).astype(np.float32)
    if target_scale <= 0:
        raise ValueError("target_scale must be positive")
    if target_scale != 1.0:
        train_targets = (train_targets / target_scale).astype(np.float32)
        eval_targets = (eval_targets / target_scale).astype(np.float32)
        print(f"  Target scaling: target / {target_scale}", flush=True)

    # Break down by SNR quintile instead of a global aggregate.
    # A global aggregate conflates channel conditions and creates a false "global best MCS"
    # impression. The quintile view exposes the inversion: high-SNR contexts favour high MCS
    # (faster rate, reliable link), low-SNR contexts favour low MCS (robust, fewer retries).
    _snr_col = "snr" if "snr" in train_df.columns else None
    if _snr_col is not None and train_df[_snr_col].notna().sum() > 100:
        try:
            train_df["_snr_q"] = pd.qcut(
                train_df[_snr_col], q=5,
                labels=["Q1(low-SNR)", "Q2", "Q3", "Q4", "Q5(high-SNR)"],
                duplicates="drop",
            )
            print(f"  Context-conditional median {target_column} by MCS (SNR quintile):", flush=True)
            print(f"  (goal: each row should show a different best MCS — that is the context-dependence the model must learn)", flush=True)
            pivot = train_df.groupby(["_snr_q", "mcs_index"], observed=False)[target_column].median().unstack("mcs_index")
            pivot.columns = [f"MCS{int(c)}" for c in pivot.columns]
            for qbin, qrow in pivot.iterrows():
                valid = qrow.dropna()
                best_mcs = (valid.idxmax() if objective == "maximize" else valid.idxmin()) if not valid.empty else "n/a"
                row_str = "  ".join(f"{c}={v:.3f}" for c, v in valid.items())
                print(f"    {qbin}: {row_str}  → best: {best_mcs}", flush=True)
            train_df.drop(columns=["_snr_q"], inplace=True)
        except Exception as _e:
            print(f"  (context-conditional breakdown skipped: {_e})", flush=True)
    else:
        print(f"  (snr column not available; skipping context-conditional breakdown)", flush=True)

    train_action_features = action_features(train_actions)
    eval_action_features = action_features(eval_actions)

    monitor_count = min(int(train_monitor_rows), len(train_states))
    monitor_rng = np.random.default_rng(seed + 104729)
    if monitor_count == len(train_states):
        train_monitor_indices = np.arange(len(train_states), dtype=np.int64)
    else:
        train_monitor_indices = np.sort(
            monitor_rng.choice(len(train_states), size=monitor_count, replace=False)
        )
    monitor_states = train_states[train_monitor_indices]
    monitor_action_features = train_action_features[train_monitor_indices]
    monitor_actions = train_actions[train_monitor_indices]
    monitor_targets = train_targets[train_monitor_indices]
    provenance["train_monitor_rows"] = int(monitor_count)
    provenance["train_monitor_source_rows_sha256"] = selected_row_indices_sha256(
        train_df.iloc[train_monitor_indices]
    )
    print(f"  Fixed train-monitor rows: {monitor_count}", flush=True)

    train_ds = TensorDataset(
        torch.from_numpy(train_states),
        torch.from_numpy(train_action_features),
        torch.from_numpy(train_targets),
    )
    # Keep minibatch ordering paired across amp/full variants.  A dedicated
    # generator prevents their different input-layer initializations from
    # consuming different amounts of the global Torch RNG before sampling.
    loader_seed = int(seed) + 271_828
    loader_generator = torch.Generator()
    loader_generator.manual_seed(loader_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    provenance["data_loader_seed"] = loader_seed
    provenance["batch_trace_every"] = batch_trace_every

    model = ActionRewardNetwork(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss() if loss_type == "huber" else nn.MSELoss()

    log = {
        "epochs": [],
        # Backward-compatible alias for the total backpropagated objective.
        "train_losses": [],
        "train_total_objective_losses": [],
        "train_full_predictive_losses": [],
        "train_dropout_predictive_losses": [],
        "train_consistency_losses": [],
        "train_monitor_predictive_losses": [],
        "train_monitor_metrics": [],
        "eval_losses": [],
        "eval_mae": [],
        "eval_rmse": [],
        "eval_bias": [],
        "eval_macro_run_loss": [],
        "eval_metrics": [],
        "optimizer_steps_at_epoch_end": [],
        "batch_trace_every": batch_trace_every,
        "batch_training_trace": [],
        "initial_train_monitor_metrics": None,
        "initial_eval_metrics": None,
        "replay_target": [],
        "replay_coverage": [],
        "state_schema": state_schema,
        "phase_dropout": phase_dropout,
        "phase_consistency_weight": phase_consistency_weight,
        "phase_regularization_scope": {
            "start": LINK_V7C_PHASE_DERIVED_START,
            "stop": LINK_V7C_PHASE_DERIVED_STOP,
            "feature_count": LINK_V7C_PHASE_DERIVED_STOP
            - LINK_V7C_PHASE_DERIVED_START,
            "includes_quality_summaries": True,
        },
        "loss_type": loss_type,
        "external_eval_dataset": eval_dataset is not None,
        "provenance": provenance,
        "early_stopping": {
            "enabled": early_stopping_patience is not None,
            "patience": early_stopping_patience,
            "min_epochs": int(early_stopping_min_epochs),
            "min_delta": float(early_stopping_min_delta),
            "monitor": (
                "eval_macro_run_loss"
                if checkpoint_metric == "eval_macro_run_loss"
                else "eval_loss"
            ),
            "best_value": None,
            "stopped_early": False,
            "stop_epoch": None,
        },
    }
    if batch_trace_every > 0:
        # Evaluation-only forward passes do not consume RNG, so the traced and
        # untraced runs retain identical initialization, minibatch order, phase
        # masks, optimization, and selected checkpoint.
        initial_train_predictions = prediction_batches(
            model,
            monitor_states,
            monitor_action_features,
            device,
        )
        initial_eval_predictions = prediction_batches(
            model,
            eval_states,
            eval_action_features,
            device,
        )
        log["initial_train_monitor_metrics"] = observed_prediction_metrics(
            initial_train_predictions,
            monitor_targets,
            monitor_actions,
            loss_type,
        )
        log["initial_eval_metrics"] = observed_prediction_metrics(
            initial_eval_predictions,
            eval_targets,
            eval_actions,
            loss_type,
            groups=eval_groups,
        )
        print(
            f"  Dense optimizer trace: every {batch_trace_every} batch(es), "
            "including step-0 train-monitor/evaluation baselines",
            flush=True,
        )
    best_eval_loss = float("inf")
    best_macro_run_loss = float("inf")
    best_policy_target = float("inf") if objective == "minimize" else -float("inf")
    best_epoch = -1
    best_state = None
    best_checkpoint_metric_name = None
    best_checkpoint_metric_value = None
    early_stop_best = None
    early_stop_bad_epochs = 0

    eval_state_tensor = torch.from_numpy(eval_states).float().to(device)
    eval_action_tensor = torch.from_numpy(eval_action_features).float().to(device)
    policy_eval_every = max(int(policy_eval_every), 1)
    batches_per_epoch = len(train_loader)
    optimizer_step = 0
    examples_seen = 0

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        total_losses = []
        full_predictive_losses = []
        dropout_predictive_losses = []
        consistency_losses = []
        for batch_index, (state_batch, action_batch, target_batch) in enumerate(
            train_loader
        ):
            state_batch = state_batch.float().to(device)
            action_batch = action_batch.float().to(device)
            target_batch = target_batch.float().to(device)

            full_pred = model(state_batch, action_batch)
            full_predictive_loss = criterion(full_pred, target_batch)
            predictive_objective = full_predictive_loss
            dropout_predictive_loss = None
            consistency_loss = None

            if state_schema == "link_v7c" and phase_dropout > 0.0:
                dropout_mask = (
                    torch.rand((state_batch.shape[0], 1), device=state_batch.device)
                    < phase_dropout
                )
                phase_dropout_state = state_batch.clone()
                phase_dropout_state[
                    :,
                    LINK_V7C_PHASE_DERIVED_START:LINK_V7C_PHASE_DERIVED_STOP,
                ] = torch.where(
                    dropout_mask,
                    torch.zeros_like(
                        phase_dropout_state[
                            :,
                            LINK_V7C_PHASE_DERIVED_START:LINK_V7C_PHASE_DERIVED_STOP,
                        ]
                    ),
                        phase_dropout_state[
                            :,
                            LINK_V7C_PHASE_DERIVED_START:LINK_V7C_PHASE_DERIVED_STOP,
                        ],
                )
                dropout_pred = model(phase_dropout_state, action_batch)
                dropout_predictive_loss = criterion(dropout_pred, target_batch)
                predictive_objective = 0.5 * (
                    full_predictive_loss + dropout_predictive_loss
                )

            if state_schema == "link_v7c" and phase_consistency_weight > 0.0:
                phase_values_masked_state = state_batch.clone()
                phase_values_masked_state[
                    :,
                    LINK_V7C_PHASE_DERIVED_START:LINK_V7C_PHASE_DERIVED_STOP,
                ] = 0.0
                phase_values_masked_pred = model(phase_values_masked_state, action_batch)
                consistency_loss = nn.functional.mse_loss(
                    full_pred,
                    phase_values_masked_pred,
                )
            total_objective = predictive_objective
            if consistency_loss is not None:
                total_objective = (
                    total_objective + phase_consistency_weight * consistency_loss
                )

            optimizer.zero_grad()
            total_objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_step += 1
            examples_seen += int(target_batch.shape[0])
            total_objective_value = float(total_objective.item())
            full_predictive_value = float(full_predictive_loss.item())
            dropout_predictive_value = (
                None
                if dropout_predictive_loss is None
                else float(dropout_predictive_loss.item())
            )
            consistency_value = (
                None if consistency_loss is None else float(consistency_loss.item())
            )
            total_losses.append(total_objective_value)
            full_predictive_losses.append(full_predictive_value)
            if dropout_predictive_loss is not None:
                dropout_predictive_losses.append(dropout_predictive_value)
            if consistency_loss is not None:
                consistency_losses.append(consistency_value)
            if batch_trace_every > 0 and (
                optimizer_step == 1
                or optimizer_step % batch_trace_every == 0
                or batch_index + 1 == batches_per_epoch
            ):
                log["batch_training_trace"].append(
                    {
                        "optimizer_step": optimizer_step,
                        "examples_seen": examples_seen,
                        "epoch": epoch + 1,
                        "batch_in_epoch": batch_index + 1,
                        "total_objective_loss": total_objective_value,
                        "intact_predictive_loss": full_predictive_value,
                        "phase_dropout_predictive_loss": dropout_predictive_value,
                        "phase_consistency_loss": consistency_value,
                    }
                )

        model.eval()
        with torch.no_grad():
            eval_pred = model(eval_state_tensor, eval_action_tensor)
        eval_predictions = eval_pred.cpu().numpy()
        eval_metrics = observed_prediction_metrics(
            eval_predictions,
            eval_targets,
            eval_actions,
            loss_type,
            groups=eval_groups,
        )
        eval_loss = float(eval_metrics["loss"])
        train_monitor_predictions = prediction_batches(
            model,
            monitor_states,
            monitor_action_features,
            device,
        )
        train_monitor_metrics = observed_prediction_metrics(
            train_monitor_predictions,
            monitor_targets,
            monitor_actions,
            loss_type,
        )
        train_monitor_loss = float(train_monitor_metrics["loss"])

        replay_target = float("nan")
        replay_coverage = float("nan")
        if len(eval_df) and ((epoch + 1) % policy_eval_every == 0 or epoch == epochs - 1):
            all_scores = score_all_actions(model, eval_states, 8, device)
            policy_actions = select_actions(all_scores, objective)
            matches = policy_actions == eval_actions
            # Replay coverage measures agreement with the logged policy, not
            # policy quality.  The score-improvement field below is likewise a
            # model-internal calibration diagnostic; offline IPS/SNIPS on a
            # randomized holdout remains the policy-quality evidence.
            replay_coverage = float(np.mean(matches))
            if np.any(matches):
                replay_target = float(np.mean(eval_targets[matches]))
            # Mean predicted score of the recommendation versus the observed
            # target mean. Positive means the model is optimistic about its own
            # choices; it is not a counterfactual performance estimate.
            policy_scores = all_scores[np.arange(len(all_scores)), policy_actions]
            policy_score_improvement = float(np.mean(policy_scores) - np.mean(eval_targets))
        else:
            policy_score_improvement = float("nan")

        train_loss = (
            float(np.mean(total_losses)) if total_losses else float("nan")
        )
        train_full_predictive_loss = (
            float(np.mean(full_predictive_losses))
            if full_predictive_losses
            else float("nan")
        )
        train_dropout_predictive_loss = (
            float(np.mean(dropout_predictive_losses))
            if dropout_predictive_losses
            else None
        )
        train_consistency_loss = (
            float(np.mean(consistency_losses)) if consistency_losses else None
        )
        monitored_values = {
            "train objective": train_loss,
            "train intact predictive loss": train_full_predictive_loss,
            "fixed train-monitor loss": train_monitor_loss,
            "evaluation loss": eval_loss,
            "evaluation MAE": float(eval_metrics["mae"]),
            "evaluation RMSE": float(eval_metrics["rmse"]),
            "evaluation bias": float(eval_metrics["bias"]),
        }
        if eval_metrics["macro_run_loss"] is not None:
            monitored_values["macro-run evaluation loss"] = float(
                eval_metrics["macro_run_loss"]
            )
        if train_dropout_predictive_loss is not None:
            monitored_values["phase-dropout predictive loss"] = (
                train_dropout_predictive_loss
            )
        if train_consistency_loss is not None:
            monitored_values["phase consistency loss"] = train_consistency_loss
        non_finite_metrics = [
            name for name, value in monitored_values.items() if not np.isfinite(value)
        ]
        if non_finite_metrics:
            raise FloatingPointError(
                f"Epoch {epoch + 1} produced non-finite metrics: "
                + ", ".join(non_finite_metrics)
            )
        log["epochs"].append(epoch)
        log["train_losses"].append(train_loss)
        log["train_total_objective_losses"].append(train_loss)
        log["train_full_predictive_losses"].append(train_full_predictive_loss)
        log["train_dropout_predictive_losses"].append(
            train_dropout_predictive_loss
        )
        log["train_consistency_losses"].append(train_consistency_loss)
        log["train_monitor_predictive_losses"].append(train_monitor_loss)
        log["train_monitor_metrics"].append(train_monitor_metrics)
        log["eval_losses"].append(eval_loss)
        log["eval_mae"].append(float(eval_metrics["mae"]))
        log["eval_rmse"].append(float(eval_metrics["rmse"]))
        log["eval_bias"].append(float(eval_metrics["bias"]))
        log["eval_macro_run_loss"].append(eval_metrics["macro_run_loss"])
        log["eval_metrics"].append(eval_metrics)
        log["optimizer_steps_at_epoch_end"].append(optimizer_step)
        log["replay_target"].append(replay_target)
        log["replay_coverage"].append(replay_coverage)
        log.setdefault("policy_score_improvement", []).append(policy_score_improvement)

        improved = False
        checkpoint_candidate_name = "eval_loss"
        checkpoint_candidate_value = eval_loss
        if checkpoint_metric == "replay_target" and not np.isnan(replay_target):
            checkpoint_candidate_name = "replay_target"
            checkpoint_candidate_value = replay_target
            if objective == "minimize":
                improved = replay_target < best_policy_target
            else:
                improved = replay_target > best_policy_target
        elif checkpoint_metric == "eval_macro_run_loss":
            macro_run_loss = eval_metrics["macro_run_loss"]
            if macro_run_loss is None:
                raise ValueError(
                    "checkpoint_metric=eval_macro_run_loss requires evaluation run groups; "
                    "provide --eval-group-column or source run metadata"
                )
            checkpoint_candidate_name = "eval_macro_run_loss"
            checkpoint_candidate_value = float(macro_run_loss)
            improved = checkpoint_candidate_value < best_macro_run_loss
        else:
            improved = eval_loss < best_eval_loss

        if improved:
            best_eval_loss = eval_loss
            if eval_metrics["macro_run_loss"] is not None:
                best_macro_run_loss = float(eval_metrics["macro_run_loss"])
            if not np.isnan(replay_target):
                best_policy_target = replay_target
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_checkpoint_metric_name = checkpoint_candidate_name
            best_checkpoint_metric_value = checkpoint_candidate_value

        # Early stopping deliberately uses a stable predictive metric. Replay
        # target is sparse and is not available on every policy-evaluation epoch.
        early_stop_value = (
            float(eval_metrics["macro_run_loss"])
            if checkpoint_metric == "eval_macro_run_loss"
            else eval_loss
        )
        if early_stop_best is None:
            early_stop_best = early_stop_value
            early_stop_bad_epochs = 0
        else:
            meaningful_improvement = (
                early_stop_value < early_stop_best - early_stopping_min_delta
            )
            if meaningful_improvement:
                early_stop_best = early_stop_value
                early_stop_bad_epochs = 0
            else:
                early_stop_bad_epochs += 1
        log["early_stopping"]["best_value"] = float(early_stop_best)

        print(
            f"  Epoch {epoch + 1}/{epochs}, train_objective={train_loss:.4f}, "
            f"train_monitor={train_monitor_loss:.4f}, eval_loss={eval_loss:.4f}, "
            f"eval_mae={float(eval_metrics['mae']):.4f}, "
            f"eval_rmse={float(eval_metrics['rmse']):.4f}, "
            f"eval_bias={float(eval_metrics['bias']):+.4f}, "
            f"replay_target={replay_target:.4f}, "
            f"policy_score_improvement={policy_score_improvement:.4f}, "
            f"replay_coverage={replay_coverage:.4f}, time={time.perf_counter() - epoch_start:.1f}s",
            flush=True,
        )

        if (
            early_stopping_patience is not None
            and epoch + 1 >= early_stopping_min_epochs
            and early_stop_bad_epochs >= early_stopping_patience
        ):
            log["early_stopping"]["stopped_early"] = True
            log["early_stopping"]["stop_epoch"] = int(epoch)
            print(
                f"  Early stopping at epoch {epoch + 1}: no improvement larger than "
                f"{early_stopping_min_delta:g} for {early_stop_bad_epochs} monitored epochs",
                flush=True,
            )
            break

    if best_state is None:
        raise FloatingPointError("No finite validation checkpoint was produced")
    model.load_state_dict(best_state)
    print(
        f"  Restored best checkpoint from epoch {best_epoch + 1} "
        f"({best_checkpoint_metric_name}={best_checkpoint_metric_value:.4f}; "
        f"requested checkpoint metric={checkpoint_metric})",
        flush=True,
    )

    best_eval_predictions = prediction_batches(
        model, eval_states, eval_action_features, device
    )
    best_eval_metrics = observed_prediction_metrics(
        best_eval_predictions,
        eval_targets,
        eval_actions,
        loss_type,
        groups=eval_groups,
    )
    best_train_monitor_predictions = prediction_batches(
        model, monitor_states, monitor_action_features, device
    )
    best_train_monitor_metrics = observed_prediction_metrics(
        best_train_monitor_predictions,
        monitor_targets,
        monitor_actions,
        loss_type,
    )
    log["best_epoch"] = int(best_epoch)
    log["epochs_completed"] = int(len(log["epochs"]))
    log["best_checkpoint_metric"] = best_checkpoint_metric_name
    log["best_checkpoint_metric_value"] = best_checkpoint_metric_value
    log["best_eval_metrics"] = best_eval_metrics
    log["best_train_monitor_metrics"] = best_train_monitor_metrics

    output_model = Path(output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
            "model_state": model.state_dict(),
            "state_dim": state_dim,
            "action_dim": 8,
            "action_feature_dim": 9,
            "hidden_dim": hidden_dim,
            "state_schema": state_schema,
            "state_context_feature": state_context_feature,
            "include_state_mcs": include_state_mcs,
            "state_feature_names": state_feature_names(state_schema, include_state_mcs),
            **feature_contract_metadata(state_schema),
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "training_log": log,
            "best_epoch": best_epoch,
            "best_checkpoint_metric": best_checkpoint_metric_name,
            "best_checkpoint_metric_value": best_checkpoint_metric_value,
            "best_eval_loss": best_eval_loss,
            "best_eval_metrics": best_eval_metrics,
            "best_train_monitor_metrics": best_train_monitor_metrics,
            "best_macro_run_loss": best_macro_run_loss,
            "best_policy_target": best_policy_target,
            "checkpoint_metric": checkpoint_metric,
            "target_column": target_column,
            "objective": objective,
            "target_scale": target_scale,
            "clip_target_min": clip_target_min,
            "clip_target_max": clip_target_max,
            "observed_only": observed_only,
            "fresh_state_only": fresh_state_only,
            "ignore_state_mcs": ignore_state_mcs,
            "phase_dropout": phase_dropout,
            "phase_consistency_weight": phase_consistency_weight,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_epochs": early_stopping_min_epochs,
            "early_stopping_min_delta": early_stopping_min_delta,
            "train_monitor_rows": monitor_count,
            "batch_trace_every": batch_trace_every,
            "eval_group_column": resolved_eval_group_column,
            "provenance": provenance,
        }
    checkpoint_temporary = output_model.with_name(
        f".{output_model.name}.tmp.{os.getpid()}"
    )
    try:
        torch.save(checkpoint_payload, checkpoint_temporary)
        checkpoint_temporary.replace(output_model)
    finally:
        checkpoint_temporary.unlink(missing_ok=True)
    checkpoint_sha256 = sha256_file(output_model)
    provenance["checkpoint_path"] = str(output_model.resolve())
    provenance["checkpoint_sha256"] = checkpoint_sha256
    provenance_path = output_model.parent / "training_provenance.json"
    atomic_write_text(provenance_path, json.dumps(provenance, indent=2) + "\n")
    log_path = output_model.parent / "reward_model_training_log.json"
    atomic_write_text(log_path, json.dumps(log, indent=2) + "\n")
    print(f"  Model saved to: {output_model}", flush=True)
    print(f"  Training log saved to: {log_path}", flush=True)
    print(f"  Training provenance saved to: {provenance_path}", flush=True)
    print(f"✅ Outcome model training complete in {time.perf_counter() - start:.1f}s!", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train supervised action-conditioned outcome model")
    parser.add_argument("--dataset", required=True, help="Path to DQN dataset CSV")
    parser.add_argument(
        "--eval-dataset",
        default=None,
        help=(
            "Optional independently grouped evaluation CSV. When provided, no "
            "training rows are used for evaluation and --eval-split is ignored."
        ),
    )
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-split", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--target-column", default="reward", help="Dataset column to predict, e.g. reward or service_ms")
    parser.add_argument("--objective", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--clip-target-min", type=float, default=None)
    parser.add_argument("--clip-target-max", type=float, default=None)
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument(
        "--checkpoint-metric",
        choices=["eval_loss", "eval_macro_run_loss", "replay_target"],
        default="eval_loss",
        help=(
            "Which metric selects the saved checkpoint. eval_macro_run_loss "
            "weights complete evaluation runs equally."
        ),
    )
    parser.add_argument(
        "--policy-eval-every",
        type=int,
        default=1,
        help="Evaluate replay estimate every N epochs",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--delivered-only",
        action="store_true",
        default=False,
        help="Filter to delivered==1 packets only before training. "
             "Removes retransmission-timeout delays that swamp the learning signal.",
    )
    parser.add_argument(
        "--observed-only",
        action="store_true",
        help="Exclude legacy synthetic and model-augmented rows",
    )
    parser.add_argument(
        "--fresh-state-only",
        action="store_true",
        help="Train only on causal states with state_age_packets == 1",
    )
    parser.add_argument(
        "--ignore-state-mcs",
        action="store_true",
        help=(
            "Exclude state_mcs_index from the outcome model so candidate MCS "
            "scores do not learn the fixed sweep's keep-current-MCS shortcut"
        ),
    )
    parser.add_argument(
        "--state-schema",
        choices=["auto", "legacy_v1", "link_v2", "link_v3", "link_v4", "link_v5", "link_v6", "link_v7c"],
        default="auto",
        help=(
            "State schema used for training. Use link_v5 for receiver-only "
            "CSI/RSSI/SNR features without ACK-history conditioning. Use "
            "link_v6 for the legacy detrended-phase representation or link_v7c "
            "for the versioned robust full-CSI contract."
        ),
    )
    parser.add_argument(
        "--phase-dropout",
        type=float,
        default=0.0,
        help=(
            "For link_v7c, randomly replace all 110 phase-derived values "
            "(108 differential-phase values plus valid fraction and coherence) "
            "with their normalized mean during training. This anchors the model "
            "to the amplitude/link fallback when phase is unreliable."
        ),
    )
    parser.add_argument(
        "--phase-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "For link_v7c, penalize score changes when all phase-derived values "
            "are masked. Use with phase dropout to limit spurious phase dependence."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop after this many monitored epochs without a meaningful improvement",
    )
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=1,
        help="Run at least this many epochs before early stopping",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum absolute validation-loss decrease that resets patience",
    )
    parser.add_argument(
        "--train-monitor-rows",
        type=int,
        default=8192,
        help=(
            "Fixed deterministic training subset used for end-of-epoch "
            "unregularized predictive metrics"
        ),
    )
    parser.add_argument(
        "--batch-trace-every",
        type=int,
        default=0,
        help=(
            "Record optimizer-step losses every N batches plus step-0 and "
            "epoch-boundary points; 0 disables dense tracing"
        ),
    )
    parser.add_argument(
        "--eval-group-column",
        default=None,
        help=(
            "Evaluation column identifying complete runs for macro-run metrics. "
            "Defaults to source_file or scenario/repeat metadata when available."
        ),
    )
    args = parser.parse_args()

    train_reward_model(
        dataset_csv=args.dataset,
        output_model=args.output,
        eval_dataset=args.eval_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_split=args.eval_split,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss,
        target_column=args.target_column,
        objective=args.objective,
        clip_target_min=args.clip_target_min,
        clip_target_max=args.clip_target_max,
        target_scale=args.target_scale,
        max_train_rows=args.max_train_rows,
        checkpoint_metric=args.checkpoint_metric,
        policy_eval_every=args.policy_eval_every,
        device=args.device,
        seed=args.seed,
        delivered_only=args.delivered_only,
        observed_only=args.observed_only,
        fresh_state_only=args.fresh_state_only,
        ignore_state_mcs=args.ignore_state_mcs,
        state_schema=args.state_schema,
        phase_dropout=args.phase_dropout,
        phase_consistency_weight=args.phase_consistency_weight,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        early_stopping_min_delta=args.early_stopping_min_delta,
        train_monitor_rows=args.train_monitor_rows,
        eval_group_column=args.eval_group_column,
        batch_trace_every=args.batch_trace_every,
    )


if __name__ == "__main__":
    main()
