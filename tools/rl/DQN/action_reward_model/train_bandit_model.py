#!/usr/bin/env python3
"""
Train a two-head contextual-bandit outcome model for MCS selection.

The model shares a trunk over [state, action_features] and predicts two
calibrated quantities per (state, action) pair:

    P(delivered | state, action)          -- delivery head, BCE loss
    E[log service_ms | state, action, delivered=1]  -- delay head, Huber loss

The policy is derived at prediction time by maximizing expected utility
computed from both heads (see policy_utils.expected_utility). Compared with
regressing a shaped scalar reward, this keeps the two physical outcomes
separately calibrated and lets the objective be re-targeted (goodput, delay,
tail) without rebuilding datasets.

Intended for randomized-logging datasets (RANDOM_SWEEP, per-packet propensity
1/8), where counterfactual action scores are supported by the data. Matched-row
policy diagnostics are reported as replay estimates; they are not called SNIPS
because mixed datasets can include rows with unknown logging propensity.

Usage:
    python train_bandit_model.py \
        --dataset datasets/rl_dqn_dataset_train_utility.csv \
        --output models/bandit_model/bandit_model.pth \
        --epochs 20 --batch-size 512 --state-schema link_v3c --seed 42
"""

from __future__ import annotations

import argparse
import copy
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dqn import (
    build_state_vector,
    feature_contract_metadata,
    resolve_state_schema_for_columns,
    state_feature_names,
    validate_dataset_feature_contract,
)
from train_reward_model import (
    action_features,
    artifact_provenance,
    atomic_write_text,
    infer_run_groups,
    normalize,
    parse_iq_raw,
    selected_row_indices_sha256,
)
from policy_utils import expected_utility, utility_scale_from_delivered_service

ACTION_DIM = 8
MODEL_TYPE = "bandit_two_head_v1"
FAIR_CHECKPOINT_SCHEMA = "receiver_fair_bandit_two_head/v1"
FAIR_OBJECTIVE_CONTRACT = "delivery_log_service_train_global_q95_utility/v1"
FAIR_REQUIRED_CSI_INPUT_SCALAR_COUNT = 114


STATE_INPUT_COLUMNS = {
    "state_mcs_index",
    "state_age_packets",
    "state_packet_gap",
    "state_missing_packets",
    "state_is_stale",
    "state_prev_delivered",
    "state_consecutive_losses",
    "state_recent_loss_rate_8",
    "rssi",
    "snr",
    "fft_gain",
    "agc_gain",
    "channel",
    "sig_len",
    "iq_mean",
    "iq_std",
    "iq_p10",
    "iq_p50",
    "iq_p90",
    "iq_raw",
    "iq_phase_detrended",
    "iq_active_amplitudes",
    "iq_phase_diff_real",
    "iq_phase_diff_imag",
    "iq_phase_valid_fraction",
    "iq_phase_coherence",
}


def _validated_numeric(frame, column, label, *, integer=False):
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} {column} contains non-finite values")
    if integer and not np.all(values == np.floor(values)):
        raise ValueError(f"{label} {column} must contain integers")
    return values


def _load_bandit_frame(
    dataset_csv,
    label,
    *,
    observed_only,
    exact_fresh_only,
    state_schema,
):
    """Load only columns used by the model, outcomes, or provenance audit."""

    artifact = Path(dataset_csv).resolve()
    header = list(pd.read_csv(artifact, nrows=0).columns)
    required = {"mcs_index", "delivered", "service_ms", "iq_raw"}
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"{label} dataset missing required columns: {missing}")

    audit_columns = {
        "reward",
        "source_file",
        "v33_row_id",
        "v33_split",
        "v33_source_kind",
        "synthetic_stale",
        "model_augmented",
    }
    selected = [
        column
        for column in header
        if column in required or column in STATE_INPUT_COLUMNS or column in audit_columns
    ]
    frame = pd.read_csv(artifact, usecols=selected)
    original_rows = len(frame)
    frame["_trainer_source_row_index"] = np.arange(original_rows, dtype=np.int64)
    print(f"  {label} rows: {original_rows}", flush=True)

    if observed_only:
        synthetic = pd.to_numeric(
            frame.get("synthetic_stale", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        augmented = pd.to_numeric(
            frame.get("model_augmented", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        if exact_fresh_only and ((synthetic != 0).any() or (augmented != 0).any()):
            raise ValueError(f"{label} exact-fresh artifact contains augmented rows")
        frame = frame[(synthetic == 0) & (augmented == 0)].reset_index(drop=True)
        print(f"  {label} observed-only: kept {len(frame)} / {original_rows} rows", flush=True)

    if exact_fresh_only:
        exact_columns = {
            "state_age_packets": 1,
            "state_packet_gap": 1,
            "state_missing_packets": 0,
            "state_is_stale": 0,
        }
        absent = sorted(set(exact_columns) - set(frame.columns))
        if absent:
            raise ValueError(
                f"{label} exact-fresh contract is missing columns: {absent}"
            )
        for column, expected in exact_columns.items():
            values = _validated_numeric(frame, column, label, integer=True)
            if not np.all(values == expected):
                raise ValueError(
                    f"{label} is not exact-fresh: {column} must equal {expected}"
                )

    if frame.empty:
        raise ValueError(f"No {label} rows remain after bandit filters")
    actions = _validated_numeric(frame, "mcs_index", label, integer=True)
    if not np.all((actions >= 0) & (actions < ACTION_DIM)):
        raise ValueError(f"{label} mcs_index must be in [0, 7]")
    delivered = _validated_numeric(frame, "delivered", label, integer=True)
    if not np.all(np.isin(delivered, (0, 1))):
        raise ValueError(f"{label} delivered must contain only 0/1")
    service = _validated_numeric(frame, "service_ms", label)
    if np.any(service < 0.0):
        raise ValueError(f"{label} service_ms must be non-negative")
    if "v33_row_id" in frame.columns:
        if frame["v33_row_id"].isna().any() or frame["v33_row_id"].duplicated().any():
            raise ValueError(f"{label} v33_row_id values must be present and unique")

    resolved_schema = resolve_state_schema_for_columns(frame.columns, state_schema)
    validate_dataset_feature_contract(artifact, resolved_schema)
    frame["iq_raw_parsed"] = frame["iq_raw"].apply(parse_iq_raw)
    return frame, original_rows, resolved_schema


def _group_macro_mean(values: np.ndarray, groups: np.ndarray | None) -> float:
    if groups is None:
        return float(np.mean(values))
    return float(
        np.mean([np.mean(values[groups == group]) for group in pd.unique(groups)])
    )


def outcome_metrics(
    logits: np.ndarray,
    log_delay: np.ndarray,
    delivered: np.ndarray,
    delay_targets: np.ndarray,
    *,
    delay_loss_weight: float,
    groups: np.ndarray | None = None,
) -> dict[str, object]:
    """Return row-weighted and equal-source two-head predictive losses."""

    logits = np.asarray(logits, dtype=np.float64)
    log_delay = np.asarray(log_delay, dtype=np.float64)
    delivered = np.asarray(delivered, dtype=np.float64)
    delay_targets = np.asarray(delay_targets, dtype=np.float64)
    bce_values = np.maximum(logits, 0.0) - logits * delivered + np.log1p(
        np.exp(-np.abs(logits))
    )
    delay_error = log_delay - delay_targets
    delay_abs = np.abs(delay_error)
    delay_huber_values = np.where(
        delay_abs < 1.0, 0.5 * np.square(delay_error), delay_abs - 0.5
    )

    def summarize(mask: np.ndarray) -> dict[str, float | int]:
        delivered_mask = mask & (delivered > 0.5)
        bce_value = float(np.mean(bce_values[mask]))
        delay_value = (
            float(np.mean(delay_huber_values[delivered_mask]))
            if np.any(delivered_mask)
            else 0.0
        )
        return {
            "rows": int(np.sum(mask)),
            "delivered_rows": int(np.sum(delivered_mask)),
            "bce": bce_value,
            "delay_huber": delay_value,
            "loss": bce_value + delay_loss_weight * delay_value,
        }

    metrics: dict[str, object] = summarize(np.ones(len(delivered), dtype=bool))
    if groups is None:
        metrics["group_count"] = 0
        metrics["macro_run_loss"] = metrics["loss"]
        metrics["per_group"] = {}
        return metrics

    group_array = np.asarray(groups, dtype=object)
    per_group = {
        str(group): summarize(group_array == group) for group in pd.unique(group_array)
    }
    metrics["group_count"] = len(per_group)
    metrics["macro_run_loss"] = float(
        np.mean([entry["loss"] for entry in per_group.values()])
    )
    metrics["per_group"] = per_group
    return metrics


class BanditOutcomeNetwork(nn.Module):
    """Shared-trunk MLP with delivery-probability and log-delay heads."""

    def __init__(self, state_dim: int, action_feature_dim: int = 9, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_feature_dim = action_feature_dim
        self.hidden_dim = hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.delivery_head = nn.Linear(hidden_dim, 1)
        self.delay_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action_feats: torch.Tensor):
        hidden = self.trunk(torch.cat([state, action_feats], dim=1))
        delivery_logit = self.delivery_head(hidden).squeeze(1)
        log_delay = self.delay_head(hidden).squeeze(1)
        return delivery_logit, log_delay


def score_all_actions(model, states: np.ndarray, device: str, batch_size: int = 4096):
    """Return (p_deliver, log_delay) arrays of shape (rows, ACTION_DIM)."""
    p_out, mu_out = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = states[start : start + batch_size]
            repeated = np.repeat(batch, ACTION_DIM, axis=0)
            actions = np.tile(np.arange(ACTION_DIM, dtype=np.int64), len(batch))
            feats = action_features(actions, ACTION_DIM)
            logit, mu = model(
                torch.from_numpy(repeated).float().to(device),
                torch.from_numpy(feats).float().to(device),
            )
            p = torch.sigmoid(logit).cpu().numpy().reshape(len(batch), ACTION_DIM)
            mu = mu.cpu().numpy().reshape(len(batch), ACTION_DIM)
            p_out.append(p)
            mu_out.append(mu)
    return np.vstack(p_out), np.vstack(mu_out)


def train_bandit_model(
    dataset_csv,
    output_model,
    eval_dataset=None,
    epochs=20,
    batch_size=512,
    eval_split=0.2,
    hidden_dim=128,
    lr=1e-3,
    weight_decay=1e-4,
    delay_loss_weight=1.0,
    delay_clip_ms=50.0,
    utility_payload_bytes=128,
    utility_loss_reward=-1.0,
    utility_tail_target_ms=0.0,
    utility_tail_weight=0.0,
    objective="utility",
    max_train_rows=None,
    checkpoint_metric="eval_loss",
    policy_eval_every=1,
    state_schema="link_v3c",
    ignore_state_mcs=False,
    observed_only=True,
    exact_fresh_only=False,
    eval_group_column=None,
    early_stopping_patience=None,
    early_stopping_min_epochs=10,
    early_stopping_min_delta=1e-4,
    device="cpu",
    seed=42,
):
    start = time.perf_counter()
    print("[Two-Head Bandit Outcome Model Training]", flush=True)
    print(f"  Dataset: {dataset_csv}", flush=True)
    if eval_dataset is not None:
        print(f"  External evaluation dataset: {eval_dataset}", flush=True)
    print(f"  Output: {output_model}", flush=True)
    print(f"  Device: {device}, Seed: {seed}", flush=True)

    if delay_loss_weight < 0.0:
        raise ValueError("delay_loss_weight must be non-negative")
    if delay_clip_ms <= 0.05:
        raise ValueError("delay_clip_ms must be greater than 0.05")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    if early_stopping_min_epochs < 1:
        raise ValueError("early_stopping_min_epochs must be positive")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    source_df, source_original_rows, resolved_schema = _load_bandit_frame(
        dataset_csv,
        "Training input",
        observed_only=observed_only,
        exact_fresh_only=exact_fresh_only,
        state_schema=state_schema,
    )
    if eval_dataset is not None:
        external_eval_df, eval_original_rows, eval_schema = _load_bandit_frame(
            eval_dataset,
            "External evaluation input",
            observed_only=observed_only,
            exact_fresh_only=exact_fresh_only,
            state_schema=state_schema,
        )
        if eval_schema != resolved_schema:
            raise ValueError(
                f"Training/evaluation state schemas differ: {resolved_schema} != {eval_schema}"
            )
        train_df = source_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        eval_df = external_eval_df.reset_index(drop=True)
        split_mode = "external_dataset"
        print(
            "  Evaluation split: external dataset (all training rows remain optimization rows)",
            flush=True,
        )
    else:
        # Backward-compatible row split for exploratory use. Matched benchmark
        # runs must pass an independently grouped external validation artifact.
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
    print(f"  Train rows: {len(train_df)}, Eval rows: {len(eval_df)}", flush=True)

    stable_row_id_overlap_count = None
    if eval_dataset is not None and {"v33_row_id"}.issubset(train_df.columns) and {
        "v33_row_id"
    }.issubset(eval_df.columns):
        train_row_ids = set(train_df["v33_row_id"].astype(str))
        eval_row_ids = set(eval_df["v33_row_id"].astype(str))
        stable_row_id_overlap_count = len(train_row_ids & eval_row_ids)
        del train_row_ids, eval_row_ids
        if stable_row_id_overlap_count:
            raise ValueError(
                "Training and external evaluation datasets overlap by "
                f"{stable_row_id_overlap_count} stable v33_row_id value(s)"
            )

    source_run_overlap_count = None
    if eval_dataset is not None and "source_file" in train_df and "source_file" in eval_df:
        train_sources = set(train_df["source_file"].dropna().astype(str))
        eval_sources = set(eval_df["source_file"].dropna().astype(str))
        source_run_overlap_count = len(train_sources & eval_sources)
        if source_run_overlap_count:
            raise ValueError(
                "Training and external evaluation datasets overlap by "
                f"{source_run_overlap_count} source run(s)"
            )

    print("  Hashing input artifacts for provenance...", flush=True)
    train_artifact = artifact_provenance(dataset_csv)
    eval_artifact = (
        artifact_provenance(eval_dataset) if eval_dataset is not None else train_artifact
    )
    if eval_dataset is not None and train_artifact["sha256"] == eval_artifact["sha256"]:
        raise ValueError("--eval-dataset must not be the training artifact")
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
        "source_run_overlap_count": source_run_overlap_count,
        "filters": {
            "observed_only": bool(observed_only),
            "exact_fresh_only": bool(exact_fresh_only),
        },
    }
    if resolved_eval_group_column is not None:
        print(
            f"  Evaluation macro grouping: {resolved_eval_group_column} "
            f"({len(pd.unique(eval_groups))} groups)",
            flush=True,
        )

    del source_df
    if eval_dataset is not None:
        del external_eval_df

    state_context_feature = (
        "state_age_packets" if "state_age_packets" in train_df.columns else "sig_len"
    )
    include_state_mcs = "state_mcs_index" in train_df.columns and not ignore_state_mcs
    if exact_fresh_only and (
        resolved_schema != "link_v5"
        or include_state_mcs
        or state_context_feature != "state_age_packets"
    ):
        raise ValueError(
            "Fair exact-fresh bandit checkpoints require receiver-only link_v5 "
            "with state_age_packets and --ignore-state-mcs"
        )
    print(f"  State schema: {resolved_schema}", flush=True)
    print(f"  State context feature: {state_context_feature}", flush=True)
    print(
        f"  Source MCS conditioning: {'enabled (one-hot)' if include_state_mcs else 'disabled'}",
        flush=True,
    )

    precompute_start = time.perf_counter()
    train_states = np.array(
        [
            build_state_vector(row, state_context_feature, include_state_mcs, resolved_schema)
            for row in train_df.itertuples(index=False)
        ],
        dtype=np.float32,
    )
    eval_states = np.array(
        [
            build_state_vector(row, state_context_feature, include_state_mcs, resolved_schema)
            for row in eval_df.itertuples(index=False)
        ],
        dtype=np.float32,
    )
    state_dim = int(train_states.shape[1])
    expected_state_dim = len(state_feature_names(resolved_schema, include_state_mcs))
    if train_states.shape != (len(train_df), expected_state_dim):
        raise ValueError(
            f"Training states have shape {train_states.shape}; expected "
            f"{(len(train_df), expected_state_dim)}"
        )
    if eval_states.shape != (len(eval_df), expected_state_dim):
        raise ValueError(
            f"Evaluation states have shape {eval_states.shape}; expected "
            f"{(len(eval_df), expected_state_dim)}"
        )
    if not np.all(np.isfinite(train_states)) or not np.all(np.isfinite(eval_states)):
        raise ValueError("Bandit state matrices contain non-finite values")
    if exact_fresh_only and state_dim != 132:
        raise ValueError(f"Fair link_v5 bandit state_dim must be 132, got {state_dim}")
    print(f"  State dimensions: {state_dim}", flush=True)
    print(f"  Precomputed states in {time.perf_counter() - precompute_start:.1f}s", flush=True)

    state_mean = train_states.mean(axis=0).astype(np.float32)
    state_std = train_states.std(axis=0).astype(np.float32)
    state_std = np.where(state_std < 1e-6, 1.0, state_std).astype(np.float32)
    train_states = normalize(train_states, state_mean, state_std).astype(np.float32)
    eval_states = normalize(eval_states, state_mean, state_std).astype(np.float32)

    train_actions = train_df["mcs_index"].to_numpy(dtype=np.int64)
    eval_actions = eval_df["mcs_index"].to_numpy(dtype=np.int64)
    train_delivered = train_df["delivered"].to_numpy(dtype=np.float32)
    eval_delivered = eval_df["delivered"].to_numpy(dtype=np.float32)

    def log_delay_targets(frame):
        service = pd.to_numeric(frame["service_ms"], errors="coerce").fillna(delay_clip_ms)
        return np.log(np.clip(service.to_numpy(dtype=np.float64), 0.05, delay_clip_ms)).astype(
            np.float32
        )

    train_log_delay = log_delay_targets(train_df)
    eval_log_delay = log_delay_targets(eval_df)

    # Utility scale from delivered train rows, matching the dataset builder's
    # q95 log-goodput convention so derived utilities are comparable to the
    # dataset's reward column.
    utility_scale = utility_scale_from_delivered_service(
        train_df.loc[train_df["delivered"] == 1, "service_ms"].to_numpy(dtype=np.float64),
        utility_payload_bytes,
    )
    print(f"  Utility scale (q95 log1p goodput): {utility_scale:.4f}", flush=True)

    train_pdr = float(train_delivered.mean())
    print(f"  Train delivery rate: {train_pdr:.3f}", flush=True)
    print("  Train PDR / median service_ms by action MCS:", flush=True)
    by_mcs = train_df.groupby("mcs_index").agg(
        pdr=("delivered", "mean"), med_ms=("service_ms", "median"), n=("delivered", "size")
    )
    for mcs_idx, row in by_mcs.iterrows():
        print(
            f"    MCS{int(mcs_idx)}: pdr={row.pdr:.3f} median_service={row.med_ms:.3f}ms n={int(row.n)}",
            flush=True,
        )

    loader_seed = int(seed) + 271_828
    loader_generator = torch.Generator()
    loader_generator.manual_seed(loader_seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_states),
            torch.from_numpy(action_features(train_actions)),
            torch.from_numpy(train_delivered),
            torch.from_numpy(train_log_delay),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    provenance["data_loader_seed"] = loader_seed

    model = BanditOutcomeNetwork(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss()

    eval_state_tensor = torch.from_numpy(eval_states).float().to(device)
    eval_action_tensor = torch.from_numpy(action_features(eval_actions)).float().to(device)
    eval_delivered_tensor = torch.from_numpy(eval_delivered).float().to(device)
    eval_log_delay_tensor = torch.from_numpy(eval_log_delay).float().to(device)
    eval_reward = (
        pd.to_numeric(eval_df.get("reward", pd.Series(np.nan, index=eval_df.index)), errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    eval_service = pd.to_numeric(eval_df["service_ms"], errors="coerce").to_numpy(dtype=np.float64)

    utility_params = {
        "objective": objective,
        "payload_bytes": utility_payload_bytes,
        "loss_reward": utility_loss_reward,
        "utility_scale": utility_scale,
        "tail_target_ms": utility_tail_target_ms,
        "tail_weight": utility_tail_weight,
    }

    log = {
        "epochs": [],
        "train_losses": [],
        "eval_losses": [],
        "eval_macro_run_loss": [],
        "eval_bce": [],
        "eval_delay_huber": [],
        "eval_delivery_accuracy": [],
        "eval_delivery_brier": [],
        "replay_matched_reward": [],
        "replay_matched_service_ms": [],
        "replay_coverage": [],
        "eval_metrics": [],
        "optimizer_steps_at_epoch_end": [],
        "examples_seen_at_epoch_end": [],
        "external_eval_dataset": eval_dataset is not None,
        "provenance": provenance,
        "early_stopping": {
            "enabled": early_stopping_patience is not None,
            "patience": early_stopping_patience,
            "min_epochs": int(early_stopping_min_epochs),
            "min_delta": float(early_stopping_min_delta),
            "monitor": checkpoint_metric,
            "stopped_early": False,
            "stop_epoch": None,
        },
    }
    best_metric = float("inf")
    best_epoch = -1
    best_state = None
    policy_eval_every = max(int(policy_eval_every), 1)
    optimizer_steps = 0
    examples_seen = 0
    early_stop_best = float("inf")
    early_stop_bad_epochs = 0

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        for state_b, action_b, delivered_b, log_delay_b in train_loader:
            state_b = state_b.float().to(device)
            action_b = action_b.float().to(device)
            delivered_b = delivered_b.float().to(device)
            log_delay_b = log_delay_b.float().to(device)

            logit, mu = model(state_b, action_b)
            loss = bce(logit, delivered_b)
            mask = delivered_b > 0.5
            if mask.any():
                loss = loss + delay_loss_weight * huber(mu[mask], log_delay_b[mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            optimizer_steps += 1
            examples_seen += int(state_b.shape[0])

        model.eval()
        with torch.no_grad():
            eval_logit, eval_mu = model(eval_state_tensor, eval_action_tensor)
            eval_prob = torch.sigmoid(eval_logit)
            delivery_accuracy = float(
                ((eval_prob > 0.5).float() == eval_delivered_tensor).float().mean().item()
            )
            delivery_brier = float(
                torch.mean(torch.square(eval_prob - eval_delivered_tensor)).item()
            )
        metrics = outcome_metrics(
            eval_logit.cpu().numpy(),
            eval_mu.cpu().numpy(),
            eval_delivered,
            eval_log_delay,
            delay_loss_weight=delay_loss_weight,
            groups=eval_groups,
        )
        eval_loss = float(metrics["loss"])
        eval_macro_loss = float(metrics["macro_run_loss"])
        eval_bce = float(metrics["bce"])
        eval_delay = float(metrics["delay_huber"])

        replay_reward = float("nan")
        replay_service = float("nan")
        coverage = float("nan")
        if len(eval_df) and ((epoch + 1) % policy_eval_every == 0 or epoch == epochs - 1):
            p_all, mu_all = score_all_actions(model, eval_states, device)
            utilities = expected_utility(p_all, mu_all, **utility_params)
            policy_actions = np.argmax(utilities, axis=1)
            matches = policy_actions == eval_actions
            coverage = float(matches.mean())
            # This is deliberately called replay agreement, not SNIPS: the
            # matched v3.3 TRAIN/VALIDATION data contains a small observational
            # supplement whose state-dependent propensity is unknown.
            if matches.any():
                if np.isfinite(eval_reward[matches]).any():
                    replay_reward = float(np.nanmean(eval_reward[matches]))
                replay_service = float(np.nanmean(eval_service[matches]))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        log["epochs"].append(epoch)
        log["train_losses"].append(train_loss)
        log["eval_losses"].append(eval_loss)
        log["eval_macro_run_loss"].append(eval_macro_loss)
        log["eval_bce"].append(eval_bce)
        log["eval_delay_huber"].append(eval_delay)
        log["eval_delivery_accuracy"].append(delivery_accuracy)
        log["eval_delivery_brier"].append(delivery_brier)
        log["replay_matched_reward"].append(replay_reward)
        log["replay_matched_service_ms"].append(replay_service)
        log["replay_coverage"].append(coverage)
        log["eval_metrics"].append(metrics)
        log["optimizer_steps_at_epoch_end"].append(optimizer_steps)
        log["examples_seen_at_epoch_end"].append(examples_seen)

        metric_value = (
            eval_macro_loss if checkpoint_metric == "eval_macro_run_loss" else eval_loss
        )
        improved = np.isfinite(metric_value) and metric_value < best_metric
        if improved:
            best_metric = metric_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if metric_value < early_stop_best - early_stopping_min_delta:
            early_stop_best = metric_value
            early_stop_bad_epochs = 0
        else:
            early_stop_bad_epochs += 1

        print(
            f"  Epoch {epoch + 1}/{epochs}, train_loss={train_loss:.4f}, "
            f"eval_bce={eval_bce:.4f}, eval_delay_huber={eval_delay:.4f}, "
            f"eval_loss={eval_loss:.4f}, eval_macro_run_loss={eval_macro_loss:.4f}, "
            f"delivery_acc={delivery_accuracy:.4f}, replay_reward={replay_reward:.4f}, "
            f"replay_service_ms={replay_service:.4f}, coverage={coverage:.4f}, "
            f"time={time.perf_counter() - epoch_start:.1f}s",
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
                f"  Early stopping at epoch {epoch + 1}: no {checkpoint_metric} "
                f"improvement larger than {early_stopping_min_delta:g} for "
                f"{early_stop_bad_epochs} monitored epochs",
                flush=True,
            )
            break

    if best_state is None:
        raise FloatingPointError("No finite validation checkpoint was produced")
    model.load_state_dict(best_state)
    print(
        f"  Restored best checkpoint from epoch {best_epoch + 1} "
        f"({checkpoint_metric}={best_metric:.4f})",
        flush=True,
    )

    with torch.no_grad():
        best_eval_logit, best_eval_mu = model(eval_state_tensor, eval_action_tensor)
    best_eval_metrics = outcome_metrics(
        best_eval_logit.cpu().numpy(),
        best_eval_mu.cpu().numpy(),
        eval_delivered,
        eval_log_delay,
        delay_loss_weight=delay_loss_weight,
        groups=eval_groups,
    )
    log["best_epoch"] = int(best_epoch)
    log["epochs_completed"] = int(len(log["epochs"]))
    log["best_checkpoint_metric"] = checkpoint_metric
    log["best_checkpoint_metric_value"] = float(best_metric)
    log["best_eval_metrics"] = best_eval_metrics
    log["early_stopping"]["best_value"] = float(early_stop_best)

    output_model = Path(output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    fair_checkpoint = bool(
        exact_fresh_only
        and eval_dataset is not None
        and resolved_schema == "link_v5"
        and not include_state_mcs
        and state_dim == 132
    )
    checkpoint_payload = {
            "model_type": MODEL_TYPE,
            "checkpoint_schema": (
                FAIR_CHECKPOINT_SCHEMA if fair_checkpoint else "bandit_two_head/v1"
            ),
            "objective_contract": (
                FAIR_OBJECTIVE_CONTRACT if fair_checkpoint else "two_head_outcome/v1"
            ),
            "training_exact_fresh_only": bool(exact_fresh_only),
            "model_state": model.state_dict(),
            "state_dim": state_dim,
            "action_dim": ACTION_DIM,
            "action_feature_dim": 9,
            "hidden_dim": hidden_dim,
            "state_schema": resolved_schema,
            "state_context_feature": state_context_feature,
            "include_state_mcs": include_state_mcs,
            "state_feature_names": state_feature_names(resolved_schema, include_state_mcs),
            **feature_contract_metadata(resolved_schema),
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "delay_clip_ms": delay_clip_ms,
            "utility_params": utility_params,
            "utility_scale_source": "training_delivered_global_q95_log1p_goodput",
            "observed_only": observed_only,
            "exact_fresh_only": exact_fresh_only,
            "ignore_state_mcs": ignore_state_mcs,
            "training_log": log,
            "best_epoch": best_epoch,
            "best_checkpoint_metric": checkpoint_metric,
            "best_checkpoint_metric_value": float(best_metric),
            "best_eval_metrics": best_eval_metrics,
            "checkpoint_metric": checkpoint_metric,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "hidden_dim": int(hidden_dim),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "delay_loss_weight": float(delay_loss_weight),
            "seed": int(seed),
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_epochs": int(early_stopping_min_epochs),
            "early_stopping_min_delta": float(early_stopping_min_delta),
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

    checkpoint_artifact = artifact_provenance(output_model)
    provenance["checkpoint_path"] = str(output_model.resolve())
    provenance["checkpoint_sha256"] = checkpoint_artifact["sha256"]
    log_path = output_model.parent / "bandit_model_training_log.json"
    provenance_path = output_model.parent / "training_provenance.json"
    atomic_write_text(log_path, json.dumps(log, indent=2) + "\n")
    atomic_write_text(provenance_path, json.dumps(provenance, indent=2) + "\n")
    print(f"  Model saved to: {output_model}", flush=True)
    print(f"  Training log saved to: {log_path}", flush=True)
    print(f"  Training provenance saved to: {provenance_path}", flush=True)
    print(f"Bandit model training complete in {time.perf_counter() - start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train two-head bandit outcome model")
    parser.add_argument("--dataset", required=True, help="Path to causal dataset CSV")
    parser.add_argument(
        "--eval-dataset",
        default=None,
        help="Independent validation CSV; required for a fair matched checkpoint",
    )
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-split", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--delay-loss-weight", type=float, default=1.0,
        help="Weight of the log-delay Huber term relative to the delivery BCE",
    )
    parser.add_argument(
        "--delay-clip-ms", type=float, default=50.0,
        help="Upper clip for service_ms before the log-delay target",
    )
    parser.add_argument("--utility-payload-bytes", type=int, default=128)
    parser.add_argument("--utility-loss-reward", type=float, default=-1.0)
    parser.add_argument("--utility-tail-target-ms", type=float, default=0.0)
    parser.add_argument("--utility-tail-weight", type=float, default=0.0)
    parser.add_argument(
        "--objective", choices=["utility", "goodput", "delay"], default="utility",
        help="Objective used for replay diagnostics and the deployed policy",
    )
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument(
        "--checkpoint-metric",
        choices=["eval_loss", "eval_macro_run_loss"],
        default="eval_loss",
    )
    parser.add_argument("--policy-eval-every", type=int, default=1)
    parser.add_argument(
        "--state-schema",
        choices=["auto", "legacy_v1", "link_v2", "link_v3c", "link_v4", "link_v5", "link_v6", "link_v7c"],
        default="link_v3c",
        help=(
            "link_v3c (default) uses compact amplitudes; link_v7c uses the "
            "versioned robust full-CSI contract"
        ),
    )
    parser.add_argument("--ignore-state-mcs", action="store_true")
    parser.add_argument(
        "--exact-fresh-only",
        action="store_true",
        help=(
            "Require state_age=1, packet_gap=1, missing=0, stale=0 on every "
            "train/eval row and emit the fair deployment contract"
        ),
    )
    parser.add_argument(
        "--eval-group-column",
        default=None,
        help="Validation source/run column for equal-weight macro loss",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--include-synthetic", action="store_true",
        help="Keep synthetic/model-augmented rows (default: observed rows only)",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_bandit_model(
        dataset_csv=args.dataset,
        output_model=args.output,
        eval_dataset=args.eval_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_split=args.eval_split,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        delay_loss_weight=args.delay_loss_weight,
        delay_clip_ms=args.delay_clip_ms,
        utility_payload_bytes=args.utility_payload_bytes,
        utility_loss_reward=args.utility_loss_reward,
        utility_tail_target_ms=args.utility_tail_target_ms,
        utility_tail_weight=args.utility_tail_weight,
        objective=args.objective,
        max_train_rows=args.max_train_rows,
        checkpoint_metric=args.checkpoint_metric,
        policy_eval_every=args.policy_eval_every,
        state_schema=args.state_schema,
        ignore_state_mcs=args.ignore_state_mcs,
        observed_only=not args.include_synthetic,
        exact_fresh_only=args.exact_fresh_only,
        eval_group_column=args.eval_group_column,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        early_stopping_min_delta=args.early_stopping_min_delta,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
