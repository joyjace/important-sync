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
import json
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
    DIFFERENTIAL_PHASE_FEATURE_START as LINK_V7C_DIFFERENTIAL_PHASE_START,
    DIFFERENTIAL_PHASE_FEATURE_STOP as LINK_V7C_DIFFERENTIAL_PHASE_STOP,
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
):
    start = time.perf_counter()
    print("[Action-Conditioned Outcome Model Training]", flush=True)
    print(f"  Dataset: {dataset_csv}", flush=True)
    print(f"  Output: {output_model}", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"  Seed: {seed}", flush=True)
    print(f"  Target column: {target_column} ({objective})", flush=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(dataset_csv)
    print(f"  Rows: {len(df)}", flush=True)
    required = {"mcs_index", target_column, "iq_raw"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    if delivered_only:
        if "delivered" not in df.columns:
            raise ValueError("--delivered-only requires a 'delivered' column in the dataset")
        before = len(df)
        df = df[df["delivered"] == 1].reset_index(drop=True)
        print(f"  --delivered-only: kept {len(df)} / {before} rows ({len(df)/before*100:.1f}%)", flush=True)

    if observed_only:
        before = len(df)
        synthetic = pd.to_numeric(
            df.get("synthetic_stale", pd.Series(0, index=df.index)),
            errors="coerce",
        ).fillna(0)
        model_augmented = pd.to_numeric(
            df.get("model_augmented", pd.Series(0, index=df.index)),
            errors="coerce",
        ).fillna(0)
        df = df[(synthetic == 0) & (model_augmented == 0)].reset_index(drop=True)
        print(f"  --observed-only: kept {len(df)} / {before} rows", flush=True)

    if fresh_state_only:
        if "state_age_packets" not in df.columns:
            raise ValueError("--fresh-state-only requires state_age_packets")
        before = len(df)
        ages = pd.to_numeric(df["state_age_packets"], errors="coerce")
        df = df[ages == 1].reset_index(drop=True)
        print(f"  --fresh-state-only: kept {len(df)} / {before} rows", flush=True)

    if df.empty:
        raise ValueError("No rows remain after reward-model filters")

    df["iq_raw_parsed"] = df["iq_raw"].apply(parse_iq_raw)

    # Shuffle before splitting so all MCS segments appear in both train and eval.
    # A temporal split puts entire MCS segments into eval, making all policy metrics
    # misleading (replay_coverage collapses to 0, policy_score_improvement is a
    # fixed constant rather than a learning signal).
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * (1 - eval_split))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    eval_df = df.iloc[split_idx:].reset_index(drop=True)

    if max_train_rows is not None and max_train_rows < len(train_df):
        train_df = train_df.sample(n=max_train_rows, random_state=seed).reset_index(drop=True)
        print(f"  Sampled train rows: {len(train_df)}", flush=True)

    print(f"  Train rows: {len(train_df)}, Eval rows: {len(eval_df)}", flush=True)

    precompute_start = time.perf_counter()
    train_rows = list(train_df.itertuples(index=False))
    eval_rows = list(eval_df.itertuples(index=False))
    state_context_feature = "state_age_packets" if "state_age_packets" in df.columns else "sig_len"
    include_state_mcs = "state_mcs_index" in df.columns and not ignore_state_mcs
    state_schema = resolve_state_schema_for_columns(df.columns, state_schema)
    validate_dataset_feature_contract(dataset_csv, state_schema)
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
    print(f"  State context feature: {state_context_feature}", flush=True)
    print(f"  State schema: {state_schema}", flush=True)
    print(
        f"  Source MCS conditioning: {'enabled (one-hot)' if include_state_mcs else 'disabled'}",
        flush=True,
    )
    if ignore_state_mcs and "state_mcs_index" in df.columns:
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

    train_actions = train_df["mcs_index"].to_numpy(dtype=np.int64)
    eval_actions = eval_df["mcs_index"].to_numpy(dtype=np.int64)
    train_targets = train_df[target_column].to_numpy(dtype=np.float32)
    eval_targets = eval_df[target_column].to_numpy(dtype=np.float32)

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

    train_ds = TensorDataset(
        torch.from_numpy(train_states),
        torch.from_numpy(train_action_features),
        torch.from_numpy(train_targets),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = ActionRewardNetwork(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss() if loss_type == "huber" else nn.MSELoss()

    log = {
        "epochs": [],
        "train_losses": [],
        "eval_losses": [],
        "replay_target": [],
        "replay_coverage": [],
        "state_schema": state_schema,
        "phase_dropout": phase_dropout,
        "phase_consistency_weight": phase_consistency_weight,
    }
    best_eval_loss = float("inf")
    best_policy_target = float("inf") if objective == "minimize" else -float("inf")
    best_epoch = -1
    best_state = None

    eval_state_tensor = torch.from_numpy(eval_states).float().to(device)
    eval_action_tensor = torch.from_numpy(eval_action_features).float().to(device)
    eval_target_tensor = torch.from_numpy(eval_targets).float().to(device)
    policy_eval_every = max(int(policy_eval_every), 1)

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        for state_batch, action_batch, target_batch in train_loader:
            state_batch = state_batch.float().to(device)
            action_batch = action_batch.float().to(device)
            target_batch = target_batch.float().to(device)

            full_pred = model(state_batch, action_batch)
            loss = criterion(full_pred, target_batch)

            if state_schema == "link_v7c" and phase_dropout > 0.0:
                dropout_mask = (
                    torch.rand((state_batch.shape[0], 1), device=state_batch.device)
                    < phase_dropout
                )
                phase_dropout_state = state_batch.clone()
                phase_dropout_state[
                    :,
                    LINK_V7C_DIFFERENTIAL_PHASE_START:LINK_V7C_DIFFERENTIAL_PHASE_STOP,
                ] = torch.where(
                    dropout_mask,
                    torch.zeros_like(
                        phase_dropout_state[
                            :,
                            LINK_V7C_DIFFERENTIAL_PHASE_START:LINK_V7C_DIFFERENTIAL_PHASE_STOP,
                        ]
                    ),
                    phase_dropout_state[
                        :,
                        LINK_V7C_DIFFERENTIAL_PHASE_START:LINK_V7C_DIFFERENTIAL_PHASE_STOP,
                    ],
                )
                dropout_pred = model(phase_dropout_state, action_batch)
                loss = 0.5 * (loss + criterion(dropout_pred, target_batch))

            if state_schema == "link_v7c" and phase_consistency_weight > 0.0:
                phase_values_masked_state = state_batch.clone()
                phase_values_masked_state[
                    :,
                    LINK_V7C_DIFFERENTIAL_PHASE_START:LINK_V7C_DIFFERENTIAL_PHASE_STOP,
                ] = 0.0
                phase_values_masked_pred = model(phase_values_masked_state, action_batch)
                consistency_loss = nn.functional.mse_loss(
                    full_pred,
                    phase_values_masked_pred,
                )
                loss = loss + phase_consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            eval_pred = model(eval_state_tensor, eval_action_tensor)
            eval_loss = float(criterion(eval_pred, eval_target_tensor).item())

        replay_target = float("nan")
        replay_coverage = float("nan")
        if len(eval_df) and ((epoch + 1) % policy_eval_every == 0 or epoch == epochs - 1):
            all_scores = score_all_actions(model, eval_states, 8, device)
            policy_actions = select_actions(all_scores, objective)
            matches = policy_actions == eval_actions
            # NOTE: replay_coverage measures agreement with the *logged* policy, not policy quality.
            # A correct model that avoids the dominant logged MCS will show ~0% coverage even when
            # learning the right thing. Use policy_score_improvement as the primary quality signal.
            replay_coverage = float(np.mean(matches))
            if np.any(matches):
                replay_target = float(np.mean(eval_targets[matches]))
            # Policy score improvement: mean predicted score of recommended action vs dataset mean.
            # Positive = model thinks its choices are better than average; grows as model learns.
            policy_scores = all_scores[np.arange(len(all_scores)), policy_actions]
            policy_score_improvement = float(np.mean(policy_scores) - np.mean(eval_targets))
        else:
            policy_score_improvement = float("nan")

        train_loss = float(np.mean(losses)) if losses else float("nan")
        log["epochs"].append(epoch)
        log["train_losses"].append(train_loss)
        log["eval_losses"].append(eval_loss)
        log["replay_target"].append(replay_target)
        log["replay_coverage"].append(replay_coverage)
        log.setdefault("policy_score_improvement", []).append(policy_score_improvement)

        improved = False
        if checkpoint_metric == "replay_target" and not np.isnan(replay_target):
            if objective == "minimize":
                improved = replay_target < best_policy_target
            else:
                improved = replay_target > best_policy_target
        else:
            improved = eval_loss < best_eval_loss

        if improved:
            best_eval_loss = eval_loss
            if not np.isnan(replay_target):
                best_policy_target = replay_target
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"  Epoch {epoch + 1}/{epochs}, train_loss={train_loss:.4f}, "
            f"eval_loss={eval_loss:.4f}, replay_target={replay_target:.4f}, "
            f"policy_score_improvement={policy_score_improvement:.4f}, "
            f"replay_coverage={replay_coverage:.4f}, time={time.perf_counter() - epoch_start:.1f}s",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"  Restored best checkpoint from epoch {best_epoch + 1} "
            f"(eval_loss={best_eval_loss:.4f}, replay_target={best_policy_target:.4f}, "
            f"checkpoint_metric={checkpoint_metric})",
            flush=True,
        )

    output_model = Path(output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
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
            "best_eval_loss": best_eval_loss,
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
        },
        output_model,
    )
    log_path = output_model.parent / "reward_model_training_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"  Model saved to: {output_model}", flush=True)
    print(f"  Training log saved to: {log_path}", flush=True)
    print(f"✅ Outcome model training complete in {time.perf_counter() - start:.1f}s!", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train supervised action-conditioned outcome model")
    parser.add_argument("--dataset", required=True, help="Path to DQN dataset CSV")
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
        choices=["eval_loss", "replay_target"],
        default="eval_loss",
        help="Which metric selects the saved checkpoint",
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
            "For link_v7c, randomly replace the 108 differential-phase values "
            "with their normalized mean during training. This anchors the model "
            "to the amplitude/link fallback when phase is unreliable."
        ),
    )
    parser.add_argument(
        "--phase-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "For link_v7c, penalize score changes when differential phase is "
            "masked. Use with phase dropout to limit spurious phase dependence."
        ),
    )
    args = parser.parse_args()

    train_reward_model(
        dataset_csv=args.dataset,
        output_model=args.output,
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
    )


if __name__ == "__main__":
    main()
