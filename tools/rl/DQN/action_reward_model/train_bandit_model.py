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
1/8), where counterfactual action scores are supported by the data. On such
datasets the per-epoch SNIPS diagnostics are unbiased policy estimates.

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
from train_reward_model import action_features, normalize, parse_iq_raw
from policy_utils import expected_utility, utility_scale_from_delivered_service

ACTION_DIM = 8
MODEL_TYPE = "bandit_two_head_v1"


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
    device="cpu",
    seed=42,
):
    start = time.perf_counter()
    print("[Two-Head Bandit Outcome Model Training]", flush=True)
    print(f"  Dataset: {dataset_csv}", flush=True)
    print(f"  Output: {output_model}", flush=True)
    print(f"  Device: {device}, Seed: {seed}", flush=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(dataset_csv)
    print(f"  Rows: {len(df)}", flush=True)
    required = {"mcs_index", "delivered", "service_ms", "iq_raw"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    if observed_only:
        before = len(df)
        synthetic = pd.to_numeric(
            df.get("synthetic_stale", pd.Series(0, index=df.index)), errors="coerce"
        ).fillna(0)
        model_augmented = pd.to_numeric(
            df.get("model_augmented", pd.Series(0, index=df.index)), errors="coerce"
        ).fillna(0)
        df = df[(synthetic == 0) & (model_augmented == 0)].reset_index(drop=True)
        print(f"  Observed-only: kept {len(df)} / {before} rows", flush=True)

    delivered_col = pd.to_numeric(df["delivered"], errors="coerce")
    df = df[delivered_col.isin([0, 1])].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows remain after filters")

    df["iq_raw_parsed"] = df["iq_raw"].apply(parse_iq_raw)

    # Shuffled row split. Adjacent packets are correlated, so this slightly
    # flatters eval metrics; scenario-level holdouts remain the honest test.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * (1 - eval_split))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    eval_df = df.iloc[split_idx:].reset_index(drop=True)
    if max_train_rows is not None and max_train_rows < len(train_df):
        train_df = train_df.sample(n=max_train_rows, random_state=seed).reset_index(drop=True)
    print(f"  Train rows: {len(train_df)}, Eval rows: {len(eval_df)}", flush=True)

    state_context_feature = "state_age_packets" if "state_age_packets" in df.columns else "sig_len"
    include_state_mcs = "state_mcs_index" in df.columns and not ignore_state_mcs
    resolved_schema = resolve_state_schema_for_columns(df.columns, state_schema)
    validate_dataset_feature_contract(dataset_csv, resolved_schema)
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

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_states),
            torch.from_numpy(action_features(train_actions)),
            torch.from_numpy(train_delivered),
            torch.from_numpy(train_log_delay),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

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
        "eval_bce": [],
        "eval_delay_huber": [],
        "eval_delivery_accuracy": [],
        "snips_reward": [],
        "snips_service_ms": [],
        "policy_coverage": [],
    }
    best_metric = float("inf") if checkpoint_metric == "eval_loss" else -float("inf")
    best_epoch = -1
    best_state = None
    policy_eval_every = max(int(policy_eval_every), 1)

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

        model.eval()
        with torch.no_grad():
            eval_logit, eval_mu = model(eval_state_tensor, eval_action_tensor)
            eval_bce = float(bce(eval_logit, eval_delivered_tensor).item())
            eval_mask = eval_delivered_tensor > 0.5
            eval_delay = (
                float(huber(eval_mu[eval_mask], eval_log_delay_tensor[eval_mask]).item())
                if bool(eval_mask.any())
                else float("nan")
            )
            eval_prob = torch.sigmoid(eval_logit)
            delivery_accuracy = float(
                ((eval_prob > 0.5).float() == eval_delivered_tensor).float().mean().item()
            )
        eval_loss = eval_bce + (
            delay_loss_weight * eval_delay if not np.isnan(eval_delay) else 0.0
        )

        snips_reward = float("nan")
        snips_service = float("nan")
        coverage = float("nan")
        if len(eval_df) and ((epoch + 1) % policy_eval_every == 0 or epoch == epochs - 1):
            p_all, mu_all = score_all_actions(model, eval_states, device)
            utilities = expected_utility(p_all, mu_all, **utility_params)
            policy_actions = np.argmax(utilities, axis=1)
            matches = policy_actions == eval_actions
            coverage = float(matches.mean())
            # With uniform 1/8 logging propensity, SNIPS reduces to the mean
            # outcome over matched rows (weights cancel).
            if matches.any():
                if np.isfinite(eval_reward[matches]).any():
                    snips_reward = float(np.nanmean(eval_reward[matches]))
                snips_service = float(np.nanmean(eval_service[matches]))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        log["epochs"].append(epoch)
        log["train_losses"].append(train_loss)
        log["eval_losses"].append(eval_loss)
        log["eval_bce"].append(eval_bce)
        log["eval_delay_huber"].append(eval_delay)
        log["eval_delivery_accuracy"].append(delivery_accuracy)
        log["snips_reward"].append(snips_reward)
        log["snips_service_ms"].append(snips_service)
        log["policy_coverage"].append(coverage)

        if checkpoint_metric == "eval_loss":
            improved = eval_loss < best_metric
            metric_value = eval_loss
        else:  # snips_reward
            improved = not np.isnan(snips_reward) and snips_reward > best_metric
            metric_value = snips_reward
        if improved:
            best_metric = metric_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"  Epoch {epoch + 1}/{epochs}, train_loss={train_loss:.4f}, "
            f"eval_bce={eval_bce:.4f}, eval_delay_huber={eval_delay:.4f}, "
            f"delivery_acc={delivery_accuracy:.4f}, snips_reward={snips_reward:.4f}, "
            f"snips_service_ms={snips_service:.4f}, coverage={coverage:.4f}, "
            f"time={time.perf_counter() - epoch_start:.1f}s",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"  Restored best checkpoint from epoch {best_epoch + 1} "
            f"({checkpoint_metric}={best_metric:.4f})",
            flush=True,
        )

    output_model = Path(output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": MODEL_TYPE,
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
            "observed_only": observed_only,
            "ignore_state_mcs": ignore_state_mcs,
            "training_log": log,
            "best_epoch": best_epoch,
            "checkpoint_metric": checkpoint_metric,
        },
        output_model,
    )
    log_path = output_model.parent / "bandit_model_training_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"  Model saved to: {output_model}", flush=True)
    print(f"  Training log saved to: {log_path}", flush=True)
    print(f"Bandit model training complete in {time.perf_counter() - start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train two-head bandit outcome model")
    parser.add_argument("--dataset", required=True, help="Path to causal dataset CSV")
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
        help="Objective used for the per-epoch SNIPS policy diagnostics",
    )
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument(
        "--checkpoint-metric", choices=["eval_loss", "snips_reward"], default="eval_loss"
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
        "--include-synthetic", action="store_true",
        help="Keep synthetic/model-augmented rows (default: observed rows only)",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_bandit_model(
        dataset_csv=args.dataset,
        output_model=args.output,
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
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
