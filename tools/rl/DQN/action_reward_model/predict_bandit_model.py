#!/usr/bin/env python3
"""
Score all MCS actions with a two-head bandit checkpoint and derive a policy.

Writes a prediction CSV compatible with shared/evaluate_policy_delay.py
(predicted_mcs, actual_mcs, service_ms, actual_reward, delivered), with the
per-action delivery probabilities, predicted delays, and utilities included
for diagnostics.

Usage:
    python predict_bandit_model.py \
        --model models/bandit_model/bandit_model.pth \
        --dataset datasets/rl_dqn_dataset_holdout_utility.csv \
        --output predictions/bandit_recommendations_holdout.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DQN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DQN_ROOT / "dqn_model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dqn import (
    build_state_vector,
    feature_contract_metadata,
    validate_dataset_feature_contract,
)
from train_reward_model import normalize, parse_iq_raw
from train_bandit_model import ACTION_DIM, MODEL_TYPE, BanditOutcomeNetwork, score_all_actions
from policy_utils import expected_utility


def load_bandit_checkpoint(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"Checkpoint is not a {MODEL_TYPE} model: {checkpoint.get('model_type')}"
        )
    # Bandit checkpoints trained before the schema rename stored the compact
    # 57-amplitude layout as "link_v3"; that name now belongs to the DQN
    # link_v3..link_v6 series, so map it to its new name "link_v3c".
    if checkpoint.get("state_schema") == "link_v3":
        checkpoint["state_schema"] = "link_v3c"
    for key, expected in feature_contract_metadata(checkpoint.get("state_schema")).items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"Checkpoint {key}={checkpoint.get(key)!r} does not match {expected!r}"
            )
    model = BanditOutcomeNetwork(
        state_dim=int(checkpoint["state_dim"]),
        action_feature_dim=int(checkpoint.get("action_feature_dim", 9)),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Bandit model MCS recommendations")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--objective", choices=["checkpoint", "utility", "goodput", "delay"],
        default="checkpoint",
        help="Policy objective; 'checkpoint' reuses the training-time settings",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    print("[Bandit Model Prediction]")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")

    model, checkpoint = load_bandit_checkpoint(args.model, args.device)
    utility_params = dict(checkpoint["utility_params"])
    if args.objective != "checkpoint":
        utility_params["objective"] = args.objective
    print(f"  Objective: {utility_params['objective']}")

    validate_dataset_feature_contract(args.dataset, checkpoint["state_schema"])
    df = pd.read_csv(args.dataset)
    df["iq_raw_parsed"] = df["iq_raw"].apply(parse_iq_raw)
    print(f"  Rows: {len(df)}")

    states = np.array(
        [
            build_state_vector(
                row,
                checkpoint["state_context_feature"],
                checkpoint["include_state_mcs"],
                checkpoint["state_schema"],
            )
            for row in df.itertuples(index=False)
        ],
        dtype=np.float32,
    )
    if states.shape[1] != int(checkpoint["state_dim"]):
        raise ValueError(
            f"Dataset produced state dim {states.shape[1]}, "
            f"checkpoint expects {checkpoint['state_dim']}"
        )
    state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
    state_std = np.asarray(checkpoint["state_std"], dtype=np.float32)
    states = normalize(states, state_mean, state_std).astype(np.float32)

    p_all, mu_all = score_all_actions(model, states, args.device, batch_size=args.batch_size)
    utilities = expected_utility(p_all, mu_all, **utility_params)
    predicted = np.argmax(utilities, axis=1)
    delay_ms_all = np.exp(mu_all)

    out = pd.DataFrame(
        {
            "seq": df.get("seq", pd.Series(range(len(df)))),
            "actual_mcs": df["mcs_index"],
            "predicted_mcs": predicted,
            "predicted_utility": utilities[np.arange(len(df)), predicted],
            "actual_reward": df.get("reward", pd.Series(np.nan, index=df.index)),
            "service_ms": df.get("service_ms", pd.Series(np.nan, index=df.index)),
            "delivered": df.get("delivered", pd.Series(np.nan, index=df.index)),
            "rssi": df.get("rssi", pd.Series(np.nan, index=df.index)),
            "snr": df.get("snr", pd.Series(np.nan, index=df.index)),
            "state_mcs_index": df.get("state_mcs_index", pd.Series(np.nan, index=df.index)),
            "state_age_packets": df.get("state_age_packets", pd.Series(np.nan, index=df.index)),
            "state_is_stale": df.get("state_is_stale", pd.Series(np.nan, index=df.index)),
        }
    )
    for action in range(ACTION_DIM):
        out[f"p_deliver_mcs{action}"] = p_all[:, action]
        out[f"pred_delay_ms_mcs{action}"] = delay_ms_all[:, action]
        out[f"utility_mcs{action}"] = utilities[:, action]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    counts = pd.Series(predicted).value_counts().sort_index()
    print("  Recommended MCS distribution:")
    for mcs_idx, count in counts.items():
        print(f"    MCS{int(mcs_idx)}: {int(count)} ({count / len(out) * 100:.1f}%)")
    print("  Mean per-action predictions:")
    for action in range(ACTION_DIM):
        print(
            f"    MCS{action}: p_deliver={p_all[:, action].mean():.3f} "
            f"delay={delay_ms_all[:, action].mean():.3f}ms "
            f"utility={utilities[:, action].mean():.4f}"
        )
    print(f"  Predictions written to: {args.output}")


if __name__ == "__main__":
    main()
