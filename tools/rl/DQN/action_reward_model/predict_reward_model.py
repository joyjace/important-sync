#!/usr/bin/env python3
"""
Predict MCS recommendations using a supervised action-conditioned outcome model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DQN_ROOT = Path(__file__).resolve().parents[1]
DQN_MODEL_DIR = DQN_ROOT / "dqn_model"
sys.path.insert(0, str(DQN_MODEL_DIR))

from predict_dqn import mcs_to_symbol
from train_dqn import (
    build_state_vector,
    feature_contract_metadata,
    validate_dataset_feature_contract,
)
from train_reward_model import ActionRewardNetwork, action_features, parse_iq_raw
from csi_link_v7c import (
    PHASE_DERIVED_FEATURE_START as LINK_V7C_PHASE_DERIVED_START,
    PHASE_DERIVED_FEATURE_STOP as LINK_V7C_PHASE_DERIVED_STOP,
)


class RewardModel:
    def __init__(self, checkpoint, device="cpu"):
        self.device = device
        self.state_dim = int(checkpoint["state_dim"])
        self.action_dim = int(checkpoint["action_dim"])
        self.action_feature_dim = int(checkpoint.get("action_feature_dim", self.action_dim))
        self.hidden_dim = int(checkpoint.get("hidden_dim", 128))
        self.state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(checkpoint["state_std"], dtype=np.float32)
        self.state_std = np.where(self.state_std < 1e-6, 1.0, self.state_std)
        self.target_column = str(checkpoint.get("target_column", "reward"))
        self.objective = str(checkpoint.get("objective", "maximize"))
        self.state_context_feature = str(checkpoint.get("state_context_feature", "sig_len"))
        self.state_schema = str(checkpoint.get("state_schema", "legacy_v1"))
        for key, expected in feature_contract_metadata(self.state_schema).items():
            if checkpoint.get(key) != expected:
                raise ValueError(
                    f"Checkpoint {key}={checkpoint.get(key)!r} does not match {expected!r}"
                )
        self.include_state_mcs = bool(
            checkpoint.get("include_state_mcs", self.state_dim in {136, 140, 143, 262})
        )
        self.net = ActionRewardNetwork(
            state_dim=self.state_dim,
            action_feature_dim=self.action_feature_dim,
            hidden_dim=self.hidden_dim,
        ).to(device)
        self.net.load_state_dict(checkpoint["model_state"])
        self.net.eval()

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        return cls(checkpoint, device=device)

    def normalize_states(self, states: np.ndarray) -> np.ndarray:
        return (states - self.state_mean) / self.state_std

    def score_all_actions(self, states: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        states = self.normalize_states(states).astype(np.float32)
        all_scores = []

        with torch.no_grad():
            for start in range(0, len(states), batch_size):
                state_batch = states[start : start + batch_size]
                repeated_states = np.repeat(state_batch, self.action_dim, axis=0)
                actions = np.tile(np.arange(self.action_dim, dtype=np.int64), len(state_batch))
                action_feats = action_features(actions, self.action_dim)

                state_tensor = torch.from_numpy(repeated_states).float().to(self.device)
                action_tensor = torch.from_numpy(action_feats).float().to(self.device)
                scores = self.net(state_tensor, action_tensor).cpu().numpy()
                all_scores.append(scores.reshape(len(state_batch), self.action_dim))

        return np.vstack(all_scores)


def safe_int(value, default=-1):
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value, default=float("nan")):
    try:
        return float(value)
    except Exception:
        return default


def select_best_actions(scores: np.ndarray, objective: str) -> np.ndarray:
    if objective == "minimize":
        return np.argmin(scores, axis=1)
    return np.argmax(scores, axis=1)


def sort_action_indices(scores: np.ndarray, objective: str) -> np.ndarray:
    if objective == "minimize":
        return np.argsort(scores)
    return np.argsort(scores)[::-1]


def predict_reward_model(
    model_path,
    dataset_csv,
    output_csv,
    device="cpu",
    batch_size=4096,
    delivered_only=False,
    observed_only=False,
    fresh_state_only=False,
    low_snr_threshold_db=None,
    low_snr_max_mcs=None,
    mask_v7c_phase=False,
):
    print("[Action-Conditioned Outcome Model Prediction]")
    print(f"  Loading model from: {model_path}")
    model = RewardModel.load(model_path, device=device)
    print("  Model loaded successfully")
    print(f"  Model target: {model.target_column} ({model.objective})")

    print(f"  Loading dataset from: {dataset_csv}")
    validate_dataset_feature_contract(dataset_csv, model.state_schema)
    df = pd.read_csv(dataset_csv)
    print(f"  Dataset rows: {len(df)}")

    if delivered_only:
        if "delivered" not in df.columns:
            raise ValueError("--delivered-only requires a 'delivered' column in the dataset")
        before = len(df)
        df = df[df["delivered"] == 1].reset_index(drop=True)
        print(f"  --delivered-only: kept {len(df)} / {before} rows ({len(df)/before*100:.1f}%)")

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
        print(f"  --observed-only: kept {len(df)} / {before} rows")

    if fresh_state_only:
        if "state_age_packets" not in df.columns:
            raise ValueError("--fresh-state-only requires state_age_packets")
        before = len(df)
        ages = pd.to_numeric(df["state_age_packets"], errors="coerce")
        df = df[ages == 1].reset_index(drop=True)
        print(f"  --fresh-state-only: kept {len(df)} / {before} rows")

    if df.empty:
        raise ValueError("No rows remain after prediction filters")

    dataset_is_causal = "state_age_packets" in df.columns
    model_is_causal = model.state_context_feature == "state_age_packets"
    if dataset_is_causal != model_is_causal:
        model_kind = "causal" if model_is_causal else "legacy same-packet"
        dataset_kind = "causal" if dataset_is_causal else "legacy same-packet"
        raise ValueError(
            f"Model expects a {model_kind} dataset, but received a {dataset_kind} dataset"
        )
    if model.include_state_mcs and "state_mcs_index" not in df.columns:
        raise ValueError(
            "Model was trained with source-MCS conditioning, but the dataset "
            "does not contain state_mcs_index"
        )

    df["iq_raw_parsed"] = df["iq_raw"].apply(parse_iq_raw)
    rows = list(df.itertuples(index=False))
    states = np.array(
        [
            build_state_vector(
                row,
                model.state_context_feature,
                model.include_state_mcs,
                model.state_schema,
            )
            for row in rows
        ],
        dtype=np.float32,
    )
    if states.shape[1] != model.state_dim:
        raise ValueError(
            f"Model expects {model.state_dim} state features, but dataset produced {states.shape[1]}"
        )
    if mask_v7c_phase:
        if model.state_schema != "link_v7c":
            raise ValueError("--mask-v7c-phase requires a link_v7c checkpoint")
        # Replace every phase-derived feature, including valid fraction and
        # coherence, with its checkpoint mean.  This makes normalized indices
        # 56:166 exactly zero and is a true phase-vs-amplitude ablation.  The
        # training regularizer intentionally masks only the 108 differential
        # values (56:164), so this qualification path is deliberately stricter.
        phase_slice = slice(
            LINK_V7C_PHASE_DERIVED_START,
            LINK_V7C_PHASE_DERIVED_STOP,
        )
        states[:, phase_slice] = model.state_mean[phase_slice]
        print(
            "  Phase ablation: masked 108 differential-phase values and "
            "2 phase-quality values"
        )

    print("  Scoring all MCS actions...")
    scores = model.score_all_actions(states, batch_size=batch_size)
    raw_best_actions = select_best_actions(scores, model.objective)
    best_actions = raw_best_actions.copy()
    guard_applied = np.zeros(len(best_actions), dtype=bool)
    if (low_snr_threshold_db is None) != (low_snr_max_mcs is None):
        raise ValueError(
            "--low-snr-threshold-db and --low-snr-max-mcs must be provided together"
        )
    if low_snr_threshold_db is not None:
        if model.state_schema != "link_v7c":
            raise ValueError("The low-SNR deployment guard is defined only for link_v7c")
        if not 0 <= int(low_snr_max_mcs) <= 7:
            raise ValueError("--low-snr-max-mcs must be in [0, 7]")
        snr_values = pd.to_numeric(df["snr"], errors="coerce").to_numpy(dtype=np.float32)
        guard_applied = (snr_values < float(low_snr_threshold_db)) & (
            best_actions > int(low_snr_max_mcs)
        )
        best_actions[guard_applied] = int(low_snr_max_mcs)
        print(
            f"  Low-SNR guard: SNR < {low_snr_threshold_db:g} dB caps at "
            f"MCS{int(low_snr_max_mcs)}; changed {int(guard_applied.sum())} rows"
        )
    best_scores = scores[np.arange(len(scores)), best_actions]

    predictions = []
    for index, row in enumerate(rows):
        values = scores[index]
        ranking = sort_action_indices(values, model.objective)
        top3_indices = ranking[:3]
        predictions.append({
            "seq": safe_int(getattr(row, "seq", index), index),
            "actual_mcs": safe_int(getattr(row, "mcs_index", -1), -1),
            "predicted_mcs": int(best_actions[index]),
            "raw_predicted_mcs": int(raw_best_actions[index]),
            "safety_guard_applied": int(guard_applied[index]),
            "phase_masked": int(mask_v7c_phase),
            "predicted_value": float(best_scores[index]),
            "predicted_target_column": model.target_column,
            "predicted_objective": model.objective,
            **{f"score_mcs{action}": float(values[action]) for action in range(model.action_dim)},
            "top3_mcs": [int(action) for action in top3_indices],
            "top3_values": [float(values[action]) for action in top3_indices],
            "actual_reward": safe_float(getattr(row, "reward", float("nan"))),
            "service_ms": safe_float(getattr(row, "service_ms", float("nan"))),
            "rssi": safe_float(getattr(row, "rssi", 0.0), 0.0),
            "snr": safe_float(getattr(row, "snr", 0.0), 0.0),
            "delivered": safe_int(getattr(row, "delivered", -1), -1),
        })

    pred_df = pd.DataFrame(predictions)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_csv, index=False)
    print(f"  Predictions saved to: {output_csv}")

    correct_preds = (pred_df["predicted_mcs"] == pred_df["actual_mcs"]).sum()
    accuracy = correct_preds / len(pred_df) if len(pred_df) else float("nan")

    print("  ✅ Prediction complete:")
    print(f"    Total predictions: {len(pred_df)}")
    print(f"    Matches actual MCS: {correct_preds} ({accuracy * 100:.2f}%)")
    print(f"    Mean predicted value: {pred_df['predicted_value'].mean():.4f}")
    print("    Note: actual_reward and service_ms are for the logged actual_mcs, not the predicted_mcs")

    if "actual_mcs" in pred_df.columns and len(pred_df):
        print("  Empirical logged baseline by actual MCS:")
        empirical = pred_df.groupby("actual_mcs").agg(
            n=("actual_mcs", "size"),
            mean_reward=("actual_reward", "mean"),
            mean_service=("service_ms", "mean"),
            median_service=("service_ms", "median"),
        ).sort_index()
        for mcs_idx, stats in empirical.iterrows():
            print(
                f"    MCS{int(mcs_idx)}: n={int(stats['n'])}, "
                f"mean_reward={float(stats['mean_reward']):.4f}, "
                f"mean_service={float(stats['mean_service']):.4f} ms, "
                f"median_service={float(stats['median_service']):.4f} ms"
            )

    print("  Top MCS selections by frequency:")
    mcs_counts = pred_df["predicted_mcs"].value_counts().sort_index()
    for mcs_idx, count in mcs_counts.items():
        pct = count / len(pred_df) * 100 if len(pred_df) else 0.0
        print(f"    MCS{int(mcs_idx)} ({mcs_to_symbol(int(mcs_idx))}): {int(count)} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Predict MCS using action-conditioned outcome model")
    parser.add_argument("--model", required=True, help="Path to trained reward model")
    parser.add_argument("--dataset", required=True, help="Path to DQN dataset CSV")
    parser.add_argument("--output", required=True, help="Path to save recommendations")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096, help="Inference batch size")
    parser.add_argument(
        "--delivered-only",
        action="store_true",
        default=False,
        help="Evaluate only delivered==1 packets",
    )
    parser.add_argument(
        "--observed-only",
        action="store_true",
        default=False,
        help="Exclude synthetic/model-augmented rows before prediction",
    )
    parser.add_argument(
        "--fresh-state-only",
        action="store_true",
        default=False,
        help="Evaluate only rows where state_age_packets == 1",
    )
    parser.add_argument(
        "--low-snr-threshold-db",
        type=float,
        default=None,
        help="Mirror the live v7c guard: cap actions below this SNR",
    )
    parser.add_argument(
        "--low-snr-max-mcs",
        type=int,
        default=None,
        help="Maximum MCS below --low-snr-threshold-db",
    )
    parser.add_argument(
        "--mask-v7c-phase",
        action="store_true",
        help="Ablation: replace all 110 v7c phase-derived values with their training mean",
    )
    args = parser.parse_args()

    predict_reward_model(
        model_path=args.model,
        dataset_csv=args.dataset,
        output_csv=args.output,
        device=args.device,
        batch_size=args.batch_size,
        delivered_only=args.delivered_only,
        observed_only=args.observed_only,
        fresh_state_only=args.fresh_state_only,
        low_snr_threshold_db=args.low_snr_threshold_db,
        low_snr_max_mcs=args.low_snr_max_mcs,
        mask_v7c_phase=args.mask_v7c_phase,
    )


if __name__ == "__main__":
    main()
