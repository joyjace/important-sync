#!/usr/bin/env python3
"""
Use trained DQN model to predict best MCS for each packet in dataset.

Usage:
    python predict_dqn.py --model dqn_mcs_model.pth \\
                          --dataset rl_dqn_dataset.csv \\
                          --output dqn_mcs_recommendations.csv \\
                          --device cpu
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Import DQN classes from train_dqn
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_dqn import DQNAgent, build_state_vector, validate_dataset_feature_contract


def mcs_to_symbol(mcs_idx):
    """Convert MCS index to WiFi rate symbol."""
    symbols = [
        "WIFI_PHY_RATE_MCS0_LGI",
        "WIFI_PHY_RATE_MCS1_LGI",
        "WIFI_PHY_RATE_MCS2_LGI",
        "WIFI_PHY_RATE_MCS3_LGI",
        "WIFI_PHY_RATE_MCS4_LGI",
        "WIFI_PHY_RATE_MCS5_LGI",
        "WIFI_PHY_RATE_MCS6_LGI",
        "WIFI_PHY_RATE_MCS7_LGI",
    ]
    return symbols[int(mcs_idx)] if 0 <= mcs_idx < 8 else "UNKNOWN"


def predict_mcs(model_path, dataset_csv, output_csv, device="cpu"):
    """
    Load trained DQN model and predict best MCS for each packet.
    
    Args:
        model_path: Path to saved DQN model checkpoint
        dataset_csv: Path to DQN dataset
        output_csv: Path to save recommendations
        device: 'cpu' or 'cuda'
    """
    print(f"[DQN Prediction]")
    print(f"  Loading model from: {model_path}")
    
    # Load model
    try:
        agent = DQNAgent.load_model(model_path, device=device)
        print(f"  Model loaded successfully")
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return
    
    # Load dataset
    print(f"  Loading dataset from: {dataset_csv}")
    validate_dataset_feature_contract(dataset_csv, agent.state_schema)
    df = pd.read_csv(dataset_csv)
    print(f"  Dataset rows: {len(df)}")
    dataset_is_causal = "state_age_packets" in df.columns
    model_is_causal = agent.state_context_feature == "state_age_packets"
    if dataset_is_causal != model_is_causal:
        model_kind = "causal" if model_is_causal else "legacy same-packet"
        dataset_kind = "causal" if dataset_is_causal else "legacy same-packet"
        raise ValueError(
            f"Model expects a {model_kind} dataset, but received a {dataset_kind} dataset"
        )
    if dataset_is_causal and (
        df["state_age_packets"].isna().any()
        or (df["state_age_packets"] < 1).any()
    ):
        raise ValueError("Invalid causal state_age_packets values in prediction dataset")
    dataset_has_state_mcs = "state_mcs_index" in df.columns
    if agent.include_state_mcs and not dataset_has_state_mcs:
        raise ValueError(
            "Model was trained with source-MCS conditioning, but the dataset "
            "does not contain state_mcs_index"
        )
    
    # Predict for each row
    print(f"  Running predictions...")
    predictions = []
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            state = agent.normalize_states(
                build_state_vector(
                    row,
                    agent.state_context_feature,
                    agent.include_state_mcs,
                    agent.state_schema,
                )
            )
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            q_values = agent.q_net(state_tensor).cpu().numpy()[0]
            
            # Best action and Q-value
            best_action = np.argmax(q_values)
            best_q_value = float(q_values[best_action])
            
            # Top 3 alternatives
            top3_indices = np.argsort(q_values)[-3:][::-1]
            top3_mcs = [int(idx) for idx in top3_indices]
            top3_q_values = [float(q_values[idx]) for idx in top3_indices]
            
            predictions.append({
                "seq": int(row["seq"]),
                "actual_mcs": int(row["mcs_index"]),
                "predicted_mcs": int(best_action),
                "predicted_q_value": best_q_value,
                **{f"q_mcs{i}": float(q_values[i]) for i in range(len(q_values))},
                "top3_mcs": top3_mcs,
                "top3_q_values": top3_q_values,
                "actual_reward": float(row["reward"]),
                "service_ms": float(row["service_ms"]),
                "rssi": float(row.get("rssi", 0) or 0),
                "snr": float(row.get("snr", 0) or 0),
                "delivered": int(row["delivered"]),
                "state_mcs_index": int(row.get("state_mcs_index", -1)),
                "state_age_packets": int(row.get("state_age_packets", 0)),
                "state_missing_packets": int(row.get("state_missing_packets", 0)),
                "state_is_stale": int(row.get("state_is_stale", 0)),
            })
            
            if (idx + 1) % 1000 == 0:
                print(f"    Predicted {idx + 1}/{len(df)} packets...")
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    Path(output_csv).expanduser().parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_csv, index=False)
    print(f"  Predictions saved to: {output_csv}")
    
    # Statistics
    correct_preds = (pred_df["predicted_mcs"] == pred_df["actual_mcs"]).sum()
    accuracy = correct_preds / len(pred_df)
    
    print(f"  ✅ Prediction complete:")
    print(f"    Total predictions: {len(pred_df)}")
    print(f"    Matches actual MCS: {correct_preds} ({accuracy*100:.2f}%)")
    print(f"    Mean predicted Q-value: {pred_df['predicted_q_value'].mean():.4f}")
    print(f"    Mean actual reward: {pred_df['actual_reward'].mean():.4f}")
    print("    Note: actual_reward is for the logged actual_mcs, not a counterfactual reward for predicted_mcs")

    # Empirical baseline from the logged data. For delay reward, higher reward means lower service_ms.
    print(f"  Empirical logged baseline by actual MCS:")
    empirical = pred_df.groupby("actual_mcs").agg(
        n=("actual_mcs", "size"),
        mean_reward=("actual_reward", "mean"),
        mean_service=("service_ms", "mean"),
        median_service=("service_ms", "median"),
    ).sort_index()
    for mcs_idx, row in empirical.iterrows():
        print(
            f"    MCS{int(mcs_idx)}: n={int(row['n'])}, "
            f"mean_reward={row['mean_reward']:.4f}, "
            f"mean_service={row['mean_service']:.4f} ms, "
            f"median_service={row['median_service']:.4f} ms"
        )
    empirical_best_mean = int(empirical["mean_reward"].idxmax())
    empirical_best_median = int(empirical["median_service"].idxmin())
    print(f"    Best by mean reward: MCS{empirical_best_mean}")
    print(f"    Best by median service: MCS{empirical_best_median}")

    print(f"  Mean learned Q-value by action:")
    q_cols = [f"q_mcs{i}" for i in range(8) if f"q_mcs{i}" in pred_df.columns]
    mean_q_by_action = pred_df[q_cols].mean()
    for col, mean_q in mean_q_by_action.items():
        print(f"    {col.replace('q_mcs', 'MCS')}: {mean_q:.4f}")
    learned_best_mean_q = int(mean_q_by_action.idxmax().replace("q_mcs", ""))
    if learned_best_mean_q != empirical_best_mean:
        print(
            f"    WARNING: learned best mean-Q action MCS{learned_best_mean_q} "
            f"differs from empirical best mean-reward action MCS{empirical_best_mean}"
        )
    
    # Top MCS selections
    print(f"  Top MCS selections by frequency:")
    mcs_counts = pred_df["predicted_mcs"].value_counts().sort_index()
    for mcs_idx, count in mcs_counts.items():
        pct = count / len(pred_df) * 100
        symbol = mcs_to_symbol(mcs_idx)
        print(f"    MCS{mcs_idx} ({symbol}): {count} ({pct:.1f}%)")
    if len(mcs_counts):
        most_common_action = int(mcs_counts.idxmax())
        if most_common_action != empirical_best_mean:
            print(
                f"    WARNING: most common recommendation MCS{most_common_action} "
                f"differs from empirical best mean-reward action MCS{empirical_best_mean}"
            )

    if dataset_is_causal:
        print("  Recommendation distribution by CSI freshness:")
        freshness_groups = (
            ("fresh age=1", pred_df["state_age_packets"] == 1),
            ("stale age>1", pred_df["state_age_packets"] > 1),
        )
        for label, mask in freshness_groups:
            subset = pred_df.loc[mask]
            print(f"    {label}: {len(subset)} rows")
            if subset.empty:
                continue
            counts = subset["predicted_mcs"].value_counts().sort_index()
            rendered = "  ".join(
                f"MCS{int(mcs)}={int(count)} ({count / len(subset) * 100:.1f}%)"
                for mcs, count in counts.items()
            )
            print(f"      {rendered}")
    
    # Recommendation validity
    print(f"  Recommendation validity:")
    q_val_std = pred_df["predicted_q_value"].std()
    q_val_mean = pred_df["predicted_q_value"].mean()
    print(f"    Mean Q-value: {q_val_mean:.4f}")
    print(f"    Std Q-value: {q_val_std:.4f}")
    if q_val_std > 0:
        print(f"    Q-values have variance (good: model is differentiating)")
    else:
        print(f"    WARNING: Q-values have no variance (model may not be trained properly)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict MCS using trained DQN")
    parser.add_argument(
        "--model", default="dqn_mcs_model.pth",
        help="Path to trained DQN model"
    )
    parser.add_argument(
        "--dataset", default="rl_dqn_dataset.csv",
        help="Path to DQN dataset"
    )
    parser.add_argument(
        "--output", default="dqn_mcs_recommendations.csv",
        help="Path to save recommendations"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    predict_mcs(
        args.model,
        args.dataset,
        args.output,
        device=args.device,
    )
