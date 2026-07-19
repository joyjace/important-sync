#!/usr/bin/env python3
"""
Evaluate an MCS policy from prediction CSVs using offline replay and IPS/SNIPS estimates.

This is more aligned with the task goal than "matches actual MCS" because the logged MCS
is not necessarily the best MCS for a packet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def infer_logged_propensity(df: pd.DataFrame, mode: str) -> dict[int, float]:
    actions = sorted(int(x) for x in df["actual_mcs"].dropna().unique())
    if not actions:
        return {}
    if mode == "uniform":
        p = 1.0 / len(actions)
        return {action: p for action in actions}
    counts = df["actual_mcs"].value_counts(normalize=True).sort_index()
    return {int(action): float(prob) for action, prob in counts.items()}


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return float("nan")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")

    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = quantile * cumulative[-1]
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def capped_mean(values: pd.Series, cap_value: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(values.clip(upper=cap_value).mean())


def snips_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return float("nan")
    return float(np.sum(weights * values) / weight_sum)


def parse_thresholds(value: str) -> list[float]:
    thresholds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        threshold = float(part)
        if threshold <= 0:
            raise ValueError("exceedance thresholds must be positive")
        thresholds.append(threshold)
    return thresholds


def summarize_static_baselines(work: pd.DataFrame, cap_value: float) -> dict[str, dict[str, float]]:
    grouped = work.groupby("actual_mcs")["service_ms"]
    return {
        "mean": {str(int(action)): float(series.mean()) for action, series in grouped},
        "median": {str(int(action)): float(series.median()) for action, series in grouped},
        "p95": {str(int(action)): float(series.quantile(0.95)) for action, series in grouped},
        "capped_mean": {str(int(action)): capped_mean(series, cap_value) for action, series in grouped},
    }


def static_candidate_metrics(work: pd.DataFrame, cap_value: float) -> dict[str, dict[str, float]]:
    candidates = {}
    for action, group in work.groupby("actual_mcs"):
        key = f"static_MCS{int(action)}"
        metrics = {
            "mean_service_ms": float(group["service_ms"].mean()),
            "median_service_ms": float(group["service_ms"].median()),
            "p95_service_ms": float(group["service_ms"].quantile(0.95)),
            "capped_mean_service_ms": capped_mean(group["service_ms"], cap_value),
            "loss_rate": float(1.0 - group["delivered"].mean()) if "delivered" in group.columns else float("nan"),
        }
        if "actual_reward" in group.columns:
            metrics["mean_reward"] = float(group["actual_reward"].mean())
        candidates[key] = metrics
    return candidates


def add_static_exceedance_metrics(
    candidates: dict[str, dict[str, float]],
    work: pd.DataFrame,
    thresholds_ms: list[float],
) -> None:
    for action, group in work.groupby("actual_mcs"):
        key = f"static_MCS{int(action)}"
        if key not in candidates:
            continue
        for threshold in thresholds_ms:
            candidates[key][f"exceed_{threshold:g}ms_rate"] = float((group["service_ms"] > threshold).mean())


def policy_exceedance_metrics(work: pd.DataFrame, thresholds_ms: list[float]) -> dict[str, float]:
    metrics = {}
    weights = work["weight"].to_numpy(dtype=np.float64)
    service = work["service_ms"].to_numpy(dtype=np.float64)
    for threshold in thresholds_ms:
        metrics[f"exceed_{threshold:g}ms_rate"] = snips_mean((service > threshold).astype(np.float64), weights)
    return metrics


def minmax_component(values_by_name: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    finite = {name: value for name, value in values_by_name.items() if np.isfinite(value)}
    if not finite:
        return {name: float("nan") for name in values_by_name}

    min_value = min(finite.values())
    max_value = max(finite.values())
    if np.isclose(min_value, max_value):
        return {name: (1.0 if np.isfinite(value) else float("nan")) for name, value in values_by_name.items()}

    scores = {}
    for name, value in values_by_name.items():
        if not np.isfinite(value):
            scores[name] = float("nan")
        elif higher_is_better:
            scores[name] = float((value - min_value) / (max_value - min_value))
        else:
            scores[name] = float((max_value - value) / (max_value - min_value))
    return scores


def composite_scores(
    candidates: dict[str, dict[str, float]],
    reward_weight: float,
    median_weight: float,
    p95_weight: float,
    capped_mean_weight: float,
    loss_weight: float,
) -> dict[str, dict[str, float]]:
    weights = {
        "reward_component": reward_weight,
        "median_component": median_weight,
        "p95_component": p95_weight,
        "capped_mean_component": capped_mean_weight,
        "loss_component": loss_weight,
    }
    positive_weight_sum = sum(weight for weight in weights.values() if weight > 0)
    if positive_weight_sum <= 0:
        raise ValueError("At least one composite score weight must be positive")

    raw_components = {
        "reward_component": minmax_component(
            {name: metrics.get("mean_reward", float("nan")) for name, metrics in candidates.items()},
            higher_is_better=True,
        ),
        "median_component": minmax_component(
            {name: metrics.get("median_service_ms", float("nan")) for name, metrics in candidates.items()},
            higher_is_better=False,
        ),
        "p95_component": minmax_component(
            {name: metrics.get("p95_service_ms", float("nan")) for name, metrics in candidates.items()},
            higher_is_better=False,
        ),
        "capped_mean_component": minmax_component(
            {name: metrics.get("capped_mean_service_ms", float("nan")) for name, metrics in candidates.items()},
            higher_is_better=False,
        ),
        "loss_component": minmax_component(
            {name: metrics.get("loss_rate", float("nan")) for name, metrics in candidates.items()},
            higher_is_better=False,
        ),
    }

    scored = {}
    for name, metrics in candidates.items():
        entry = dict(metrics)
        weighted_sum = 0.0
        used_weight = 0.0
        for component, weight in weights.items():
            value = raw_components[component].get(name, float("nan"))
            entry[component] = value
            if weight > 0 and np.isfinite(value):
                weighted_sum += weight * value
                used_weight += weight
        entry["composite_score"] = float(weighted_sum / used_weight) if used_weight > 0 else float("nan")
        scored[name] = entry
    return scored


def evaluate_predictions(
    df: pd.DataFrame,
    propensity_mode: str,
    cap_quantile: float,
    exceedance_thresholds_ms: list[float],
    score_reward_weight: float,
    score_median_weight: float,
    score_p95_weight: float,
    score_capped_mean_weight: float,
    score_loss_weight: float,
) -> dict[str, object]:
    required = {"predicted_mcs", "actual_mcs", "service_ms"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Prediction CSV missing required columns: {missing}")

    work = df.copy()
    work = work.dropna(subset=["predicted_mcs", "actual_mcs", "service_ms"])
    if work.empty:
        raise ValueError("No valid rows available after dropping NaNs")

    work["predicted_mcs"] = work["predicted_mcs"].astype(int)
    work["actual_mcs"] = work["actual_mcs"].astype(int)
    work["service_ms"] = work["service_ms"].astype(float)
    if "actual_reward" in work.columns:
        work["actual_reward"] = pd.to_numeric(work["actual_reward"], errors="coerce")

    propensity = infer_logged_propensity(work, propensity_mode)
    work["logged_p"] = work["actual_mcs"].map(propensity)
    work["match"] = work["predicted_mcs"] == work["actual_mcs"]
    work["weight"] = np.where(work["match"], 1.0 / work["logged_p"], 0.0)

    replay = work[work["match"]].copy()
    coverage = float(work["match"].mean())
    cap_value = float(work["service_ms"].quantile(cap_quantile))

    result = {
        "rows": int(len(work)),
        "coverage": coverage,
        "logged_propensity_mode": propensity_mode,
        "cap_quantile": float(cap_quantile),
        "cap_value_service_ms": cap_value,
        "logged_propensity": {str(k): v for k, v in propensity.items()},
        "predicted_distribution": {
            str(int(action)): int(count)
            for action, count in work["predicted_mcs"].value_counts().sort_index().items()
        },
        "static_logged_service_ms": summarize_static_baselines(work, cap_value),
    }

    result["replay_mean_service_ms"] = float(replay["service_ms"].mean()) if len(replay) else float("nan")
    result["ips_mean_service_ms"] = float((work["weight"] * work["service_ms"]).mean())
    result["snips_mean_service_ms"] = snips_mean(work["service_ms"].to_numpy(dtype=np.float64), work["weight"].to_numpy(dtype=np.float64))

    result["replay_capped_mean_service_ms"] = capped_mean(replay["service_ms"], cap_value) if len(replay) else float("nan")
    result["snips_capped_mean_service_ms"] = snips_mean(
        work["service_ms"].clip(upper=cap_value).to_numpy(dtype=np.float64),
        work["weight"].to_numpy(dtype=np.float64),
    )

    result["replay_median_service_ms"] = float(replay["service_ms"].median()) if len(replay) else float("nan")
    result["replay_p95_service_ms"] = float(replay["service_ms"].quantile(0.95)) if len(replay) else float("nan")
    result["snips_median_service_ms"] = weighted_quantile(
        work["service_ms"].to_numpy(dtype=np.float64),
        work["weight"].to_numpy(dtype=np.float64),
        0.5,
    )
    result["snips_p95_service_ms"] = weighted_quantile(
        work["service_ms"].to_numpy(dtype=np.float64),
        work["weight"].to_numpy(dtype=np.float64),
        0.95,
    )

    if "actual_reward" in work.columns:
        valid_reward = work["actual_reward"].notna()
        reward_work = work[valid_reward].copy()
        reward_replay = reward_work[reward_work["match"]]
        result["replay_mean_reward"] = float(reward_replay["actual_reward"].mean()) if len(reward_replay) else float("nan")
        result["ips_mean_reward"] = float((reward_work["weight"] * reward_work["actual_reward"]).mean()) if len(reward_work) else float("nan")
        result["snips_mean_reward"] = snips_mean(
            reward_work["actual_reward"].to_numpy(dtype=np.float64),
            reward_work["weight"].to_numpy(dtype=np.float64),
        )

    candidates = static_candidate_metrics(work, cap_value)
    add_static_exceedance_metrics(candidates, work, exceedance_thresholds_ms)
    policy_metrics = {
        "mean_service_ms": result["snips_mean_service_ms"],
        "median_service_ms": result["snips_median_service_ms"],
        "p95_service_ms": result["snips_p95_service_ms"],
        "capped_mean_service_ms": result["snips_capped_mean_service_ms"],
        "loss_rate": (
            1.0 - snips_mean(
                work["delivered"].to_numpy(dtype=np.float64),
                work["weight"].to_numpy(dtype=np.float64),
            )
            if "delivered" in work.columns
            else float("nan")
        ),
    }
    if "snips_mean_reward" in result:
        policy_metrics["mean_reward"] = result["snips_mean_reward"]
    policy_metrics.update(policy_exceedance_metrics(work, exceedance_thresholds_ms))
    candidates["policy"] = policy_metrics

    result["composite_score_weights"] = {
        "reward": score_reward_weight,
        "median": score_median_weight,
        "p95": score_p95_weight,
        "capped_mean": score_capped_mean_weight,
        "loss": score_loss_weight,
    }
    result["composite_candidates"] = composite_scores(
        candidates,
        reward_weight=score_reward_weight,
        median_weight=score_median_weight,
        p95_weight=score_p95_weight,
        capped_mean_weight=score_capped_mean_weight,
        loss_weight=score_loss_weight,
    )
    result["composite_ranking"] = [
        name
        for name, _metrics in sorted(
            result["composite_candidates"].items(),
            key=lambda item: item[1].get("composite_score", -float("inf")),
            reverse=True,
        )
    ]

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy delay from prediction CSV")
    parser.add_argument("--predictions", required=True, help="Prediction CSV from predict_dqn.py or predict_reward_model.py")
    parser.add_argument(
        "--logged-propensity",
        choices=["empirical", "uniform"],
        default="empirical",
        help="How to estimate logging policy probabilities for IPS/SNIPS",
    )
    parser.add_argument(
        "--cap-quantile",
        type=float,
        default=0.99,
        help="Quantile used to cap service_ms for robust mean reporting",
    )
    parser.add_argument("--score-reward-weight", type=float, default=0.40, help="Composite score weight for mean reward")
    parser.add_argument("--score-median-weight", type=float, default=0.25, help="Composite score weight for median service time")
    parser.add_argument("--score-p95-weight", type=float, default=0.25, help="Composite score weight for p95 service time")
    parser.add_argument("--score-capped-mean-weight", type=float, default=0.10, help="Composite score weight for capped mean service time")
    parser.add_argument("--score-loss-weight", type=float, default=0.0, help="Composite score weight for packet loss rate")
    parser.add_argument(
        "--exceedance-thresholds-ms",
        default="1,5,10",
        help="Comma-separated service_ms thresholds to report as exceedance rates",
    )
    parser.add_argument("--summary-out", default=None, help="Optional JSON summary output path")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    summary = evaluate_predictions(
        df,
        args.logged_propensity,
        args.cap_quantile,
        exceedance_thresholds_ms=parse_thresholds(args.exceedance_thresholds_ms),
        score_reward_weight=args.score_reward_weight,
        score_median_weight=args.score_median_weight,
        score_p95_weight=args.score_p95_weight,
        score_capped_mean_weight=args.score_capped_mean_weight,
        score_loss_weight=args.score_loss_weight,
    )

    print("[Policy Delay Evaluation]")
    print(f"  Rows: {summary['rows']}")
    print(f"  Coverage (predicted == logged): {summary['coverage']:.4f}")
    print(f"  Replay mean service_ms: {summary['replay_mean_service_ms']:.4f}")
    print(f"  IPS mean service_ms: {summary['ips_mean_service_ms']:.4f}")
    print(f"  SNIPS mean service_ms: {summary['snips_mean_service_ms']:.4f}")
    print(f"  Capped mean quantile: {summary['cap_quantile']:.3f} (cap={summary['cap_value_service_ms']:.4f} ms)")
    print(f"  Replay capped mean service_ms: {summary['replay_capped_mean_service_ms']:.4f}")
    print(f"  SNIPS capped mean service_ms: {summary['snips_capped_mean_service_ms']:.4f}")
    print(f"  Replay median service_ms: {summary['replay_median_service_ms']:.4f}")
    print(f"  SNIPS median service_ms: {summary['snips_median_service_ms']:.4f}")
    print(f"  Replay p95 service_ms: {summary['replay_p95_service_ms']:.4f}")
    print(f"  SNIPS p95 service_ms: {summary['snips_p95_service_ms']:.4f}")
    if "replay_mean_reward" in summary:
        print(f"  Replay mean reward: {summary['replay_mean_reward']:.4f}")
        print(f"  IPS mean reward: {summary['ips_mean_reward']:.4f}")
        print(f"  SNIPS mean reward: {summary['snips_mean_reward']:.4f}")
    print("  Logged static mean baselines by actual MCS:")
    for action, value in summary["static_logged_service_ms"]["mean"].items():
        print(f"    MCS{action}: {value:.4f} ms")
    print("  Logged static median baselines by actual MCS:")
    for action, value in summary["static_logged_service_ms"]["median"].items():
        print(f"    MCS{action}: {value:.4f} ms")
    print("  Logged static p95 baselines by actual MCS:")
    for action, value in summary["static_logged_service_ms"]["p95"].items():
        print(f"    MCS{action}: {value:.4f} ms")
    print("  Logged static capped-mean baselines by actual MCS:")
    for action, value in summary["static_logged_service_ms"]["capped_mean"].items():
        print(f"    MCS{action}: {value:.4f} ms")
    print("  Composite score ranking (higher is better):")
    print(
        "    weights: "
        f"reward={summary['composite_score_weights']['reward']:.2f}, "
        f"median={summary['composite_score_weights']['median']:.2f}, "
        f"p95={summary['composite_score_weights']['p95']:.2f}, "
        f"capped_mean={summary['composite_score_weights']['capped_mean']:.2f}, "
        f"loss={summary['composite_score_weights']['loss']:.2f}"
    )
    for rank, name in enumerate(summary["composite_ranking"], start=1):
        metrics = summary["composite_candidates"][name]
        label = "learned_policy" if name == "policy" else name.replace("static_", "")
        print(
            f"    {rank}. {label}: score={metrics['composite_score']:.4f}, "
            f"reward={metrics.get('mean_reward', float('nan')):.4f}, "
            f"median={metrics.get('median_service_ms', float('nan')):.4f} ms, "
            f"p95={metrics.get('p95_service_ms', float('nan')):.4f} ms, "
            f"capped_mean={metrics.get('capped_mean_service_ms', float('nan')):.4f} ms"
        )
    if args.exceedance_thresholds_ms.strip():
        print("  Exceedance rates by candidate:")
        for name in summary["composite_ranking"]:
            metrics = summary["composite_candidates"][name]
            label = "learned_policy" if name == "policy" else name.replace("static_", "")
            parts = []
            for key, value in sorted(metrics.items()):
                if key.startswith("exceed_") and key.endswith("ms_rate"):
                    threshold = key.removeprefix("exceed_").removesuffix("ms_rate")
                    parts.append(f">{threshold}ms={value * 100:.2f}%")
            if parts:
                print(f"    {label}: " + ", ".join(parts))

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"  Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
