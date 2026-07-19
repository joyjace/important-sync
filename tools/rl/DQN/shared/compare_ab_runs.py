#!/usr/bin/env python3
"""
Compare live A/B runs (e.g., bandit policy vs Minstrel vs static MCS).

Each input is LABEL=path/to/ack_data.csv. Reports delivery, service-time
percentiles, goodput, the dataset-comparable utility reward, the MCS usage
histogram, and a recommendation-collision check (periodic loss bursts locked
to a delivered-packet cadence indicate receiver reco traffic interfering with
the data stream — see censor_reco_bursts.py).

Usage:
    python compare_ab_runs.py \
        bandit=runs/ab1_bandit/ack_data.csv \
        minstrel=runs/ab1_minstrel/ack_data.csv \
        static3=runs/ab1_static3/ack_data.csv \
        --payload-bytes 128 --utility-scale 8.35
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def utility_reward(service_ms: np.ndarray, delivered: np.ndarray, payload_bytes: int,
                   scale: float, loss_reward: float, tail_target_ms: float,
                   tail_weight: float) -> np.ndarray:
    service_ms = np.clip(service_ms, 1e-6, None)
    goodput_kbps = np.where(delivered == 1, payload_bytes * 8.0 / service_ms, 0.0)
    reward = 2.0 * np.clip(np.log1p(goodput_kbps) / scale, 0.0, 1.0) - 1.0
    reward = np.where(delivered == 1, reward, loss_reward)
    if tail_target_ms > 0 and tail_weight > 0:
        tail = np.clip((service_ms - tail_target_ms) / tail_target_ms, 0.0, 1.0)
        reward = np.where(delivered == 1, reward - tail_weight * tail, reward)
    return np.clip(reward, -1.0, 1.0)


def collision_check(delivered: np.ndarray, period: int = 20) -> tuple[float, float]:
    """Return (median delivered-count between loss-burst starts, burstiness).

    A median near `period` with low spread means receiver reco traffic is
    still colliding with the data stream. Returns (nan, nan) with few losses.
    """
    loss = 1 - delivered
    starts = np.flatnonzero((loss == 1) & (np.r_[0, loss[:-1]] == 0))
    if len(starts) < 30:
        return float("nan"), float("nan")
    cum = np.cumsum(delivered)
    dcount = np.diff(cum[starts])
    p_loss = float(loss.mean())
    p_loss_given_loss = float(loss[1:][loss[:-1] == 1].mean())
    burstiness = p_loss_given_loss / max(p_loss, 1e-9)
    return float(np.median(dcount)), burstiness


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare A/B live runs")
    parser.add_argument("runs", nargs="+", help="LABEL=path/to/ack_data.csv")
    parser.add_argument("--payload-bytes", type=int, default=128)
    parser.add_argument("--utility-scale", type=float, default=8.35,
                        help="log1p-goodput scale; use the training value for comparability")
    parser.add_argument("--utility-loss-reward", type=float, default=-1.0)
    parser.add_argument("--utility-tail-target-ms", type=float, default=1.0)
    parser.add_argument("--utility-tail-weight", type=float, default=0.10)
    args = parser.parse_args()

    rows = []
    mcs_tables = {}
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"Expected LABEL=path, got: {spec}")
        label, path = spec.split("=", 1)
        ack = pd.read_csv(path).sort_values("seq").reset_index(drop=True)
        delivered = ack["delivered"].to_numpy(dtype=np.int8)
        service_ms = ack["service_us"].to_numpy(dtype=np.float64) / 1000.0
        svc_ok = service_ms[delivered == 1]
        reward = utility_reward(
            service_ms, delivered, args.payload_bytes, args.utility_scale,
            args.utility_loss_reward, args.utility_tail_target_ms, args.utility_tail_weight,
        )
        duration_s = len(ack) / 200.0
        goodput_kbps = delivered.sum() * args.payload_bytes * 8.0 / 1000.0 / duration_s
        period_med, burstiness = collision_check(delivered)

        rows.append({
            "run": label,
            "packets": len(ack),
            "pdr": float(delivered.mean()),
            "svc_med_ms": float(np.median(svc_ok)) if len(svc_ok) else float("nan"),
            "svc_p95_ms": float(np.percentile(svc_ok, 95)) if len(svc_ok) else float("nan"),
            "svc_p99_ms": float(np.percentile(svc_ok, 99)) if len(svc_ok) else float("nan"),
            "goodput_kbps": goodput_kbps,
            "mean_reward": float(reward.mean()),
            "loss_burst_period_delivered": period_med,
            "loss_burstiness": burstiness,
        })
        mcs_tables[label] = (
            ack.groupby("mcs_index")
            .agg(share=("delivered", "size"), pdr=("delivered", "mean"))
            .assign(share=lambda t: t["share"] / len(ack))
        )

    table = pd.DataFrame(rows).set_index("run")
    pd.set_option("display.width", 160)
    print("\n=== A/B comparison ===")
    print(table.round(4).to_string())

    print("\n=== MCS usage per run ===")
    for label, t in mcs_tables.items():
        parts = "  ".join(
            f"MCS{int(m)}:{r.share:.0%}({r.pdr:.2f})" for m, r in t.iterrows()
        )
        print(f"{label:12s} {parts}")

    print("\nNotes:")
    print("- mean_reward uses the training utility formula; higher is better.")
    print("- loss_burst_period_delivered near 20 with burstiness >> 1 means the")
    print("  receiver's recommendation traffic is colliding with data packets;")
    print("  check the recommendation send timing before trusting the run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
