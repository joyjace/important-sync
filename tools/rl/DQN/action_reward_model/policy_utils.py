#!/usr/bin/env python3
"""
Policy derivation for the two-head bandit outcome model.

Given per-action delivery probabilities and log-delay predictions, compute an
expected-utility score per action. The "utility" objective mirrors the dataset
builder's bounded log-goodput reward so model-derived utilities are directly
comparable to the dataset's reward column.
"""

from __future__ import annotations

import numpy as np


def utility_scale_from_delivered_service(service_ms: np.ndarray, payload_bytes: int) -> float:
    """q95 of log1p(goodput_kbps) over delivered packets (dataset convention)."""
    service_ms = np.asarray(service_ms, dtype=np.float64)
    service_ms = service_ms[np.isfinite(service_ms)]
    if len(service_ms) == 0:
        return 1.0
    goodput_kbps = payload_bytes * 8.0 / np.clip(service_ms, 1e-6, None)
    scores = np.log1p(goodput_kbps)
    return max(float(np.quantile(scores, 0.95)), 1e-6)


def expected_utility(
    p_deliver: np.ndarray,
    log_delay_mu: np.ndarray,
    objective: str = "utility",
    payload_bytes: int = 128,
    loss_reward: float = -1.0,
    utility_scale: float = 8.44,
    tail_target_ms: float = 0.0,
    tail_weight: float = 0.0,
) -> np.ndarray:
    """
    Score actions from the two calibrated heads.

    Args:
        p_deliver: P(delivered | s, a), any shape
        log_delay_mu: E[log service_ms | s, a, delivered], same shape
        objective:
            'utility': p * bounded log-goodput utility + (1-p) * loss_reward,
                       mirroring the dataset builder's utility reward
            'goodput': expected goodput in kbit/s (p * payload_bits / delay)
            'delay':   negative delivery-adjusted delay (delay / p), so that
                       argmax still selects the best action

    Returns array of scores, higher is better for every objective.
    """
    p = np.asarray(p_deliver, dtype=np.float64)
    delay_ms = np.exp(np.asarray(log_delay_mu, dtype=np.float64))
    delay_ms = np.clip(delay_ms, 1e-3, 1e4)

    if objective == "utility":
        goodput_kbps = payload_bytes * 8.0 / delay_ms
        success_utility = 2.0 * np.clip(np.log1p(goodput_kbps) / utility_scale, 0.0, 1.0) - 1.0
        if tail_target_ms > 0.0 and tail_weight > 0.0:
            tail_excess = np.maximum((delay_ms - tail_target_ms) / tail_target_ms, 0.0)
            success_utility = success_utility - tail_weight * np.clip(tail_excess, 0.0, 1.0)
        success_utility = np.clip(success_utility, -1.0, 1.0)
        return p * success_utility + (1.0 - p) * loss_reward

    if objective == "goodput":
        return p * payload_bytes * 8.0 / delay_ms

    if objective == "delay":
        return -(delay_ms / np.maximum(p, 1e-3))

    raise ValueError(f"Unknown objective: {objective}")
