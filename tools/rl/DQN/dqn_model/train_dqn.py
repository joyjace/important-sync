#!/usr/bin/env python3
"""
Train a Deep Q-Learning (DQN) agent for MCS selection based on CSI state and service-time reward.

Architecture:
    - State: legacy_v1 or link_v2 CSI/link-control features
    - Action: MCS index 0-7 (8 actions)
    - Reward: per-packet utility/delay/PDR reward from the dataset
    - Q-function: 2-layer neural network with experience replay

Usage:
    python train_dqn.py --dataset rl_dqn_dataset.csv \\
                        --output dqn_mcs_model.pth \\
                        --epochs 100 \\
                        --batch-size 32 \\
                        --buffer-size 10000 \\
                        --eval-split 0.2

Output:
    - dqn_mcs_model.pth: Trained Q-network weights
    - training_log.json: Episode rewards, loss history
    - Optional TensorBoard event files with training and evaluation visuals
"""

import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pathlib import Path
import random
import time


DQN_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = DQN_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from csi_link_v7c import (
    CONTRACT_SHA256 as LINK_V7C_CONTRACT_SHA256,
    FEATURE_COUNT as LINK_V7C_FEATURE_COUNT,
    FEATURE_NAMES as LINK_V7C_FEATURE_NAMES,
    SCHEMA_ID as LINK_V7C_CONTRACT_ID,
    validate_dataset_artifact as validate_link_v7c_dataset_artifact,
)


LEGACY_NUMERIC_FEATURE_COLUMNS = [
    "rssi",
    "snr",
    "fft_gain",
    "agc_gain",
    "channel",
    "context",
    "iq_mean",
    "iq_std",
    "iq_p10",
    "iq_p50",
    "iq_p90",
]

LINK_V2_NUMERIC_FEATURE_COLUMNS = [
    "rssi",
    "snr",
    "fft_gain",
    "agc_gain",
    "channel",
    "sig_len",
    "state_age_packets",
    "state_packet_gap",
    "state_missing_packets",
    "state_is_stale",
    "iq_mean",
    "iq_std",
    "iq_p10",
    "iq_p50",
    "iq_p90",
]

LINK_V3_NUMERIC_FEATURE_COLUMNS = LINK_V2_NUMERIC_FEATURE_COLUMNS + [
    "state_prev_delivered",
    "state_consecutive_losses",
    "state_recent_loss_rate_8",
]

LINK_V4_NUMERIC_FEATURE_COLUMNS = [
    "rssi",
    "snr",
    "fft_gain",
    "agc_gain",
    "channel",
    "sig_len",
    "log1p_state_age_packets",
    "log1p_state_packet_gap",
    "log1p_state_missing_packets",
    "state_is_stale",
    "iq_mean",
    "iq_std",
    "iq_p10",
    "iq_p50",
    "iq_p90",
    "state_prev_delivered",
    "log1p_state_consecutive_losses",
    "state_recent_loss_rate_8",
]

LINK_V5_NUMERIC_FEATURE_COLUMNS = [
    "rssi",
    "snr",
    "fft_gain",
    "agc_gain",
    "channel",
    "sig_len",
    "log1p_state_age_packets",
    "log1p_state_packet_gap",
    "log1p_state_missing_packets",
    "state_is_stale",
    "iq_mean",
    "iq_std",
    "iq_p10",
    "iq_p50",
    "iq_p90",
]

LINK_V6_NUMERIC_FEATURE_COLUMNS = [
    *LINK_V5_NUMERIC_FEATURE_COLUMNS,
    "phase_mean",
    "phase_std",
    "phase_p10",
    "phase_p50",
    "phase_p90",
]

# link_v7c replaces the fragile absolute/unwrapped phase vector with the
# versioned 166-value HT20 transform. Its five amplitude summaries are derived
# from the same 56 active amplitudes (not the DC-inclusive legacy 57-bin view).
LINK_V7C_NUMERIC_FEATURE_COLUMNS = [
    *LINK_V5_NUMERIC_FEATURE_COLUMNS[:10],
    "iq_active_mean",
    "iq_active_std",
    "iq_active_p10",
    "iq_active_p50",
    "iq_active_p90",
]
LINK_V7C_STATE_SCHEMA = "link_v7c"
LINK_V7C_STATE_CONTRACT_ID = "link_v7c_receiveronly_v1"
LINK_V7C_STATE_CONTRACT_SHA256 = "8c88cd04a5e2c28366f567a5928e8df6df685f05284e5683a6a4d3cfd0beb790"
LINK_V7C_REQUIRED_COLUMNS = {
    "iq_active_amplitudes",
    "iq_phase_diff_real",
    "iq_phase_diff_imag",
    "iq_phase_valid_fraction",
    "iq_phase_coherence",
}


# HT20 HT-LTF CSI on ESP32-C5/C6 carries 57 subcarriers. legacy_v1/link_v2
# reserve 117 amplitude slots (zero-padded); link_v3c drops the padding.
LINK_V3C_AMPLITUDE_COUNT = 57
DEFAULT_AMPLITUDE_COUNT = 117


def schema_amplitude_count(state_schema):
    if state_schema == "link_v3c":
        return LINK_V3C_AMPLITUDE_COUNT
    if state_schema == LINK_V7C_STATE_SCHEMA:
        return 56
    return DEFAULT_AMPLITUDE_COUNT


def feature_contract_metadata(state_schema):
    """Return immutable model metadata for contract-bound state schemas."""

    if state_schema != LINK_V7C_STATE_SCHEMA:
        return {}
    return {
        "csi_feature_contract_id": LINK_V7C_CONTRACT_ID,
        "csi_feature_contract_sha256": LINK_V7C_CONTRACT_SHA256,
        "csi_feature_count": LINK_V7C_FEATURE_COUNT,
        "state_contract_id": LINK_V7C_STATE_CONTRACT_ID,
        "state_contract_sha256": LINK_V7C_STATE_CONTRACT_SHA256,
    }


def validate_dataset_feature_contract(dataset_csv, state_schema):
    """Reject a link_v7c CSV unless its artifact and qualification are exact."""

    if state_schema != LINK_V7C_STATE_SCHEMA:
        return
    return validate_link_v7c_dataset_artifact(dataset_csv)


def resolve_state_schema_for_columns(columns, requested="auto"):
    if requested != "auto":
        if requested not in {"legacy_v1", "link_v2", "link_v3", "link_v3c", "link_v4", "link_v5", "link_v6", LINK_V7C_STATE_SCHEMA}:
            raise ValueError(f"Unknown state schema: {requested}")
        if requested == "link_v6" and "iq_phase_detrended" not in set(columns):
            raise ValueError("link_v6 requires an iq_phase_detrended column; rebuild datasets with current build_dqn_dataset.py")
        if requested == LINK_V7C_STATE_SCHEMA:
            missing = sorted(LINK_V7C_REQUIRED_COLUMNS - set(columns))
            if missing:
                raise ValueError(
                    "link_v7c requires the versioned full-CSI columns; missing "
                    + ", ".join(missing)
                )
        return requested
    # auto never selects compact/contract schemas: changing the CSI layout
    # must always be an explicit, provenance-checked choice.
    link_v3_markers = {
        "state_prev_delivered",
        "state_consecutive_losses",
        "state_recent_loss_rate_8",
    }
    if link_v3_markers.issubset(set(columns)):
        return "link_v4"
    link_v2_markers = {"state_packet_gap", "state_missing_packets", "state_is_stale"}
    return "link_v2" if link_v2_markers.issubset(set(columns)) else "legacy_v1"


def state_feature_names(state_schema, include_state_mcs):
    amplitude_count = schema_amplitude_count(state_schema)
    if state_schema in ("link_v2", "link_v3c"):
        names = [f"iq_amp_{idx}" for idx in range(amplitude_count)] + LINK_V2_NUMERIC_FEATURE_COLUMNS
    elif state_schema == "link_v3":
        names = [f"iq_amp_{idx}" for idx in range(117)] + LINK_V3_NUMERIC_FEATURE_COLUMNS
    elif state_schema == "link_v4":
        names = [f"iq_amp_{idx}" for idx in range(117)] + LINK_V4_NUMERIC_FEATURE_COLUMNS
    elif state_schema == "link_v5":
        names = [f"iq_amp_{idx}" for idx in range(117)] + LINK_V5_NUMERIC_FEATURE_COLUMNS
    elif state_schema == "link_v6":
        names = (
            [f"iq_amp_{idx}" for idx in range(117)]
            + [f"iq_phase_residual_{idx}" for idx in range(117)]
            + LINK_V6_NUMERIC_FEATURE_COLUMNS
        )
    elif state_schema == LINK_V7C_STATE_SCHEMA:
        if include_state_mcs:
            raise ValueError("link_v7c_receiveronly_v1 forbids source-MCS conditioning")
        names = list(LINK_V7C_FEATURE_NAMES) + LINK_V7C_NUMERIC_FEATURE_COLUMNS
    elif state_schema == "legacy_v1":
        names = [f"iq_amp_{idx}" for idx in range(amplitude_count)] + LEGACY_NUMERIC_FEATURE_COLUMNS
    else:
        raise ValueError(f"Unknown state schema: {state_schema}")
    if include_state_mcs:
        names += [f"state_mcs_is_{idx}" for idx in range(8)]
    return names


def link_v7c_state_contract_manifest():
    """Return the canonical receiver-only 181-value state contract."""

    names = state_feature_names(LINK_V7C_STATE_SCHEMA, False)
    return {
        "state_contract_id": LINK_V7C_STATE_CONTRACT_ID,
        "csi_feature_contract_id": LINK_V7C_CONTRACT_ID,
        "csi_feature_contract_sha256": LINK_V7C_CONTRACT_SHA256,
        "include_state_mcs": False,
        "link_feature_names": list(LINK_V7C_NUMERIC_FEATURE_COLUMNS),
        "state_feature_count": len(names),
        "state_feature_names": names,
    }


_LINK_V7C_STATE_CANONICAL_JSON = json.dumps(
    link_v7c_state_contract_manifest(),
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
)
if hashlib.sha256(
    _LINK_V7C_STATE_CANONICAL_JSON.encode("utf-8")
).hexdigest() != LINK_V7C_STATE_CONTRACT_SHA256:
    raise RuntimeError("link_v7c receiver-only state contract digest mismatch")


def link_v7c_active_amplitude_summary(amplitudes):
    """Mirror the firmware's float32 summary over the 56 active amplitudes."""

    values = np.asarray(amplitudes, dtype=np.float32)
    if values.shape != (56,) or not np.all(np.isfinite(values)):
        raise ValueError("link_v7c active amplitudes must be finite float32[56]")
    total = np.float32(0.0)
    total_sq = np.float32(0.0)
    for value in values:
        total = np.float32(total + value)
        total_sq = np.float32(total_sq + np.float32(value * value))
    count = np.float32(56.0)
    mean = np.float32(total / count)
    variance = np.float32(np.float32(total_sq / count) - np.float32(mean * mean))
    if variance < np.float32(0.0):
        variance = np.float32(0.0)
    std = np.float32(np.sqrt(variance))
    sorted_values = np.sort(values)

    def quantile(q):
        position = np.float32(np.float32(q) * np.float32(55.0))
        lower = int(position)
        upper = min(lower + 1, 55)
        fraction = np.float32(position - np.float32(lower))
        return np.float32(
            np.float32(sorted_values[lower] * np.float32(np.float32(1.0) - fraction))
            + np.float32(sorted_values[upper] * fraction)
        )

    return np.asarray(
        [mean, std, quantile(0.10), quantile(0.50), quantile(0.90)],
        dtype=np.float32,
    )


def create_tensorboard_writer(log_dir):
    """Create a TensorBoard writer only when visual logging is requested."""
    if log_dir is None:
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard logging requires the 'tensorboard' package. "
            "Install it with: pip install tensorboard"
        ) from exc

    log_dir = Path(log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def log_model_histograms(writer, model, step):
    """Log model parameters and available gradients for diagnostics."""
    for name, parameter in model.named_parameters():
        tag_name = name.replace(".", "/")
        writer.add_histogram(f"model/parameters/{tag_name}", parameter.detach().cpu(), step)
        if parameter.grad is not None:
            writer.add_histogram(
                f"model/gradients/{tag_name}", parameter.grad.detach().cpu(), step
            )


class QNetwork(nn.Module):
    """Neural network for Q-value approximation."""

    def __init__(
        self,
        state_dim=128,
        action_dim=8,
        hidden_dim=128,
        hidden_layers=2,
        dropout=0.0,
        layer_norm=False,
    ):
        super(QNetwork, self).__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")

        layers = []
        in_dim = state_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,)
        
        Returns:
            Q-values: Tensor of shape (batch_size, action_dim) or (action_dim,)
        """
        return self.net(state)


class DQNAgent:
    """Deep Q-Learning agent with experience replay."""
    
    def __init__(
        self,
        state_dim=128,
        action_dim=8,
        hidden_dim=128,
        hidden_layers=2,
        dropout=0.0,
        layer_norm=False,
        state_schema="legacy_v1",
        state_context_feature="sig_len",
        include_state_mcs=False,
        lr=1e-3,
        weight_decay=0.0,
        gamma=0.9,
        cql_alpha=0.0,
        q_target_clip=None,
        q_value_regularization=0.0,
        loss_type="mse",
        epsilon_start=0.1,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        device="cpu",
    ):
        self.state_dim = state_dim
        self.state_context_feature = state_context_feature
        self.include_state_mcs = include_state_mcs
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.state_schema = state_schema
        self.lr = lr
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        if q_target_clip is not None and q_target_clip <= 0:
            raise ValueError("q_target_clip must be positive when provided")
        self.q_target_clip = q_target_clip
        if q_value_regularization < 0:
            raise ValueError("q_value_regularization must be non-negative")
        self.q_value_regularization = q_value_regularization
        self.loss_type = loss_type
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.state_mean = None
        self.state_std = None
        
        # Q-network and target network
        self.q_net = QNetwork(
            state_dim,
            action_dim,
            hidden_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            layer_norm=layer_norm,
        ).to(device)
        self.target_net = QNetwork(
            state_dim,
            action_dim,
            hidden_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            layer_norm=layer_norm,
        ).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Training stats
        self.training_log = {
            "episodes": [],
            "mean_rewards": [],
            "losses": [],
        }
    
    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection.
        
        Args:
            state: Numpy array of shape (state_dim,)
            training: If True, use epsilon-greedy; else use greedy
        
        Returns:
            action: Integer in [0, action_dim)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = self.normalize_states(np.asarray(state, dtype=np.float32))
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax(dim=1).item()

    def set_state_normalizer(self, mean, std):
        self.state_mean = np.asarray(mean, dtype=np.float32)
        self.state_std = np.asarray(std, dtype=np.float32)
        self.state_std = np.where(self.state_std < 1e-6, 1.0, self.state_std)

    def normalize_states(self, states):
        if self.state_mean is None or self.state_std is None:
            return states
        return (states - self.state_mean) / self.state_std
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self, batch_size):
        """Train on a batch from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return None
        self.q_net.train()
        
        # Sample batch
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).float().to(self.device)
        
        # Compute Q-values
        q_values_all = self.q_net(states)  # (batch_size, action_dim)
        q_values = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch_size,)
        
        # Double DQN: online network selects, target network evaluates.
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            if self.q_target_clip is not None:
                target_q_values = target_q_values.clamp(
                    -self.q_target_clip,
                    self.q_target_clip,
                )
        
        # TD loss. Huber is often safer with service-time outliers than plain MSE.
        if self.loss_type == "huber":
            td_loss = nn.SmoothL1Loss()(q_values, target_q_values)
        else:
            td_loss = nn.MSELoss()(q_values, target_q_values)

        # Conservative Q-learning penalty to discourage over-estimating unseen actions.
        # This is helpful for offline RL where extrapolation error can collapse policy quality.
        if self.cql_alpha > 0:
            cql_loss = (torch.logsumexp(q_values_all, dim=1) - q_values).mean()
            loss = td_loss + self.cql_alpha * cql_loss
        else:
            loss = td_loss

        if self.q_value_regularization > 0 and self.q_target_clip is not None:
            excess_q = torch.relu(torch.abs(q_values_all) - self.q_target_clip)
            loss = loss + self.q_value_regularization * torch.mean(excess_q * excess_q)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Soft update of target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, path):
        """Save model and optimizer state."""
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_net_state": self.q_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
            "dropout": self.dropout,
            "layer_norm": self.layer_norm,
            "state_schema": self.state_schema,
            "lr": self.lr,
            "q_target_clip": self.q_target_clip,
            "q_value_regularization": self.q_value_regularization,
            "state_mean": None if self.state_mean is None else self.state_mean.tolist(),
            "state_std": None if self.state_std is None else self.state_std.tolist(),
            "state_context_feature": self.state_context_feature,
            "include_state_mcs": self.include_state_mcs,
            "state_feature_names": state_feature_names(self.state_schema, self.include_state_mcs),
            **feature_contract_metadata(self.state_schema),
            "training_log": self.training_log,
        }, path)
        print(f"  Model saved to: {path}")
    
    @classmethod
    def load_model(cls, path, device="cpu"):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        state_schema = checkpoint.get("state_schema", "legacy_v1")
        for key, expected in feature_contract_metadata(state_schema).items():
            if checkpoint.get(key) != expected:
                raise ValueError(
                    f"Checkpoint {key}={checkpoint.get(key)!r} does not match {expected!r}"
                )
        agent = cls(
            state_dim=checkpoint["state_dim"],
            action_dim=checkpoint["action_dim"],
            hidden_dim=checkpoint.get("hidden_dim", 128),
            hidden_layers=checkpoint.get("hidden_layers", 2),
            dropout=checkpoint.get("dropout", 0.0),
            layer_norm=checkpoint.get("layer_norm", False),
            state_schema=state_schema,
            state_context_feature=checkpoint.get("state_context_feature", "sig_len"),
            include_state_mcs=checkpoint.get(
                "include_state_mcs", checkpoint.get("state_dim") in {136, 140, 143, 262}
            ),
            lr=checkpoint.get("lr", 1e-3),
            q_target_clip=checkpoint.get("q_target_clip"),
            q_value_regularization=checkpoint.get("q_value_regularization", 0.0),
            device=device,
        )
        agent.q_net.load_state_dict(checkpoint["q_net_state"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        agent.q_net.eval()
        if checkpoint.get("state_mean") is not None and checkpoint.get("state_std") is not None:
            agent.set_state_normalizer(checkpoint["state_mean"], checkpoint["state_std"])
        agent.training_log = checkpoint.get("training_log", {})
        return agent


def build_state_vector(
    row,
    state_context_feature="auto",
    include_state_mcs="auto",
    state_schema="auto",
):
    """
    Build a state vector from one dataset row.

    Components:
        - IQ amplitudes (117 slots for legacy_v1/link_v2, 57 for link_v3c;
          shorter data is zero-padded, longer data truncated)
        - legacy_v1: 11 extracted CSI features: rssi, snr, fft_gain,
          agc_gain, channel, context, iq_mean, iq_std, iq_p10, iq_p50, iq_p90
        - link_v2/link_v3c: 15 link-control features: rssi, snr, fft_gain,
          agc_gain, channel, sig_len, state_age_packets, state_packet_gap,
          state_missing_packets, state_is_stale, iq_mean/std/p10/p50/p90
        - link_v3: link_v2 plus causal ACK-history features:
          state_prev_delivered, state_consecutive_losses,
          state_recent_loss_rate_8
        - link_v4: link_v3 with log-transformed packet/loss counters
        - link_v5: receiver-only link_v2 with log-transformed packet counters
          and no ACK-history features
        - link_v6: link_v5 plus 117 unwrapped linear-detrended phase residuals
          and phase residual summary statistics
        - link_v7c: the versioned 166-value HT20 amplitude/differential-phase
          contract plus the receiver-only link_v5 controls
        - Optional 8-value one-hot encoding of the MCS that produced the CSI

    ``context`` is ``state_age_packets`` for causal legacy datasets and
    ``sig_len`` for same-packet legacy datasets. ``link_v2`` keeps both
    ``sig_len`` and the causal age/gap/missing controls. ``link_v3c`` matches
    link_v2 but keeps only the 57 real HT20 subcarrier amplitudes instead of
    padding to 117.
    """
    def get_value(obj, key, default=0):
        if hasattr(obj, "get"):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def safe_float(value, default=0.0):
        try:
            v = float(value)
            return default if np.isnan(v) else v
        except Exception:
            return default

    def safe_log1p_count(value, default=0.0):
        return float(np.log1p(max(safe_float(value, default), 0.0)))

    if state_context_feature == "auto":
        state_context_feature = (
            "state_age_packets"
            if get_value(row, "state_age_packets", None) is not None
            else "sig_len"
        )
    if state_context_feature not in {"sig_len", "state_age_packets"}:
        raise ValueError(f"Unknown state context feature: {state_context_feature}")
    if include_state_mcs == "auto":
        include_state_mcs = get_value(row, "state_mcs_index", None) is not None
    if state_schema == "auto":
        state_schema = resolve_state_schema_for_columns(
            [key for key in ("state_packet_gap", "state_missing_packets", "state_is_stale") if get_value(row, key, None) is not None],
            "auto",
        )

    amplitude_count = schema_amplitude_count(state_schema)

    def parse_vector(value, length=117):
        try:
            if isinstance(value, str):
                raw = value.strip("[]").replace("\n", " ").replace(",", " ")
                arr = np.fromstring(raw, sep=" ")
            elif isinstance(value, list):
                arr = np.asarray(value)
            else:
                arr = np.asarray(value) if hasattr(value, "__iter__") else np.asarray([value])
            arr = np.asarray(arr, dtype=np.float32)
        except Exception:
            arr = np.zeros(length, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if len(arr) < length:
            arr = np.pad(arr, (0, length - len(arr)), mode="constant")
        return arr[:length]

    def parse_contract_vector(column, length):
        value = get_value(row, column, None)
        if value is None:
            raise ValueError(f"link_v7c row is missing {column}")
        try:
            if isinstance(value, str):
                raw = value.strip("[]").replace("\n", " ").replace(",", " ")
                array = np.fromstring(raw, sep=" ", dtype=np.float32)
            else:
                array = np.asarray(value, dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"link_v7c row has invalid {column}") from exc
        if array.shape != (length,):
            raise ValueError(
                f"link_v7c {column} must contain exactly {length} values, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"link_v7c {column} contains a non-finite value")
        return np.asarray(array, dtype=np.float32)

    # Extract IQ raw amplitudes (117 values; 57 for the compact link_v3c layout)
    iq_raw = get_value(row, "iq_raw_parsed", None)
    if iq_raw is None:
        iq_raw = get_value(row, "iq_raw", np.zeros(amplitude_count))
    iq_raw = parse_vector(iq_raw, amplitude_count)

    iq_phase_detrended = np.zeros(117, dtype=np.float32)
    if state_schema == "link_v6":
        phase_value = get_value(row, "iq_phase_detrended", None)
        if phase_value is None:
            raise ValueError("link_v6 requires iq_phase_detrended; rebuild datasets with current build_dqn_dataset.py")
        iq_phase_detrended = parse_vector(
            phase_value,
            117,
        )

    link_v7c_features = None
    if state_schema == LINK_V7C_STATE_SCHEMA:
        link_v7c_features = np.concatenate(
            [
                parse_contract_vector("iq_active_amplitudes", 56),
                parse_contract_vector("iq_phase_diff_real", 54),
                parse_contract_vector("iq_phase_diff_imag", 54),
                np.asarray(
                    [
                        safe_float(get_value(row, "iq_phase_valid_fraction", np.nan), np.nan),
                        safe_float(get_value(row, "iq_phase_coherence", np.nan), np.nan),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        if link_v7c_features.shape != (LINK_V7C_FEATURE_COUNT,) or not np.all(
            np.isfinite(link_v7c_features)
        ):
            raise ValueError("link_v7c row has invalid phase quality features")

    if state_schema == "legacy_v1":
        csi_features = np.array([
            safe_float(get_value(row, "rssi", 0)),
            safe_float(get_value(row, "snr", 0)),
            safe_float(get_value(row, "fft_gain", 0)),
            safe_float(get_value(row, "agc_gain", 0)),
            safe_float(get_value(row, "channel", 0)),
            safe_float(get_value(row, state_context_feature, 0)),
            safe_float(get_value(row, "iq_mean", 0)),
            safe_float(get_value(row, "iq_std", 0)),
            safe_float(get_value(row, "iq_p10", 0)),
            safe_float(get_value(row, "iq_p50", 0)),
            safe_float(get_value(row, "iq_p90", 0)),
        ], dtype=np.float32)
    elif state_schema in {"link_v2", "link_v3", "link_v3c", "link_v4", "link_v5", "link_v6", LINK_V7C_STATE_SCHEMA}:
        state_age = safe_float(get_value(row, "state_age_packets", 1.0), 1.0)
        state_gap = safe_float(get_value(row, "state_packet_gap", state_age), state_age)
        state_missing = safe_float(
            get_value(row, "state_missing_packets", max(state_gap - 1.0, 0.0)),
            max(state_gap - 1.0, 0.0),
        )
        state_is_stale = safe_float(
            get_value(row, "state_is_stale", 1.0 if (state_age > 1.0 or state_missing > 0.0) else 0.0),
            0.0,
        )
        feature_values = [
            safe_float(get_value(row, "rssi", 0)),
            safe_float(get_value(row, "snr", 0)),
            safe_float(get_value(row, "fft_gain", 0)),
            safe_float(get_value(row, "agc_gain", 0)),
            safe_float(get_value(row, "channel", 0)),
            safe_float(get_value(row, "sig_len", 0)),
        ]
        if state_schema in {"link_v4", "link_v5", "link_v6", LINK_V7C_STATE_SCHEMA}:
            feature_values.extend(
                [
                    safe_log1p_count(state_age),
                    safe_log1p_count(state_gap),
                    safe_log1p_count(state_missing),
                    state_is_stale,
                ]
            )
        else:
            feature_values.extend([state_age, state_gap, state_missing, state_is_stale])
        if state_schema == LINK_V7C_STATE_SCHEMA:
            feature_values.extend(
                link_v7c_active_amplitude_summary(link_v7c_features[:56]).tolist()
            )
        else:
            feature_values.extend(
                [
                    safe_float(get_value(row, "iq_mean", 0)),
                    safe_float(get_value(row, "iq_std", 0)),
                    safe_float(get_value(row, "iq_p10", 0)),
                    safe_float(get_value(row, "iq_p50", 0)),
                    safe_float(get_value(row, "iq_p90", 0)),
                ]
            )
        if state_schema == "link_v6":
            feature_values.extend(
                [
                    safe_float(get_value(row, "phase_mean", 0)),
                    safe_float(get_value(row, "phase_std", 0)),
                    safe_float(get_value(row, "phase_p10", 0)),
                    safe_float(get_value(row, "phase_p50", 0)),
                    safe_float(get_value(row, "phase_p90", 0)),
                ]
            )
        if state_schema in {"link_v3", "link_v4"}:
            consecutive_losses = safe_float(
                get_value(row, "state_consecutive_losses", 0.0), 0.0
            )
            feature_values.extend(
                [
                    safe_float(get_value(row, "state_prev_delivered", 1.0), 1.0),
                    safe_log1p_count(consecutive_losses)
                    if state_schema == "link_v4"
                    else consecutive_losses,
                    safe_float(get_value(row, "state_recent_loss_rate_8", 0.0), 0.0),
                ]
            )
        csi_features = np.array(feature_values, dtype=np.float32)
    else:
        raise ValueError(f"Unknown state schema: {state_schema}")
    
    # Base state: contract CSI or legacy amplitudes/phase plus link features.
    if state_schema == "link_v6":
        state = np.concatenate([iq_raw[:117], iq_phase_detrended[:117], csi_features]).astype(np.float32)
    elif state_schema == LINK_V7C_STATE_SCHEMA:
        state = np.concatenate([link_v7c_features, csi_features]).astype(np.float32)
    else:
        state = np.concatenate([iq_raw[:amplitude_count], csi_features]).astype(np.float32)

    if include_state_mcs:
        raw_state_mcs = get_value(row, "state_mcs_index", None)
        try:
            state_mcs = int(raw_state_mcs)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid state_mcs_index: {raw_state_mcs}") from exc
        if state_mcs < 0 or state_mcs >= 8:
            raise ValueError(f"state_mcs_index must be in [0, 7], got {state_mcs}")
        state_mcs_one_hot = np.zeros(8, dtype=np.float32)
        state_mcs_one_hot[state_mcs] = 1.0
        state = np.concatenate([state, state_mcs_one_hot])

    return state.astype(np.float32)


def order_temporal_dataframe(df):
    """Sort rows into a stable per-run sequence before building transitions."""
    ordered = df.copy()
    if "source_file" in ordered.columns:
        ordered["_episode_key"] = ordered["source_file"].astype(str)
    elif "source_scenario" in ordered.columns:
        ordered["_episode_key"] = ordered["source_scenario"].astype(str)
    else:
        ordered["_episode_key"] = "single_run"
    ordered["_seq_sort"] = pd.to_numeric(ordered.get("seq", 0), errors="coerce").fillna(0)
    ordered["_state_age_sort"] = pd.to_numeric(
        ordered.get("state_age_packets", 0), errors="coerce"
    ).fillna(0)
    ordered["_synthetic_sort"] = pd.to_numeric(
        ordered.get("synthetic_stale", 0), errors="coerce"
    ).fillna(0)
    ordered = ordered.sort_values(
        ["_episode_key", "_synthetic_sort", "_seq_sort", "_state_age_sort"],
        kind="mergesort",
    ).reset_index(drop=True)
    return ordered


def build_temporal_next_indices(frame):
    """
    Return next-state indices and done flags for logged temporal transitions.

    Observed rows transition to the next observed row in the same source run.
    Legacy synthetic stale rows remain terminal. Model-augmented rows may carry
    explicit transition IDs so stale age/action trajectories participate in
    gamma-discounted training without relying on CSV row adjacency.
    """
    n = len(frame)
    next_indices = np.arange(n, dtype=np.int64)
    dones = np.ones(n, dtype=np.float32)
    if n == 0:
        return next_indices, dones

    if "_episode_key" in frame.columns:
        episode_keys = frame["_episode_key"].astype(str)
    elif "source_file" in frame.columns:
        episode_keys = frame["source_file"].astype(str)
    elif "source_scenario" in frame.columns:
        episode_keys = frame["source_scenario"].astype(str)
    else:
        episode_keys = pd.Series(["single_run"] * n, index=frame.index)

    synthetic = pd.to_numeric(
        frame.get("synthetic_stale", pd.Series(0, index=frame.index)),
        errors="coerce",
    ).fillna(0).astype(int)

    model_augmented = pd.to_numeric(
        frame.get("model_augmented", pd.Series(0, index=frame.index)),
        errors="coerce",
    ).fillna(0).astype(int)

    observed_positions = np.flatnonzero(
        (synthetic.to_numpy() == 0) & (model_augmented.to_numpy() == 0)
    )
    observed = pd.DataFrame({
        "position": observed_positions,
        "episode": episode_keys.iloc[observed_positions].to_numpy(),
    })
    for _episode, group in observed.groupby("episode", sort=False):
        positions = group["position"].to_numpy(dtype=np.int64)
        if len(positions) < 2:
            continue
        next_indices[positions[:-1]] = positions[1:]
        dones[positions[:-1]] = 0.0

    modeled_positions = np.flatnonzero(model_augmented.to_numpy() == 1)
    if len(modeled_positions):
        required = {"model_transition_id", "model_next_transition_id"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"Model-augmented rows are missing transition columns: {missing}"
            )
        transition_ids = pd.to_numeric(
            frame["model_transition_id"], errors="coerce"
        ).fillna(0).astype(np.int64).to_numpy()
        next_transition_ids = pd.to_numeric(
            frame["model_next_transition_id"], errors="coerce"
        ).fillna(0).astype(np.int64).to_numpy()
        modeled_ids = transition_ids[modeled_positions]
        if (modeled_ids <= 0).any():
            raise ValueError("Model-augmented transition IDs must be positive")
        if len(np.unique(modeled_ids)) != len(modeled_ids):
            raise ValueError("Model-augmented transition IDs must be unique")
        position_by_id = {
            int(transition_ids[position]): int(position)
            for position in modeled_positions
        }
        for position in modeled_positions:
            next_id = int(next_transition_ids[position])
            next_position = position_by_id.get(next_id)
            if next_id > 0 and next_position is not None:
                next_indices[position] = next_position
                dones[position] = 0.0
    return next_indices, dones


def train_dqn(
    dataset_csv,
    output_model,
    epochs=100,
    batch_size=32,
    buffer_size=10000,
    eval_split=0.2,
    device="cpu",
    train_every=1,
    max_train_steps=None,
    gamma=0.9,
    cql_alpha=0.0,
    q_target_clip=None,
    q_value_regularization=0.0,
    loss_type="mse",
    min_reward=None,
    reward_scale=1.0,
    hidden_dim=128,
    hidden_layers=2,
    dropout=0.0,
    layer_norm=False,
    lr=1e-3,
    weight_decay=0.0,
    normalize_states=True,
    ignore_state_mcs=False,
    state_schema="auto",
    seed=42,
    tensorboard_dir=None,
    tensorboard_histograms_every=0,
):
    """
    Train DQN agent on dataset.
    
    Args:
        dataset_csv: Path to RL DQN dataset
        output_model: Path to save trained model
        epochs: Number of training epochs (iterations through data)
        batch_size: Batch size for experience replay
        buffer_size: Size of replay buffer
        eval_split: Fraction of data for evaluation
        device: 'cpu' or 'cuda'
        train_every: Run one optimizer step every N transitions
        max_train_steps: Optional cap on transitions processed per epoch
        gamma: Discount factor for temporal offline Q-learning
        cql_alpha: Conservative Q penalty weight to reduce action overestimation
        q_target_clip: Optional symmetric Bellman-target bound
        q_value_regularization: Optional penalty for Q outputs beyond q_target_clip
        loss_type: 'mse' or 'huber' TD loss
        min_reward: Optional lower bound for reward targets, useful for delay outliers
        reward_scale: Divide training rewards by this positive value
        hidden_dim: Width of each hidden layer
        hidden_layers: Number of hidden layers in the Q-network
        dropout: Optional dropout probability between hidden layers
        layer_norm: Add LayerNorm after each hidden linear layer
        lr: Adam learning rate
        weight_decay: Adam L2 regularization
        normalize_states: Standardize state features using train split statistics
        ignore_state_mcs: Ignore state_mcs_index even when the causal dataset
                          contains it, for a CSI-only ablation
        state_schema: 'auto', 'legacy_v1', 'link_v2', 'link_v3', 'link_v3c', 'link_v4', 'link_v5', or 'link_v6'
        seed: Random seed for reproducible shuffling and initialization
        tensorboard_dir: Optional directory for TensorBoard event files
        tensorboard_histograms_every: Log parameter/gradient histograms every N
            epochs; 0 disables them
    """
    if tensorboard_histograms_every < 0:
        raise ValueError("tensorboard_histograms_every must be non-negative")

    run_start = time.perf_counter()
    print(f"[DQN Training]", flush=True)
    print(f"  Loading dataset from: {dataset_csv}", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"  Seed: {seed}", flush=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    df = pd.read_csv(dataset_csv)
    print(f"  Dataset rows: {len(df)}", flush=True)
    if "state_age_packets" in df.columns:
        if df["state_age_packets"].isna().any():
            raise ValueError(
                "Causal dataset contains missing state_age_packets values; "
                "do not merge causal and legacy datasets"
            )
        if (df["state_age_packets"] < 1).any():
            raise ValueError("Causal state_age_packets must be at least 1")
    if "state_mcs_index" in df.columns:
        state_mcs_numeric = pd.to_numeric(df["state_mcs_index"], errors="coerce")
        if (
            state_mcs_numeric.isna().any()
            or (~state_mcs_numeric.between(0, 7)).any()
            or ((state_mcs_numeric % 1) != 0).any()
        ):
            raise ValueError("state_mcs_index must contain integer values from 0 to 7")
    
    # Handle iq_raw deserialization
    def parse_iq_raw(x):
        if isinstance(x, str):
            try:
                raw = x.strip("[]").replace("\n", " ").replace(",", " ")
                arr = np.fromstring(raw, sep=" ")
                if len(arr) == 0:
                    return np.zeros(117, dtype=np.float32)
                arr = np.asarray(arr, dtype=np.float32)
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                return np.zeros(117)
        elif isinstance(x, (list, np.ndarray)):
            arr = np.array(x, dtype=np.float32)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            return np.zeros(117)
    
    df["iq_raw_parsed"] = df["iq_raw"].apply(parse_iq_raw)
    df = order_temporal_dataframe(df)
    print(f"  Parsed IQ raw values in {time.perf_counter() - run_start:.1f}s", flush=True)
    
    # Split into train/eval
    split_idx = int(len(df) * (1 - eval_split))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    eval_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"  Train rows: {len(train_df)}, Eval rows: {len(eval_df)}", flush=True)

    # Precompute arrays once to avoid heavy per-epoch pandas iteration.
    precompute_start = time.perf_counter()
    train_rows = list(train_df.itertuples(index=False))
    eval_rows = list(eval_df.itertuples(index=False))
    state_context_feature = (
        "state_age_packets" if "state_age_packets" in df.columns else "sig_len"
    )
    include_state_mcs = "state_mcs_index" in df.columns and not ignore_state_mcs
    resolved_state_schema = resolve_state_schema_for_columns(df.columns, state_schema)
    validate_dataset_feature_contract(dataset_csv, resolved_state_schema)
    print(f"  State context feature: {state_context_feature}", flush=True)
    print(f"  State schema: {resolved_state_schema}", flush=True)
    print(
        f"  Source MCS conditioning: {'enabled (one-hot)' if include_state_mcs else 'disabled'}",
        flush=True,
    )
    if ignore_state_mcs and "state_mcs_index" in df.columns:
        print("  Source MCS column intentionally ignored for ablation", flush=True)
    train_states = np.array(
        [
            build_state_vector(
                row,
                state_context_feature,
                include_state_mcs,
                resolved_state_schema,
            )
            for row in train_rows
        ],
        dtype=np.float32,
    )
    eval_states = np.array(
        [
            build_state_vector(
                row,
                state_context_feature,
                include_state_mcs,
                resolved_state_schema,
            )
            for row in eval_rows
        ],
        dtype=np.float32,
    )
    state_dim = int(train_states.shape[1]) if len(train_states) else len(
        state_feature_names(resolved_state_schema, include_state_mcs)
    )
    print(f"  State dimensions: {state_dim}", flush=True)
    train_actions = train_df["mcs_index"].to_numpy(dtype=np.int64)
    train_rewards = train_df["reward"].to_numpy(dtype=np.float32)
    train_next_indices, train_dones = build_temporal_next_indices(train_df)
    temporal_transitions = int((train_dones == 0.0).sum())
    train_model_augmented = pd.to_numeric(
        train_df.get("model_augmented", pd.Series(0, index=train_df.index)),
        errors="coerce",
    ).fillna(0).astype(int).to_numpy()
    modeled_temporal = int(
        ((train_model_augmented == 1) & (train_dones == 0.0)).sum()
    )
    print(
        f"  Temporal transitions: {temporal_transitions}, terminal rows: {int((train_dones == 1.0).sum())}",
        flush=True,
    )
    if int((train_model_augmented == 1).sum()):
        print(
            f"  Model-augmented train rows: {int((train_model_augmented == 1).sum())}, "
            f"linked modeled transitions: {modeled_temporal}",
            flush=True,
        )
    diagnostic_train_df = train_df.loc[train_model_augmented == 0]
    empirical_reward_by_mcs = (
        diagnostic_train_df.groupby("mcs_index")["reward"].mean().sort_index()
    )
    best_mean = int(empirical_reward_by_mcs.idxmax()) if len(empirical_reward_by_mcs) else -1
    if len(empirical_reward_by_mcs):
        print("  Empirical train mean reward by MCS:", flush=True)
        for mcs_idx, mean_reward in empirical_reward_by_mcs.items():
            print(f"    MCS{int(mcs_idx)}: {mean_reward:.4f}", flush=True)
        print(f"    Best by train mean reward: MCS{best_mean}", flush=True)
    if min_reward is not None:
        clipped_count = int((train_rewards < min_reward).sum())
        train_rewards = np.maximum(train_rewards, min_reward).astype(np.float32)
        print(
            f"  Reward lower clipping: min_reward={min_reward}, "
            f"clipped_train_rows={clipped_count}",
            flush=True,
        )
    if reward_scale <= 0:
        raise ValueError("reward_scale must be positive")
    if reward_scale != 1.0:
        train_rewards = (train_rewards / reward_scale).astype(np.float32)
        print(f"  Reward scaling: train_reward = reward / {reward_scale}", flush=True)
    print(f"  Precomputed state vectors in {time.perf_counter() - precompute_start:.1f}s", flush=True)

    # Break down by SNR quintile instead of a global aggregate.
    # A global aggregate conflates channel conditions and creates a false "global best MCS"
    # impression. The quintile view exposes the inversion: high-SNR contexts favour high MCS,
    # low-SNR contexts favour low MCS. The DQN must learn this context-dependence, not a
    # single globally best action.
    _snr_col = "snr" if "snr" in diagnostic_train_df.columns else None
    if _snr_col is not None and diagnostic_train_df[_snr_col].notna().sum() > 100:
        try:
            snr_bins = pd.qcut(diagnostic_train_df[_snr_col], q=5, duplicates="drop")
            category_count = len(snr_bins.cat.categories)
            labels = [
                "Q1(low-SNR)"
                if idx == 0
                else f"Q{idx + 1}(high-SNR)"
                if idx == category_count - 1
                else f"Q{idx + 1}"
                for idx in range(category_count)
            ]
            diagnostic_train_df = diagnostic_train_df.copy()
            diagnostic_train_df["_snr_q"] = snr_bins.cat.rename_categories(labels)
            print("  Context-conditional median service_ms by MCS (SNR quintile):", flush=True)
            print("  (goal: each row should show a different best MCS — that is the context-dependence the model must learn)", flush=True)
            pivot = (
                diagnostic_train_df.groupby(
                    ["_snr_q", "mcs_index"], observed=False
                )["service_ms"]
                .median()
                .unstack("mcs_index")
            )
            pivot.columns = [f"MCS{int(c)}" for c in pivot.columns]
            for qbin, qrow in pivot.iterrows():
                valid = qrow.dropna()
                best_mcs = valid.idxmin() if not valid.empty else "n/a"
                row_str = "  ".join(f"{c}={v:.3f}" for c, v in valid.items())
                print(f"    {qbin}: {row_str}  → best: {best_mcs}", flush=True)
        except Exception as _e:
            print(f"  (context-conditional breakdown skipped: {_e})", flush=True)
    else:
        print("  (snr column unavailable or too small; skipping context-conditional breakdown)", flush=True)
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=8,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
        layer_norm=layer_norm,
        state_schema=resolved_state_schema,
        state_context_feature=state_context_feature,
        include_state_mcs=include_state_mcs,
        lr=lr,
        weight_decay=weight_decay,
        gamma=gamma,
        cql_alpha=cql_alpha,
        q_target_clip=q_target_clip,
        q_value_regularization=q_value_regularization,
        loss_type=loss_type,
        epsilon_start=0.1,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=buffer_size,
        device=device,
    )

    tensorboard_writer = create_tensorboard_writer(tensorboard_dir)
    if tensorboard_writer is not None:
        tensorboard_config = {
            "dataset": str(dataset_csv),
            "output_model": str(output_model),
            "train_rows": len(train_df),
            "eval_rows": len(eval_df),
            "state_dim": state_dim,
            "state_schema": resolved_state_schema,
            "epochs": epochs,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "train_every": train_every,
            "max_train_steps": max_train_steps,
            "gamma": gamma,
            "cql_alpha": cql_alpha,
            "q_target_clip": q_target_clip,
            "q_value_regularization": q_value_regularization,
            "loss": loss_type,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "dropout": dropout,
            "layer_norm": layer_norm,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "normalize_states": normalize_states,
            "seed": seed,
            "device": device,
        }
        tensorboard_writer.add_text(
            "run/config",
            "```json\n" + json.dumps(tensorboard_config, indent=2) + "\n```",
            0,
        )
        tensorboard_writer.add_graph(
            agent.q_net, torch.zeros((1, state_dim), dtype=torch.float32, device=device)
        )
        tensorboard_writer.add_scalar("dataset/train_rows", len(train_df), 0)
        tensorboard_writer.add_scalar("dataset/eval_rows", len(eval_df), 0)
        print(f"  TensorBoard logs: {Path(tensorboard_dir).expanduser()}", flush=True)

    if normalize_states:
        state_mean = train_states.mean(axis=0)
        state_std = train_states.std(axis=0)
        agent.set_state_normalizer(state_mean, state_std)
        train_states = agent.normalize_states(train_states).astype(np.float32)
        eval_states = agent.normalize_states(eval_states).astype(np.float32)
        print("  State normalization: enabled", flush=True)
    else:
        print("  State normalization: disabled", flush=True)
    train_next_states = train_states[train_next_indices] if len(train_states) else train_states
    
    # Training loop
    print(f"  Starting training for {epochs} epochs...", flush=True)
    print(
        f"  Q-network: hidden_dim={hidden_dim}, hidden_layers={hidden_layers}, "
        f"dropout={dropout}, layer_norm={layer_norm}",
        flush=True,
    )
    print(f"  Train cadence: optimizer step every {train_every} transition(s)", flush=True)
    print(
        f"  Gamma: {gamma}, CQL alpha: {cql_alpha}, "
        f"Q target clip: {q_target_clip}, "
        f"Q value regularization: {q_value_regularization}, "
        f"loss: {loss_type}, lr: {lr}, weight_decay: {weight_decay}",
        flush=True,
    )
    if max_train_steps is not None:
        print(f"  Max transitions per epoch: {max_train_steps}", flush=True)
    cumulative_loss = 0.0
    num_losses = 0
    train_size = len(train_states)
    
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        # Shuffle indices; states/actions/rewards stay in contiguous arrays.
        shuffled_idx = np.random.permutation(train_size)
        
        epoch_loss = 0.0
        epoch_count = 0
        
        # Experience replay: iterate through precomputed arrays.
        for step, idx in enumerate(shuffled_idx):
            if max_train_steps is not None and step >= max_train_steps:
                break

            state = train_states[idx]
            action = int(train_actions[idx])
            reward = float(train_rewards[idx])
            next_state = train_next_states[idx]
            done = bool(train_dones[idx])
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train on batch every N transitions to reduce wall-clock time.
            if (step + 1) % train_every == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    epoch_loss += loss
                    epoch_count += 1
                    cumulative_loss += loss
                    num_losses += 1
        
        agent.update_target_network()
        agent.decay_epsilon()
        
        avg_epoch_loss = epoch_loss / max(epoch_count, 1)
        agent.training_log["episodes"].append(epoch)
        agent.training_log["losses"].append(avg_epoch_loss)
        epoch_number = epoch + 1
        epoch_seconds = time.perf_counter() - epoch_start

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("train/loss", avg_epoch_loss, epoch_number)
            tensorboard_writer.add_scalar(
                "train/epsilon", agent.epsilon, epoch_number
            )
            tensorboard_writer.add_scalar(
                "train/epoch_seconds", epoch_seconds, epoch_number
            )
            tensorboard_writer.add_scalar(
                "train/optimizer_updates", epoch_count, epoch_number
            )
            tensorboard_writer.add_scalar(
                "train/replay_buffer_size", len(agent.replay_buffer), epoch_number
            )
            tensorboard_writer.add_scalar(
                "train/learning_rate",
                agent.optimizer.param_groups[0]["lr"],
                epoch_number,
            )
            if (
                tensorboard_histograms_every > 0
                and epoch_number % tensorboard_histograms_every == 0
            ):
                log_model_histograms(
                    tensorboard_writer, agent.q_net, epoch_number
                )
        
        print(
            f"  Epoch {epoch_number}/{epochs}, "
            f"steps={min(train_size, max_train_steps or train_size)}, "
            f"updates={epoch_count}, "
            f"loss={avg_epoch_loss:.4f}, "
            f"epsilon={agent.epsilon:.4f}, "
            f"time={epoch_seconds:.1f}s",
            flush=True,
        )
    
    # Evaluate on test set
    eval_start = time.perf_counter()
    print(f"  Evaluating on {len(eval_df)} samples...", flush=True)
    q_values_list = []
    agent.q_net.eval()
    
    with torch.no_grad():
        eval_tensor = torch.from_numpy(eval_states).float().to(device)
        q_values_list = agent.q_net(eval_tensor).cpu().numpy()
    
    q_values_array = np.array(q_values_list)
    avg_q_value = np.mean(q_values_array)
    policy_actions = np.argmax(q_values_array, axis=1)
    policy_counts = pd.Series(policy_actions).value_counts().sort_index()
    
    print(f"  Evaluation complete in {time.perf_counter() - eval_start:.1f}s:", flush=True)
    print(f"    Mean Q-value: {avg_q_value:.4f}", flush=True)
    print(f"    Q-value range: [{q_values_array.min():.4f}, {q_values_array.max():.4f}]", flush=True)
    if (
        q_target_clip is not None
        and float(np.max(np.abs(q_values_array))) > q_target_clip * 1.5
    ):
        print(
            "    WARNING: learned Q-values materially exceed the configured "
            f"target bound +/-{q_target_clip}; treat this checkpoint as unstable",
            flush=True,
        )
    print("    Mean Q-value by action:", flush=True)
    for mcs_idx, mean_q in enumerate(q_values_array.mean(axis=0)):
        print(f"      MCS{mcs_idx}: {mean_q:.4f}", flush=True)
    print("    Greedy policy distribution on eval split:", flush=True)
    for mcs_idx, count in policy_counts.items():
        pct = count / len(eval_df) * 100
        print(f"      MCS{int(mcs_idx)}: {int(count)} ({pct:.1f}%)", flush=True)
    if len(policy_counts) and policy_counts.idxmax() != best_mean:
        print(
            f"    WARNING: Most common eval action MCS{int(policy_counts.idxmax())} "
            f"differs from empirical best mean-reward action MCS{best_mean}",
            flush=True,
        )

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar("eval/q_value_mean", avg_q_value, epochs)
        tensorboard_writer.add_scalar(
            "eval/q_value_min", float(q_values_array.min()), epochs
        )
        tensorboard_writer.add_scalar(
            "eval/q_value_max", float(q_values_array.max()), epochs
        )
        tensorboard_writer.add_histogram(
            "eval/q_values/all_actions", q_values_array, epochs
        )
        policy_rows = ["| Action | Mean Q-value | Greedy policy share |", "|---|---:|---:|"]
        for mcs_idx in range(agent.action_dim):
            mean_q = float(q_values_array[:, mcs_idx].mean())
            action_share = float((policy_actions == mcs_idx).mean())
            tensorboard_writer.add_scalar(
                f"eval/mean_q_by_action/MCS{mcs_idx}", mean_q, epochs
            )
            tensorboard_writer.add_scalar(
                f"eval/policy_share/MCS{mcs_idx}", action_share, epochs
            )
            tensorboard_writer.add_histogram(
                f"eval/q_value_distribution/MCS{mcs_idx}",
                q_values_array[:, mcs_idx],
                epochs,
            )
            policy_rows.append(
                f"| MCS{mcs_idx} | {mean_q:.4f} | {action_share:.2%} |"
            )
        tensorboard_writer.add_text(
            "eval/action_summary", "\n".join(policy_rows), epochs
        )
        tensorboard_writer.flush()
        tensorboard_writer.close()
    
    # Save model
    agent.save_model(output_model)
    
    # Save training log
    log_path = Path(output_model).parent / "dqn_training_log.json"
    with open(log_path, "w") as f:
        json.dump(agent.training_log, f, indent=2)
    print(f"  Training log saved to: {log_path}", flush=True)
    
    print(f"✅ DQN training complete in {time.perf_counter() - run_start:.1f}s!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN for MCS selection")
    parser.add_argument(
        "--dataset", default="rl_dqn_dataset.csv",
        help="Path to DQN dataset"
    )
    parser.add_argument(
        "--output", default="dqn_mcs_model.pth",
        help="Path to save trained model"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for experience replay"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=10000,
        help="Size of experience replay buffer"
    )
    parser.add_argument(
        "--eval-split", type=float, default=0.2,
        help="Fraction of data for evaluation"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--train-every", type=int, default=1,
        help="Run optimizer step every N transitions (e.g., 4 for faster training)"
    )
    parser.add_argument(
        "--max-train-steps", type=int, default=None,
        help="Optional cap on transitions processed per epoch"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.9,
        help="Discount factor for temporal offline Q-learning"
    )
    parser.add_argument(
        "--cql-alpha", type=float, default=0.0,
        help="Conservative Q penalty weight (higher = less optimistic on unseen actions)"
    )
    parser.add_argument(
        "--q-target-clip",
        type=float,
        default=None,
        help="Optional symmetric Bellman-target bound, e.g. 10 for rewards in [-1, 1]",
    )
    parser.add_argument(
        "--q-value-regularization",
        type=float,
        default=0.0,
        help=(
            "Optional penalty weight for Q outputs outside +/-q-target-clip. "
            "Useful when targets are bounded but unselected actions spike."
        ),
    )
    parser.add_argument(
        "--loss",
        choices=["mse", "huber"],
        default="mse",
        help="TD loss function"
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=None,
        help="Optional lower bound for reward targets, e.g. -10 for delay rewards"
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Divide reward targets by this value to improve numerical stability"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Width of each hidden layer in the Q-network"
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=2,
        help="Number of hidden layers in the Q-network"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability between hidden layers"
    )
    parser.add_argument(
        "--layer-norm",
        action="store_true",
        help="Add LayerNorm after each hidden linear layer"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam weight decay / L2 regularization"
    )
    parser.add_argument(
        "--no-normalize-states",
        action="store_true",
        help="Disable train-split state standardization"
    )
    parser.add_argument(
        "--ignore-state-mcs",
        action="store_true",
        help=(
            "Ignore state_mcs_index in a causal dataset, producing a CSI-only "
            "state for source-MCS ablation"
        ),
    )
    parser.add_argument(
        "--state-schema",
        choices=["auto", "legacy_v1", "link_v2", "link_v3", "link_v3c", "link_v4", "link_v5", "link_v6", "link_v7c"],
        default="auto",
        help=(
            "State feature schema. auto uses link_v4 when ACK-history columns "
            "are present, otherwise link_v2 when packet-gap columns are present. "
            "link_v4 log-transforms unbounded packet/loss counters. link_v5 "
            "is receiver-only and excludes ACK-history features. link_v6 adds "
            "unwrapped linear-detrended phase residuals to link_v5. link_v3c "
            "is link_v2 with the compact 57-amplitude CSI layout. link_v7c "
            "uses the versioned robust full-CSI contract (explicit only)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help="Optional directory for TensorBoard event files"
    )
    parser.add_argument(
        "--tensorboard-histograms-every",
        type=int,
        default=0,
        help="Log parameter/gradient histograms every N epochs (0 disables)"
    )
    
    args = parser.parse_args()
    
    train_dqn(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        eval_split=args.eval_split,
        device=args.device,
        train_every=args.train_every,
        max_train_steps=args.max_train_steps,
        gamma=args.gamma,
        cql_alpha=args.cql_alpha,
        q_target_clip=args.q_target_clip,
        q_value_regularization=args.q_value_regularization,
        loss_type=args.loss,
        min_reward=args.min_reward,
        reward_scale=args.reward_scale,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
        layer_norm=args.layer_norm,
        lr=args.lr,
        weight_decay=args.weight_decay,
        normalize_states=not args.no_normalize_states,
        ignore_state_mcs=args.ignore_state_mcs,
        state_schema=args.state_schema,
        seed=args.seed,
        tensorboard_dir=args.tensorboard_dir,
        tensorboard_histograms_every=args.tensorboard_histograms_every,
    )
