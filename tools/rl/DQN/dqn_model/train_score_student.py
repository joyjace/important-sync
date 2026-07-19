#!/usr/bin/env python3
"""Train a DQN-shaped receiver-only student from an action-reward teacher.

The student is an 8-output QNetwork so it can reuse the DQN firmware exporter,
but the objective is supervised distillation:

    receiver state -> teacher score for MCS0..MCS7

No Bellman targets, replay bootstrapping, ACK-history features, or source-MCS
shortcuts are required.
"""

from __future__ import annotations

import argparse
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

SCRIPT_DIR = Path(__file__).resolve().parent
DQN_ROOT = SCRIPT_DIR.parent
REWARD_MODEL_DIR = DQN_ROOT / "action_reward_model"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REWARD_MODEL_DIR))

from predict_reward_model import RewardModel
from train_dqn import (
    QNetwork,
    build_state_vector,
    feature_contract_metadata,
    resolve_state_schema_for_columns,
    state_feature_names,
    validate_dataset_feature_contract,
)
from train_reward_model import parse_iq_raw


def normalize(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (states - mean) / std


def force_teacher_fresh(frame: pd.DataFrame) -> pd.DataFrame:
    fresh = frame.copy()
    fresh["state_age_packets"] = 1
    fresh["state_packet_gap"] = 1
    fresh["state_missing_packets"] = 0
    fresh["state_is_stale"] = 0
    return fresh


def build_states(
    frame: pd.DataFrame,
    state_context_feature: str,
    include_state_mcs: bool,
    state_schema: str,
) -> np.ndarray:
    rows = list(frame.itertuples(index=False))
    return np.asarray(
        [
            build_state_vector(
                row,
                state_context_feature,
                include_state_mcs,
                state_schema,
            )
            for row in rows
        ],
        dtype=np.float32,
    )


def action_distribution(scores: np.ndarray) -> dict[str, int]:
    actions = np.argmax(scores, axis=1)
    return {
        f"MCS{idx}": int((actions == idx).sum())
        for idx in range(scores.shape[1])
        if int((actions == idx).sum()) > 0
    }


def train_score_student(
    dataset_csv: Path,
    teacher_model_path: Path,
    output_model: Path,
    state_schema: str = "link_v5",
    ignore_state_mcs: bool = True,
    teacher_state_mode: str = "fresh",
    epochs: int = 15,
    batch_size: int = 512,
    eval_split: float = 0.2,
    max_rows: int | None = None,
    hidden_dim: int = 128,
    hidden_layers: int = 2,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    loss_type: str = "huber",
    ce_class_weights: str = "inverse_sqrt",
    device: str = "cpu",
    seed: int = 42,
) -> None:
    start = time.perf_counter()
    print("[DQN-Shaped Score Student Training]", flush=True)
    print(f"  Dataset: {dataset_csv}", flush=True)
    print(f"  Teacher: {teacher_model_path}", flush=True)
    print(f"  Output: {output_model}", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"  Seed: {seed}", flush=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    frame = pd.read_csv(dataset_csv)
    if max_rows is not None and max_rows < len(frame):
        frame = frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        print(f"  Sampled rows: {len(frame)}", flush=True)
    else:
        frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if frame.empty:
        raise ValueError("dataset is empty")
    if "iq_raw" not in frame.columns:
        raise ValueError("dataset must contain iq_raw")
    frame["iq_raw_parsed"] = frame["iq_raw"].apply(parse_iq_raw)

    teacher = RewardModel.load(teacher_model_path, device=device)
    if teacher.action_dim != 8:
        raise ValueError(f"teacher must score 8 actions, got {teacher.action_dim}")
    if teacher.objective != "maximize":
        raise ValueError("score student expects a reward/maximize teacher")
    if teacher.include_state_mcs:
        raise ValueError("teacher must be trained without source-MCS conditioning")
    validate_dataset_feature_contract(dataset_csv, teacher.state_schema)

    requested_schema = resolve_state_schema_for_columns(frame.columns, state_schema)
    validate_dataset_feature_contract(dataset_csv, requested_schema)
    state_context_feature = "state_age_packets" if "state_age_packets" in frame.columns else "sig_len"
    include_state_mcs = "state_mcs_index" in frame.columns and not ignore_state_mcs
    print(f"  Student state schema: {requested_schema}", flush=True)
    print(f"  Student source MCS conditioning: {include_state_mcs}", flush=True)
    print(f"  Teacher state mode: {teacher_state_mode}", flush=True)

    split_idx = int(len(frame) * (1 - eval_split))
    train_df = frame.iloc[:split_idx].reset_index(drop=True)
    eval_df = frame.iloc[split_idx:].reset_index(drop=True)
    print(f"  Train rows: {len(train_df)}, Eval rows: {len(eval_df)}", flush=True)

    precompute_start = time.perf_counter()
    train_states = build_states(
        train_df,
        state_context_feature,
        include_state_mcs,
        requested_schema,
    )
    eval_states = build_states(
        eval_df,
        state_context_feature,
        include_state_mcs,
        requested_schema,
    )

    train_teacher_df = force_teacher_fresh(train_df) if teacher_state_mode == "fresh" else train_df
    eval_teacher_df = force_teacher_fresh(eval_df) if teacher_state_mode == "fresh" else eval_df
    train_teacher_states = build_states(
        train_teacher_df,
        teacher.state_context_feature,
        teacher.include_state_mcs,
        teacher.state_schema,
    )
    eval_teacher_states = build_states(
        eval_teacher_df,
        teacher.state_context_feature,
        teacher.include_state_mcs,
        teacher.state_schema,
    )
    train_targets = teacher.score_all_actions(train_teacher_states, batch_size=4096).astype(np.float32)
    eval_targets = teacher.score_all_actions(eval_teacher_states, batch_size=4096).astype(np.float32)

    state_dim = int(train_states.shape[1])
    if len(eval_states) and eval_states.shape[1] != state_dim:
        raise ValueError("train/eval state dimensions differ")
    print(f"  State dimensions: {state_dim}", flush=True)
    print(
        f"  Teacher target range: [{float(train_targets.min()):.4f}, "
        f"{float(train_targets.max()):.4f}]",
        flush=True,
    )
    print(f"  Teacher train argmax distribution: {action_distribution(train_targets)}", flush=True)
    print(f"  Precomputed states/targets in {time.perf_counter() - precompute_start:.1f}s", flush=True)

    state_mean = train_states.mean(axis=0).astype(np.float32)
    state_std = train_states.std(axis=0).astype(np.float32)
    state_std = np.where(state_std < 1e-6, 1.0, state_std).astype(np.float32)
    train_states = normalize(train_states, state_mean, state_std).astype(np.float32)
    eval_states = normalize(eval_states, state_mean, state_std).astype(np.float32)

    train_teacher_actions = np.argmax(train_targets, axis=1).astype(np.int64)
    eval_teacher_actions = np.argmax(eval_targets, axis=1).astype(np.int64)

    train_dataset_tensors = [torch.from_numpy(train_states)]
    if loss_type == "ce":
        train_dataset_tensors.append(torch.from_numpy(train_teacher_actions))
    else:
        train_dataset_tensors.append(torch.from_numpy(train_targets))
    train_loader = DataLoader(
        TensorDataset(*train_dataset_tensors),
        batch_size=batch_size,
        shuffle=True,
    )
    eval_tensor = torch.from_numpy(eval_states).float().to(device)
    eval_target_tensor = torch.from_numpy(eval_targets).float().to(device)
    eval_action_tensor = torch.from_numpy(eval_teacher_actions).long().to(device)

    model = QNetwork(
        state_dim=state_dim,
        action_dim=8,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        dropout=0.0,
        layer_norm=False,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if loss_type == "huber":
        criterion: nn.Module = nn.SmoothL1Loss()
    elif loss_type == "mse":
        criterion = nn.MSELoss()
    else:
        class_weights = None
        if ce_class_weights != "none":
            counts = np.bincount(train_teacher_actions, minlength=8).astype(np.float32)
            counts = np.maximum(counts, 1.0)
            if ce_class_weights == "inverse":
                weights = 1.0 / counts
            else:
                weights = 1.0 / np.sqrt(counts)
            weights = weights / weights.mean()
            class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
            print(
                "  CE class weights: "
                + " ".join(f"MCS{i}={float(weights[i]):.3f}" for i in range(8)),
                flush=True,
            )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    log = {
        "epochs": [],
        "train_losses": [],
        "eval_losses": [],
        "eval_argmax_agreement": [],
    }
    best_loss = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        for states_batch, targets_batch in train_loader:
            states_batch = states_batch.float().to(device)
            optimizer.zero_grad()
            predictions = model(states_batch)
            if loss_type == "ce":
                targets_batch = targets_batch.long().to(device)
                loss = criterion(predictions, targets_batch)
            else:
                targets_batch = targets_batch.float().to(device)
                loss = criterion(predictions, targets_batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            eval_pred = model(eval_tensor)
            if loss_type == "ce":
                eval_loss = float(criterion(eval_pred, eval_action_tensor).detach().cpu())
            else:
                eval_loss = float(criterion(eval_pred, eval_target_tensor).detach().cpu())
            pred_actions = torch.argmax(eval_pred, dim=1).cpu().numpy()
            agreement = float(np.mean(pred_actions == eval_teacher_actions))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
        log["epochs"].append(epoch)
        log["train_losses"].append(train_loss)
        log["eval_losses"].append(eval_loss)
        log["eval_argmax_agreement"].append(agreement)
        print(
            f"  Epoch {epoch + 1}/{epochs}, train_loss={train_loss:.5f}, "
            f"eval_loss={eval_loss:.5f}, eval_argmax_agreement={agreement:.1%}, "
            f"time={time.perf_counter() - epoch_start:.1f}s",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best checkpoint from epoch {best_epoch + 1}", flush=True)

    output_model.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "q_net_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "state_dim": state_dim,
        "action_dim": 8,
        "hidden_dim": hidden_dim,
        "hidden_layers": hidden_layers,
        "dropout": 0.0,
        "layer_norm": False,
        "state_schema": requested_schema,
        "state_context_feature": state_context_feature,
        "include_state_mcs": include_state_mcs,
        "state_feature_names": state_feature_names(requested_schema, include_state_mcs),
        **feature_contract_metadata(requested_schema),
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "lr": lr,
        "gamma": 0.0,
        "cql_alpha": 0.0,
        "student_training": "action_reward_distillation",
        "student_loss": loss_type,
        "ce_class_weights": ce_class_weights if loss_type == "ce" else "none",
        "teacher_model": str(teacher_model_path),
        "teacher_state_mode": teacher_state_mode,
        "training_log": log,
        "best_epoch": best_epoch,
        "best_eval_loss": best_loss,
    }
    torch.save(checkpoint, output_model)
    log_path = output_model.parent / "score_student_training_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"  Model saved to: {output_model}", flush=True)
    print(f"  Training log saved to: {log_path}", flush=True)
    print(f"✅ Score student training complete in {time.perf_counter() - start:.1f}s!", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DQN-shaped score student from reward model")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--state-schema",
        choices=["auto", "legacy_v1", "link_v2", "link_v3", "link_v4", "link_v5", "link_v6", "link_v7c"],
        default="link_v5",
    )
    parser.add_argument("--ignore-state-mcs", action="store_true", default=False)
    parser.add_argument(
        "--teacher-state-mode",
        choices=["fresh", "as_is"],
        default="fresh",
        help="Use fresh to score teacher targets with age/gap/missing reset to fresh CSI",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-split", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["mse", "huber", "ce"], default="huber")
    parser.add_argument(
        "--ce-class-weights",
        choices=["none", "inverse_sqrt", "inverse"],
        default="inverse_sqrt",
        help="Class weighting for --loss ce",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_score_student(
        dataset_csv=args.dataset,
        teacher_model_path=args.teacher_model,
        output_model=args.output,
        state_schema=args.state_schema,
        ignore_state_mcs=args.ignore_state_mcs,
        teacher_state_mode=args.teacher_state_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_split=args.eval_split,
        max_rows=args.max_rows,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss,
        ce_class_weights=args.ce_class_weights,
        device=args.device,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
