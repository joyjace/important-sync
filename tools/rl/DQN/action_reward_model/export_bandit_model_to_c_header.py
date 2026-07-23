#!/usr/bin/env python3
"""
Export a two-head bandit checkpoint to a standalone C header.

The generated header contains the weights AND a self-contained static inline
inference implementation (bandit_model_score_actions / _predict_best_mcs), so
firmware integration is: include the header, fill the state array with the
same layout as training (see BANDIT_MODEL_* macros), call predict.

The deployed policy objective is the dataset-comparable bounded utility:
    u(a) = p(a) * success_utility(delay(a)) + (1 - p(a)) * loss_reward
with the utility constants baked in from the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DQN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DQN_ROOT.parents[2]
sys.path.insert(0, str(DQN_ROOT / "dqn_model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dqn import (
    build_state_vector,
    feature_contract_metadata,
    schema_amplitude_count,
    state_feature_names,
    validate_dataset_feature_contract,
)
from train_reward_model import parse_iq_raw
from train_bandit_model import (
    ACTION_DIM,
    FAIR_CHECKPOINT_SCHEMA,
    FAIR_OBJECTIVE_CONTRACT,
    FAIR_REQUIRED_CSI_INPUT_SCALAR_COUNT,
)
from predict_bandit_model import load_bandit_checkpoint


DEPLOYMENT_RECORD_SCHEMA = "bandit_firmware_deployment/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def portable_path(path: Path) -> str:
    """Use repository-relative paths so deployment records are reproducible."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def reject_output_collisions(
    output_path: Path,
    deployment_record_path: Path,
    protected_paths: list[Path],
) -> None:
    output = output_path.resolve()
    record = deployment_record_path.resolve()
    if output == record:
        raise ValueError("--output and --deployment-record must be different files")
    protected = {path.resolve() for path in protected_paths}
    for candidate, label in ((output, "header output"), (record, "deployment record")):
        if candidate in protected:
            raise ValueError(f"Bandit {label} collides with a protected input/seal")


def validate_fair_deployment_authorization(
    model_path: Path,
    verify_dataset: Path,
    selection_path: Path,
    qualification_complete_path: Path,
) -> dict:
    """Bind export to the validation-selected, qualification-sealed candidate."""

    model_path = model_path.resolve()
    verify_dataset = verify_dataset.resolve()
    selection_path = selection_path.resolve()
    qualification_complete_path = qualification_complete_path.resolve()
    selection = read_json_object(selection_path)
    if selection.get("schema") != "v3_3_matched_bandit_candidate_selection/v1":
        raise ValueError("Bandit candidate-selection schema is not deployable")
    if selection.get("selection_data") != "VALIDATION only":
        raise ValueError("Bandit deployment requires validation-only candidate selection")
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("Bandit candidate selection has no selected checkpoint")
    model_sha = sha256_file(model_path)
    selected_seed = int(selected.get("seed", -1))
    selected_suffix = Path("models") / f"seed_{selected_seed}" / "bandit_model.pth"
    stored_selected_path = Path(str(selected.get("path", ""))).expanduser()
    stored_path_matches = stored_selected_path.resolve() == model_path
    suffix_matches = (
        tuple(stored_selected_path.parts[-len(selected_suffix.parts) :])
        == selected_suffix.parts
        and (selection_path.parent / selected_suffix).resolve() == model_path
    )
    if (
        not (stored_path_matches or suffix_matches)
        or selected.get("sha256") != model_sha
    ):
        raise ValueError("Export model is not the validation-selected bandit checkpoint")

    experiment_manifest_path = selection_path.parent / "manifest.json"
    experiment_manifest = read_json_object(experiment_manifest_path)
    if experiment_manifest.get("schema") != "v3_3_matched_bandit_plan/v1":
        raise ValueError("Bandit training manifest schema is not deployable")
    validation = experiment_manifest.get("datasets", {}).get("validation", {})
    if validation.get("sha256") != sha256_file(verify_dataset):
        raise ValueError(
            "--verify-dataset must be the sealed v3.3 validation artifact"
        )

    qualification = read_json_object(qualification_complete_path)
    if (
        qualification.get("schema")
        != "v3_3_matched_bandit_qualification_complete/v1"
        or qualification.get("selection_unchanged_by_qualification") is not True
    ):
        raise ValueError("Bandit qualification completion seal is invalid")
    selection_sha = sha256_file(selection_path)
    if (
        qualification.get("candidate_selection_sha256") != selection_sha
        or int(qualification.get("preselected_seed", -1))
        != int(selected.get("seed", -2))
    ):
        raise ValueError("Qualification does not bind the selected bandit candidate")

    qualification_manifest_path = (
        qualification_complete_path.parent / "qualification_manifest.json"
    )
    qualification_manifest = read_json_object(qualification_manifest_path)
    if (
        qualification_manifest.get("schema")
        != "v3_3_matched_bandit_qualification_plan/v1"
    ):
        raise ValueError("Bandit qualification manifest schema is invalid")
    canonical_plan = dict(qualification_manifest)
    canonical_plan.pop("plan_sha256", None)
    canonical_plan.pop("created_utc", None)
    if (
        qualification_manifest.get("plan_sha256")
        != qualification.get("plan_sha256")
        or canonical_sha256(canonical_plan) != qualification.get("plan_sha256")
        or qualification_manifest.get("candidate_selection", {}).get("sha256")
        != selection_sha
        or int(
            qualification_manifest.get("candidate_selection", {}).get(
                "selected_seed", -1
            )
        )
        != int(selected["seed"])
        or qualification_manifest.get("bandit_manifest", {}).get("sha256")
        != sha256_file(experiment_manifest_path)
    ):
        raise ValueError("Qualification plan does not bind the training/selection seals")

    comparison_path = qualification_complete_path.parent / "comparison_summary.json"
    if qualification.get("comparison_summary_sha256") != sha256_file(comparison_path):
        raise ValueError("Qualification comparison summary differs from its seal")
    comparison = read_json_object(comparison_path)
    if (
        comparison.get("schema") != "v3_3_bandit_model_comparison/v1"
        or int(comparison.get("preselected_seed", -1)) != int(selected["seed"])
        or comparison.get("qualification_must_not_reselect_seed") is not True
        or comparison.get("provenance", {}).get("candidate_selection_sha256")
        != selection_sha
    ):
        raise ValueError("Qualification comparison does not bind the selected seed")
    if (
        int(qualification.get("prediction_count", -1)) != 27
        or int(qualification.get("evaluation_count", -1)) != 42
        or qualification.get("row_identity_reference")
        != "existing amplitude reward-model predictions"
    ):
        raise ValueError("Qualification completion counts/identity are incomplete")

    selected_comparison_path = (
        qualification_complete_path.parent / "selected_candidate_comparison.json"
    )
    if (
        qualification.get("selected_candidate_comparison_sha256")
        != sha256_file(selected_comparison_path)
        or qualification.get("dqn_reused_without_execution_or_reselection") is not True
    ):
        raise ValueError(
            "Qualification does not seal the selected bandit-vs-DQN comparison"
        )
    selected_comparison = read_json_object(selected_comparison_path)
    dqn_seed = int(qualification.get("preselected_dqn_seed", -1))
    candidate_ids = {
        (str(candidate.get("policy")), int(candidate.get("seed", -1)))
        for candidate in selected_comparison.get("candidates", [])
        if isinstance(candidate, dict)
    }
    if (
        selected_comparison.get("schema")
        != "v3_3_preselected_bandit_dqn_comparison/v1"
        or selected_comparison.get("headline_scope")
        != "validation_preselected_candidates_only"
        or selected_comparison.get(
            "qualification_does_not_select_or_replace_either_candidate"
        )
        is not True
        or ("bandit_two_head", int(selected["seed"])) not in candidate_ids
        or ("dqn_gamma0", dqn_seed) not in candidate_ids
        or int(selected_comparison.get("dqn_reuse", {}).get("selected_seed", -1))
        != dqn_seed
        or selected_comparison.get("dqn_reuse", {}).get("reuse_contract")
        != "sealed_summary_only_no_dqn_execution_or_reselection/v1"
    ):
        raise ValueError(
            "Selected-candidate comparison does not bind both preselected policies"
        )

    grid_path = selection_path.parent / "grid_complete.json"
    grid_sha = sha256_file(grid_path)
    if (
        qualification_manifest.get("bandit_grid_complete", {}).get("sha256")
        != grid_sha
        or qualification.get("bandit_grid_complete_sha256") != grid_sha
    ):
        raise ValueError("Qualification does not bind the sealed bandit grid")
    return {
        "selected_seed": int(selected["seed"]),
        "preselected_dqn_seed": dqn_seed,
        "model_sha256": model_sha,
        "qualification_completed_utc": qualification.get("completed_utc"),
        "candidate_selection": {
            "path": portable_path(selection_path),
            "sha256": selection_sha,
        },
        "training_manifest": {
            "path": portable_path(experiment_manifest_path),
            "sha256": sha256_file(experiment_manifest_path),
        },
        "qualification_manifest": {
            "path": portable_path(qualification_manifest_path),
            "sha256": sha256_file(qualification_manifest_path),
            "plan_sha256": qualification.get("plan_sha256"),
        },
        "qualification_complete": {
            "path": portable_path(qualification_complete_path),
            "sha256": sha256_file(qualification_complete_path),
        },
        "grid_complete": {
            "path": portable_path(grid_path),
            "sha256": grid_sha,
        },
        "comparison_summary": {
            "path": portable_path(comparison_path),
            "sha256": sha256_file(comparison_path),
        },
        "selected_candidate_comparison": {
            "path": portable_path(selected_comparison_path),
            "sha256": sha256_file(selected_comparison_path),
        },
        "verification_dataset": {
            "path": portable_path(verify_dataset),
            "sha256": sha256_file(verify_dataset),
        },
    }


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def format_float_array(name: str, array: np.ndarray, per_line: int = 8) -> str:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    lines = [f"static const float {name}[{flat.size}] = {{"]
    for index in range(0, flat.size, per_line):
        values = ", ".join(f"{float(v):.8e}f" for v in flat[index : index + per_line])
        lines.append(f"    {values},")
    lines.append("};")
    return "\n".join(lines)


def load_export_values(model_path: Path) -> dict:
    _model, checkpoint = load_bandit_checkpoint(model_path, "cpu")
    state = checkpoint["model_state"]

    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint.get("action_dim", ACTION_DIM))
    action_feature_dim = int(checkpoint.get("action_feature_dim", 9))
    hidden_dim = int(checkpoint.get("hidden_dim", 128))
    input_dim = state_dim + action_feature_dim
    utility = checkpoint["utility_params"]
    checkpoint_schema = str(checkpoint.get("checkpoint_schema", "legacy_unspecified"))
    objective_contract = str(
        checkpoint.get("objective_contract", "legacy_unspecified")
    )
    is_fair_checkpoint = checkpoint_schema == FAIR_CHECKPOINT_SCHEMA
    training_exact_fresh_only = bool(
        checkpoint.get("training_exact_fresh_only", False)
    )

    if action_dim != ACTION_DIM:
        raise ValueError(
            f"Firmware bandit exporter requires action_dim={ACTION_DIM}, got {action_dim}"
        )
    if action_feature_dim != ACTION_DIM + 1:
        raise ValueError(
            "Firmware bandit exporter requires eight one-hot action values plus "
            f"one normalized action value, got action_feature_dim={action_feature_dim}"
        )
    if utility.get("objective", "utility") != "utility":
        raise ValueError(
            "Firmware bandit export currently implements only objective='utility'; "
            f"checkpoint objective is {utility.get('objective')!r}"
        )

    state_schema = checkpoint["state_schema"]
    include_state_mcs = bool(checkpoint["include_state_mcs"])
    if is_fair_checkpoint:
        if state_schema != "link_v5" or state_dim != 132 or include_state_mcs:
            raise ValueError(
                "Fair bandit checkpoints must use receiver-only link_v5 with 132 inputs"
            )
        if objective_contract != FAIR_OBJECTIVE_CONTRACT:
            raise ValueError(
                "Fair bandit checkpoint has an unexpected objective contract: "
                f"{objective_contract!r}"
            )
        if not training_exact_fresh_only:
            raise ValueError(
                "Fair bandit checkpoint must explicitly declare "
                "training_exact_fresh_only=True"
            )

    default_required_scalars = (
        FAIR_REQUIRED_CSI_INPUT_SCALAR_COUNT
        if is_fair_checkpoint or state_schema == "link_v7c"
        else 0
    )
    required_csi_input_scalar_count = int(
        checkpoint.get(
            "required_csi_input_scalar_count", default_required_scalars
        )
    )
    requires_valid_first_word = bool(
        checkpoint.get(
            "requires_valid_first_word",
            is_fair_checkpoint or state_schema == "link_v7c",
        )
    )
    if (is_fair_checkpoint or state_schema == "link_v7c") and (
        required_csi_input_scalar_count != FAIR_REQUIRED_CSI_INPUT_SCALAR_COUNT
        or not requires_valid_first_word
    ):
        raise ValueError(
            "Contract-bound bandit firmware input must require exactly 114 CSI "
            "scalars and a valid first word"
        )

    utility_numeric = {
        "payload_bytes": float(utility["payload_bytes"]),
        "loss_reward": float(utility["loss_reward"]),
        "utility_scale": float(utility["utility_scale"]),
        "tail_target_ms": float(utility["tail_target_ms"]),
        "tail_weight": float(utility["tail_weight"]),
    }
    if not all(np.isfinite(value) for value in utility_numeric.values()):
        raise ValueError("Bandit utility parameters must all be finite")
    if utility_numeric["payload_bytes"] <= 0.0:
        raise ValueError("Bandit utility payload_bytes must be positive")
    if utility_numeric["utility_scale"] <= 0.0:
        raise ValueError("Bandit utility_scale must be positive")
    if (
        utility_numeric["tail_target_ms"] < 0.0
        or utility_numeric["tail_weight"] < 0.0
    ):
        raise ValueError("Bandit tail target and weight must be non-negative")

    values = {
        "state_dim": state_dim,
        "action_feature_dim": action_feature_dim,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "state_schema": state_schema,
        "amplitude_count": schema_amplitude_count(state_schema),
        "include_state_mcs": include_state_mcs,
        "state_context_feature": checkpoint["state_context_feature"],
        "utility": utility,
        "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "checkpoint_schema": checkpoint_schema,
        "objective_contract": objective_contract,
        "is_fair_checkpoint": is_fair_checkpoint,
        "training_exact_fresh_only": training_exact_fresh_only,
        "required_csi_input_scalar_count": required_csi_input_scalar_count,
        "requires_valid_first_word": requires_valid_first_word,
        "state_mean": np.asarray(checkpoint["state_mean"], dtype=np.float32),
        "state_std": np.asarray(checkpoint["state_std"], dtype=np.float32),
        "w1": to_numpy(state["trunk.0.weight"]).astype(np.float32),
        "b1": to_numpy(state["trunk.0.bias"]).astype(np.float32),
        "w2": to_numpy(state["trunk.2.weight"]).astype(np.float32),
        "b2": to_numpy(state["trunk.2.bias"]).astype(np.float32),
        "delivery_w": to_numpy(state["delivery_head.weight"]).astype(np.float32).reshape(-1),
        "delivery_b": float(to_numpy(state["delivery_head.bias"]).reshape(-1)[0]),
        "delay_w": to_numpy(state["delay_head.weight"]).astype(np.float32).reshape(-1),
        "delay_b": float(to_numpy(state["delay_head.bias"]).reshape(-1)[0]),
        "checkpoint": checkpoint,
        "contract_metadata": feature_contract_metadata(state_schema),
    }

    if values["state_context_feature"] != "state_age_packets":
        raise ValueError("Firmware bandit export requires a causal checkpoint")
    expected_feature_names = state_feature_names(state_schema, include_state_mcs)
    stored_feature_names = checkpoint.get("state_feature_names")
    if stored_feature_names is not None and list(stored_feature_names) != expected_feature_names:
        raise ValueError("Checkpoint state_feature_names are out of order or invalid")
    if is_fair_checkpoint and stored_feature_names is None:
        raise ValueError("Fair bandit checkpoint is missing state_feature_names")
    for key, expected in values["contract_metadata"].items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"Checkpoint {key}={checkpoint.get(key)!r} does not match expected {expected!r}"
            )
    expected_shapes = {
        "state_mean": (state_dim,),
        "state_std": (state_dim,),
        "w1": (hidden_dim, input_dim),
        "b1": (hidden_dim,),
        "w2": (hidden_dim, hidden_dim),
        "b2": (hidden_dim,),
        "delivery_w": (hidden_dim,),
        "delay_w": (hidden_dim,),
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(f"Unexpected {name} shape {values[name].shape}; expected {expected}")
        if not np.all(np.isfinite(values[name])):
            raise ValueError(f"Checkpoint {name} contains non-finite values")
    if np.any(values["state_std"] < 1e-6):
        raise ValueError(
            "Checkpoint state_std values must be finite and at least 1e-6; "
            "the exporter will not silently repair normalization"
        )
    for name in ("delivery_b", "delay_b"):
        if not np.isfinite(values[name]):
            raise ValueError(f"Checkpoint {name} is non-finite")
    return values


def numpy_score_actions_float32(states: np.ndarray, values: dict):
    """Replicate the emitted C inference in float32, in the same order."""
    rows = states.shape[0]
    p = np.zeros((rows, ACTION_DIM), dtype=np.float32)
    mu = np.zeros((rows, ACTION_DIM), dtype=np.float32)
    normalized = ((states - values["state_mean"]) / values["state_std"]).astype(np.float32)
    for action in range(ACTION_DIM):
        one_hot = np.zeros((rows, values["action_feature_dim"]), dtype=np.float32)
        one_hot[:, action] = 1.0
        one_hot[:, ACTION_DIM] = np.float32(action / 7.0)
        inp = np.concatenate([normalized, one_hot], axis=1).astype(np.float32)
        h1 = np.maximum(inp @ values["w1"].T.astype(np.float32) + values["b1"], np.float32(0))
        h2 = np.maximum(h1 @ values["w2"].T.astype(np.float32) + values["b2"], np.float32(0))
        logit = h2 @ values["delivery_w"] + np.float32(values["delivery_b"])
        p[:, action] = 1.0 / (1.0 + np.exp(-logit, dtype=np.float32))
        mu[:, action] = h2 @ values["delay_w"] + np.float32(values["delay_b"])
    return p, mu


def numpy_policy_outputs_float32(states: np.ndarray, values: dict):
    """Replicate all emitted C outputs, including delay and utility clamps."""

    p, mu = numpy_score_actions_float32(states, values)
    utility = values["utility"]
    payload_bits = np.float32(float(utility["payload_bytes"]) * 8.0)
    loss_reward = np.float32(utility["loss_reward"])
    utility_scale = np.float32(utility["utility_scale"])
    tail_target_ms = np.float32(utility["tail_target_ms"])
    tail_weight = np.float32(utility["tail_weight"])

    with np.errstate(over="ignore", invalid="ignore"):
        delay_ms = np.exp(mu, dtype=np.float32)
    delay_ms = np.clip(
        delay_ms, np.float32(1e-3), np.float32(1e4)
    ).astype(np.float32)
    goodput_kbps = (payload_bits / delay_ms).astype(np.float32)
    success_utility = (
        np.float32(2.0)
        * np.clip(
            np.log1p(goodput_kbps, dtype=np.float32) / utility_scale,
            np.float32(0.0),
            np.float32(1.0),
        )
        - np.float32(1.0)
    ).astype(np.float32)
    if tail_target_ms > 0.0 and tail_weight > 0.0:
        tail_excess = np.maximum(
            (delay_ms - tail_target_ms) / tail_target_ms, np.float32(0.0)
        )
        success_utility = (
            success_utility
            - tail_weight
            * np.clip(tail_excess, np.float32(0.0), np.float32(1.0))
        ).astype(np.float32)
    success_utility = np.clip(
        success_utility, np.float32(-1.0), np.float32(1.0)
    ).astype(np.float32)
    scores = (
        p * success_utility + (np.float32(1.0) - p) * loss_reward
    ).astype(np.float32)
    return p, mu, delay_ms, scores


def torch_policy_outputs_float32(p: np.ndarray, mu: np.ndarray, utility: dict):
    """Independent float32 reference for the complete firmware objective."""

    p_tensor = torch.from_numpy(np.asarray(p, dtype=np.float32))
    mu_tensor = torch.from_numpy(np.asarray(mu, dtype=np.float32))
    delay_ms = torch.exp(mu_tensor).clamp(min=1e-3, max=1e4)
    goodput_kbps = np.float32(float(utility["payload_bytes"]) * 8.0) / delay_ms
    success_utility = (
        2.0
        * torch.clamp(
            torch.log1p(goodput_kbps) / np.float32(utility["utility_scale"]),
            min=0.0,
            max=1.0,
        )
        - 1.0
    )
    if float(utility["tail_target_ms"]) > 0.0 and float(utility["tail_weight"]) > 0.0:
        tail_target = np.float32(utility["tail_target_ms"])
        tail_excess = torch.clamp(
            (delay_ms - tail_target) / tail_target, min=0.0, max=1.0
        )
        success_utility = (
            success_utility
            - np.float32(utility["tail_weight"]) * tail_excess
        )
    success_utility = success_utility.clamp(min=-1.0, max=1.0)
    scores = (
        p_tensor * success_utility
        + (1.0 - p_tensor) * np.float32(utility["loss_reward"])
    )
    return delay_ms.cpu().numpy(), scores.cpu().numpy()


def verify_dataset(dataset_path: Path, values: dict, rows: int) -> None:
    validate_dataset_feature_contract(dataset_path, values["state_schema"])
    frame = pd.read_csv(dataset_path, nrows=rows)
    if frame.empty:
        raise ValueError("Export verification dataset contains no rows")
    frame["iq_raw_parsed"] = frame["iq_raw"].apply(parse_iq_raw)
    states = np.asarray(
        [
            build_state_vector(
                row,
                values["state_context_feature"],
                values["include_state_mcs"],
                values["state_schema"],
            )
            for row in frame.itertuples(index=False)
        ],
        dtype=np.float32,
    )
    if states.shape[1] != values["state_dim"]:
        raise ValueError(
            f"Verify dataset produced state dim {states.shape[1]}, expected {values['state_dim']}"
        )

    model, checkpoint = load_bandit_checkpoint(Path(values["checkpoint_path"]), "cpu")
    from train_bandit_model import score_all_actions
    from train_reward_model import normalize

    normalized = normalize(states, values["state_mean"], values["state_std"]).astype(np.float32)
    torch_p, torch_mu = score_all_actions(model, normalized, "cpu")
    numpy_p, numpy_mu, numpy_delay, numpy_utility = numpy_policy_outputs_float32(
        states, values
    )

    p_err = float(np.max(np.abs(torch_p - numpy_p)))
    mu_err = float(np.max(np.abs(torch_mu - numpy_mu)))
    torch_delay, torch_utility = torch_policy_outputs_float32(
        torch_p, torch_mu, values["utility"]
    )
    delay_rel_err = float(
        np.max(
            np.abs(torch_delay - numpy_delay)
            / np.maximum(np.abs(torch_delay), np.float32(1e-3))
        )
    )
    utility_err = float(np.max(np.abs(torch_utility - numpy_utility)))
    if not all(
        np.all(np.isfinite(array))
        for array in (
            torch_p,
            torch_mu,
            torch_delay,
            torch_utility,
            numpy_p,
            numpy_mu,
            numpy_delay,
            numpy_utility,
        )
    ):
        raise ValueError("Export verification produced non-finite policy outputs")
    action_matches = int(
        np.sum(np.argmax(torch_utility, axis=1) == np.argmax(numpy_utility, axis=1))
    )
    if (
        action_matches != len(states)
        or p_err > 1e-4
        or mu_err > 1e-3
        or delay_rel_err > 2e-3
        or utility_err > 2e-4
    ):
        raise ValueError(
            f"Export verification failed: action_matches={action_matches}/{len(states)}, "
            f"p_err={p_err:.8g}, mu_err={mu_err:.8g}, "
            f"delay_rel_err={delay_rel_err:.8g}, utility_err={utility_err:.8g}"
        )
    print(
        f"  Verification: {action_matches}/{len(states)} best actions matched, "
        f"p_err={p_err:.3g}, mu_err={mu_err:.3g}, "
        f"delay_rel_err={delay_rel_err:.3g}, utility_err={utility_err:.3g}"
    )


INFERENCE_CODE = r"""
static inline float bandit_model_sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline float bandit_model_clampf(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

/* Score all 8 candidate MCS actions for one (unnormalized) state.
 * Any of the output arrays may be NULL. */
static void bandit_model_score_actions(const float state[BANDIT_MODEL_STATE_DIM],
                                       float p_deliver_out[BANDIT_MODEL_ACTION_DIM],
                                       float delay_ms_out[BANDIT_MODEL_ACTION_DIM],
                                       float utility_out[BANDIT_MODEL_ACTION_DIM])
{
    float input[BANDIT_MODEL_INPUT_DIM];
    float hidden1[BANDIT_MODEL_HIDDEN_DIM];
    float hidden2[BANDIT_MODEL_HIDDEN_DIM];

    for (int i = 0; i < BANDIT_MODEL_STATE_DIM; ++i) {
        input[i] = (state[i] - bandit_model_state_mean[i]) / bandit_model_state_std[i];
    }

    for (int action = 0; action < BANDIT_MODEL_ACTION_DIM; ++action) {
        for (int i = 0; i < BANDIT_MODEL_ACTION_DIM; ++i) {
            input[BANDIT_MODEL_STATE_DIM + i] = (i == action) ? 1.0f : 0.0f;
        }
        input[BANDIT_MODEL_STATE_DIM + BANDIT_MODEL_ACTION_DIM] = (float)action / 7.0f;

        for (int out = 0; out < BANDIT_MODEL_HIDDEN_DIM; ++out) {
            float sum = bandit_model_b1[out];
            const int row = out * BANDIT_MODEL_INPUT_DIM;
            for (int in = 0; in < BANDIT_MODEL_INPUT_DIM; ++in) {
                sum += bandit_model_w1[row + in] * input[in];
            }
            hidden1[out] = sum > 0.0f ? sum : 0.0f;
        }
        for (int out = 0; out < BANDIT_MODEL_HIDDEN_DIM; ++out) {
            float sum = bandit_model_b2[out];
            const int row = out * BANDIT_MODEL_HIDDEN_DIM;
            for (int in = 0; in < BANDIT_MODEL_HIDDEN_DIM; ++in) {
                sum += bandit_model_w2[row + in] * hidden1[in];
            }
            hidden2[out] = sum > 0.0f ? sum : 0.0f;
        }

        float logit = BANDIT_MODEL_DELIVERY_BIAS;
        float mu = BANDIT_MODEL_DELAY_BIAS;
        for (int in = 0; in < BANDIT_MODEL_HIDDEN_DIM; ++in) {
            logit += bandit_model_delivery_w[in] * hidden2[in];
            mu += bandit_model_delay_w[in] * hidden2[in];
        }

        const float p = bandit_model_sigmoidf(logit);
        float delay_ms = expf(mu);
        delay_ms = bandit_model_clampf(delay_ms, 1e-3f, 1e4f);

        const float goodput_kbps = BANDIT_MODEL_PAYLOAD_BITS / delay_ms;
        float success_utility =
            2.0f * bandit_model_clampf(log1pf(goodput_kbps) / BANDIT_MODEL_UTILITY_SCALE,
                                       0.0f, 1.0f) - 1.0f;
#if BANDIT_MODEL_TAIL_ENABLED
        {
            float tail_excess = (delay_ms - BANDIT_MODEL_TAIL_TARGET_MS) / BANDIT_MODEL_TAIL_TARGET_MS;
            if (tail_excess < 0.0f) {
                tail_excess = 0.0f;
            }
            success_utility -= BANDIT_MODEL_TAIL_WEIGHT * bandit_model_clampf(tail_excess, 0.0f, 1.0f);
        }
#endif
        success_utility = bandit_model_clampf(success_utility, -1.0f, 1.0f);
        const float utility = p * success_utility + (1.0f - p) * BANDIT_MODEL_LOSS_REWARD;

        if (p_deliver_out != NULL) {
            p_deliver_out[action] = p;
        }
        if (delay_ms_out != NULL) {
            delay_ms_out[action] = delay_ms;
        }
        if (utility_out != NULL) {
            utility_out[action] = utility;
        }
    }
}

static uint8_t bandit_model_predict_best_mcs(const float state[BANDIT_MODEL_STATE_DIM],
                                             float *best_utility_out,
                                             float *second_utility_out)
{
    float utility[BANDIT_MODEL_ACTION_DIM];
    bandit_model_score_actions(state, NULL, NULL, utility);

    uint8_t best_action = 0;
    float best = -1e30f;
    float second = -1e30f;
    for (uint8_t action = 0; action < BANDIT_MODEL_ACTION_DIM; ++action) {
        if (utility[action] > best) {
            second = best;
            best = utility[action];
            best_action = action;
        } else if (utility[action] > second) {
            second = utility[action];
        }
    }
    if (best_utility_out != NULL) {
        *best_utility_out = best;
    }
    if (second_utility_out != NULL) {
        *second_utility_out = second;
    }
    return best_action;
}
"""


def write_header(output_path: Path, values: dict) -> None:
    guard = "GENERATED_BANDIT_MODEL_H"
    utility = values["utility"]
    tail_enabled = utility["tail_target_ms"] > 0.0 and utility["tail_weight"] > 0.0
    sections = [
        "/* Auto-generated by export_bandit_model_to_c_header.py. */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <math.h>",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        f"#define BANDIT_MODEL_STATE_DIM {values['state_dim']}",
        f"#define BANDIT_MODEL_ACTION_DIM {ACTION_DIM}",
        f"#define BANDIT_MODEL_ACTION_FEATURE_DIM {values['action_feature_dim']}",
        f"#define BANDIT_MODEL_INPUT_DIM {values['input_dim']}",
        f"#define BANDIT_MODEL_HIDDEN_DIM {values['hidden_dim']}",
        f"#define BANDIT_MODEL_AMPLITUDE_COUNT {values['amplitude_count']}",
        f'#define BANDIT_MODEL_CHECKPOINT_SHA256 "{values["checkpoint_sha256"]}"',
        f'#define BANDIT_MODEL_CHECKPOINT_SCHEMA "{values["checkpoint_schema"]}"',
        f'#define BANDIT_MODEL_OBJECTIVE_CONTRACT "{values["objective_contract"]}"',
        f"#define BANDIT_MODEL_TRAINING_EXACT_FRESH_ONLY {1 if values['training_exact_fresh_only'] else 0}",
        f"#define BANDIT_MODEL_REQUIRED_CSI_INPUT_SCALAR_COUNT {values['required_csi_input_scalar_count']}",
        f"#define BANDIT_MODEL_REQUIRES_VALID_FIRST_WORD {1 if values['requires_valid_first_word'] else 0}",
        f"#define BANDIT_MODEL_INCLUDES_STATE_MCS {1 if values['include_state_mcs'] else 0}",
        "#define BANDIT_MODEL_CONTEXT_IS_STATE_AGE_PACKETS 1",
        f"#define BANDIT_MODEL_STATE_SCHEMA_{values['state_schema'].upper()} 1",
        f'#define BANDIT_MODEL_CSI_FEATURE_CONTRACT_ID "{values["contract_metadata"].get("csi_feature_contract_id", "")}"',
        f'#define BANDIT_MODEL_CSI_FEATURE_CONTRACT_SHA256 "{values["contract_metadata"].get("csi_feature_contract_sha256", "")}"',
        f'#define BANDIT_MODEL_CSI_FEATURE_COUNT {values["contract_metadata"].get("csi_feature_count", 0)}',
        f'#define BANDIT_MODEL_STATE_CONTRACT_ID "{values["contract_metadata"].get("state_contract_id", "")}"',
        f'#define BANDIT_MODEL_STATE_CONTRACT_SHA256 "{values["contract_metadata"].get("state_contract_sha256", "")}"',
        "",
        f"#define BANDIT_MODEL_PAYLOAD_BITS {float(utility['payload_bytes'] * 8):.8e}f",
        f"#define BANDIT_MODEL_LOSS_REWARD {float(utility['loss_reward']):.8e}f",
        f"#define BANDIT_MODEL_UTILITY_SCALE {float(utility['utility_scale']):.8e}f",
        f"#define BANDIT_MODEL_TAIL_ENABLED {1 if tail_enabled else 0}",
        f"#define BANDIT_MODEL_TAIL_TARGET_MS {float(utility['tail_target_ms']):.8e}f",
        f"#define BANDIT_MODEL_TAIL_WEIGHT {float(utility['tail_weight']):.8e}f",
        f"#define BANDIT_MODEL_DELIVERY_BIAS {values['delivery_b']:.8e}f",
        f"#define BANDIT_MODEL_DELAY_BIAS {values['delay_b']:.8e}f",
        "",
        format_float_array("bandit_model_state_mean", values["state_mean"]),
        "",
        format_float_array("bandit_model_state_std", values["state_std"]),
        "",
        format_float_array("bandit_model_w1", values["w1"]),
        "",
        format_float_array("bandit_model_b1", values["b1"]),
        "",
        format_float_array("bandit_model_w2", values["w2"]),
        "",
        format_float_array("bandit_model_b2", values["b2"]),
        "",
        format_float_array("bandit_model_delivery_w", values["delivery_w"]),
        "",
        format_float_array("bandit_model_delay_w", values["delay_w"]),
        "",
        INFERENCE_CODE.strip(),
        "",
        f"#endif /* {guard} */",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text("\n".join(sections) + "\n", encoding="utf-8")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export bandit checkpoint to a C header")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-dataset", type=Path, default=None)
    parser.add_argument("--verify-rows", type=int, default=512)
    parser.add_argument(
        "--candidate-selection",
        type=Path,
        help="Sealed validation-only candidate_selection.json (required for fair checkpoints)",
    )
    parser.add_argument(
        "--qualification-complete",
        type=Path,
        help="Qualification completion seal (required for fair checkpoints)",
    )
    parser.add_argument(
        "--deployment-record",
        type=Path,
        help="Output deployment record (default: OUTPUT.deployment.json)",
    )
    parser.add_argument(
        "--allow-exact-fresh-only-export",
        action="store_true",
        help=(
            "Explicitly acknowledge that an exact-fresh-only checkpoint has no "
            "qualified stale/loss behavior"
        ),
    )
    args = parser.parse_args()

    if args.verify_rows <= 0:
        raise ValueError("--verify-rows must be positive")

    deployment_record = (
        args.deployment_record
        if args.deployment_record is not None
        else Path(f"{args.output}.deployment.json")
    )
    explicit_inputs = [args.model]
    explicit_inputs.extend(
        path
        for path in (
            args.verify_dataset,
            args.candidate_selection,
            args.qualification_complete,
        )
        if path is not None
    )
    reject_output_collisions(args.output, deployment_record, explicit_inputs)

    print("[Bandit Model C Export]")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    values = load_export_values(args.model)
    values["checkpoint_path"] = str(args.model)
    if values["is_fair_checkpoint"] and args.verify_dataset is None:
        raise ValueError(
            "Fair comparison bandit checkpoints require --verify-dataset before export"
        )
    if (
        values["training_exact_fresh_only"]
        and not args.allow_exact_fresh_only_export
    ):
        raise ValueError(
            "This bandit checkpoint was trained only on exact-fresh states. Export "
            "is blocked unless --allow-exact-fresh-only-export explicitly accepts "
            "that stale/loss firmware behavior is unqualified."
        )
    authorization = None
    if values["is_fair_checkpoint"]:
        if args.candidate_selection is None or args.qualification_complete is None:
            raise ValueError(
                "Fair comparison bandit export requires --candidate-selection and "
                "--qualification-complete so firmware cannot use an unselected or "
                "unqualified checkpoint"
            )
        authorization = validate_fair_deployment_authorization(
            args.model,
            args.verify_dataset,
            args.candidate_selection,
            args.qualification_complete,
        )
        protected_seals = []
        for record in authorization.values():
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                continue
            protected_path = Path(record["path"])
            if not protected_path.is_absolute():
                protected_path = REPO_ROOT / protected_path
            protected_seals.append(protected_path)
        reject_output_collisions(
            args.output,
            deployment_record,
            [*explicit_inputs, *protected_seals],
        )
    if args.verify_dataset is not None:
        verify_dataset(args.verify_dataset, values, args.verify_rows)
    write_header(args.output, values)
    if authorization is not None:
        header_text_sha = hashlib.sha256(
            args.output.read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        atomic_write_json(
            deployment_record,
            {
                "schema": DEPLOYMENT_RECORD_SCHEMA,
                "authorization": authorization,
                "checkpoint": {
                    "path": portable_path(args.model),
                    "sha256": values["checkpoint_sha256"],
                    "schema": values["checkpoint_schema"],
                    "objective_contract": values["objective_contract"],
                },
                "header": {
                    "path": portable_path(args.output),
                    "sha256": sha256_file(args.output),
                    "normalized_text_sha256": header_text_sha,
                },
                "verification_rows": int(args.verify_rows),
                "exact_fresh_only_export_acknowledged": True,
            },
        )
        print(f"  Deployment record: {deployment_record}")
    print(
        "  Exported network: "
        f"input {values['input_dim']} (state {values['state_dim']} + action 9) -> "
        f"{values['hidden_dim']} -> {values['hidden_dim']} -> 2 heads "
        f"(schema={values['state_schema']}, amplitudes={values['amplitude_count']}, "
        f"state_mcs={'yes' if values['include_state_mcs'] else 'no'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
