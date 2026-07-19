#!/usr/bin/env python3
"""Export a trained DQN Q-network checkpoint to a standalone C header."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_dqn import (
    build_state_vector,
    feature_contract_metadata,
    state_feature_names,
    validate_dataset_feature_contract,
)


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def format_float_array(name: str, array: np.ndarray, per_line: int = 8) -> str:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    lines = [f"static const float {name}[{flat.size}] = {{"]
    for index in range(0, flat.size, per_line):
        values = ", ".join(f"{float(value):.8e}f" for value in flat[index : index + per_line])
        lines.append(f"    {values},")
    lines.append("};")
    return "\n".join(lines)


def load_export_values(model_path: Path) -> dict:
    checkpoint = torch.load(model_path, map_location="cpu")

    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])
    hidden_dim = int(checkpoint.get("hidden_dim", 128))
    hidden_layers = int(checkpoint.get("hidden_layers", 2))
    dropout = float(checkpoint.get("dropout", 0.0))
    layer_norm = bool(checkpoint.get("layer_norm", False))
    state_context_feature = checkpoint.get("state_context_feature", "sig_len")
    state_schema = checkpoint.get("state_schema", "legacy_v1")
    include_state_mcs = bool(
        checkpoint.get("include_state_mcs", state_dim in {136, 140, 143, 262})
    )

    if action_dim != 8:
        raise ValueError(f"Firmware DQN exporter requires action_dim=8, got {action_dim}")
    if hidden_layers != 2:
        raise ValueError(f"Firmware DQN exporter requires two hidden layers, got {hidden_layers}")
    if dropout != 0.0:
        raise ValueError("Firmware DQN exporter does not support dropout")
    if layer_norm:
        raise ValueError("Firmware DQN exporter does not support LayerNorm")
    if state_context_feature != "state_age_packets":
        raise ValueError(
            "Firmware DQN requires a causal checkpoint using state_age_packets"
        )
    expected_state_dim = len(state_feature_names(state_schema, include_state_mcs))
    if state_dim != expected_state_dim:
        raise ValueError(
            f"Checkpoint state_dim={state_dim} does not match schema "
            f"{state_schema} include_state_mcs={include_state_mcs} "
            f"(expected {expected_state_dim})"
        )
    contract_metadata = feature_contract_metadata(state_schema)
    for key, expected in contract_metadata.items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"Checkpoint {key}={checkpoint.get(key)!r} does not match expected {expected!r}"
            )

    model_state = checkpoint["q_net_state"]
    values = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "state_schema": state_schema,
        "include_state_mcs": include_state_mcs,
        "state_mean": to_numpy(checkpoint["state_mean"]).astype(np.float32),
        "state_std": to_numpy(checkpoint["state_std"]).astype(np.float32),
        "w1": to_numpy(model_state["net.0.weight"]).astype(np.float32),
        "b1": to_numpy(model_state["net.0.bias"]).astype(np.float32),
        "w2": to_numpy(model_state["net.2.weight"]).astype(np.float32),
        "b2": to_numpy(model_state["net.2.bias"]).astype(np.float32),
        "w3": to_numpy(model_state["net.4.weight"]).astype(np.float32),
        "b3": to_numpy(model_state["net.4.bias"]).astype(np.float32),
        "checkpoint": checkpoint,
        "contract_metadata": contract_metadata,
    }
    values["state_std"] = np.where(
        np.abs(values["state_std"]) < 1e-6,
        1.0,
        values["state_std"],
    ).astype(np.float32)

    expected_shapes = {
        "state_mean": (state_dim,),
        "state_std": (state_dim,),
        "w1": (hidden_dim, state_dim),
        "b1": (hidden_dim,),
        "w2": (hidden_dim, hidden_dim),
        "b2": (hidden_dim,),
        "w3": (action_dim, hidden_dim),
        "b3": (action_dim,),
    }
    for name, expected_shape in expected_shapes.items():
        if values[name].shape != expected_shape:
            raise ValueError(
                f"Unexpected {name} shape {values[name].shape}; expected {expected_shape}"
            )

    return values


def numpy_forward(states: np.ndarray, values: dict) -> np.ndarray:
    normalized = (states - values["state_mean"]) / values["state_std"]
    hidden1 = np.maximum(normalized @ values["w1"].T + values["b1"], 0.0)
    hidden2 = np.maximum(hidden1 @ values["w2"].T + values["b2"], 0.0)
    return hidden2 @ values["w3"].T + values["b3"]


def verify_dataset(dataset_path: Path, values: dict, rows: int) -> None:
    validate_dataset_feature_contract(dataset_path, values["state_schema"])
    frame = pd.read_csv(dataset_path, nrows=rows)
    states = np.asarray(
        [
            build_state_vector(
                row,
                "state_age_packets",
                values["include_state_mcs"],
                values["state_schema"],
            )
            for _, row in frame.iterrows()
        ],
        dtype=np.float32,
    )

    checkpoint = values["checkpoint"]
    # Instantiate directly to avoid relying on serialized module objects.
    from train_dqn import QNetwork

    network = QNetwork(
        state_dim=values["state_dim"],
        action_dim=values["action_dim"],
        hidden_dim=values["hidden_dim"],
        hidden_layers=2,
    )
    network.load_state_dict(checkpoint["q_net_state"])
    network.eval()

    normalized = (states - values["state_mean"]) / values["state_std"]
    with torch.no_grad():
        torch_q = network(torch.from_numpy(normalized).float()).cpu().numpy()
    numpy_q = numpy_forward(states, values)

    max_abs_error = float(np.max(np.abs(torch_q - numpy_q)))
    action_matches = int(
        np.sum(np.argmax(torch_q, axis=1) == np.argmax(numpy_q, axis=1))
    )
    if action_matches != len(states) or max_abs_error > 1e-4:
        raise ValueError(
            f"Export verification failed: action_matches={action_matches}/{len(states)}, "
            f"max_abs_error={max_abs_error:.8g}"
        )
    print(
        f"  Verification: {action_matches}/{len(states)} actions matched, "
        f"max_abs_error={max_abs_error:.8g}"
    )


def write_header(output_path: Path, values: dict) -> None:
    guard = "GENERATED_DQN_MODEL_H"
    sections = [
        "/* Auto-generated by export_dqn_to_c_header.py. */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        f"#define DQN_MODEL_STATE_DIM {values['state_dim']}",
        f"#define DQN_MODEL_ACTION_DIM {values['action_dim']}",
        f"#define DQN_MODEL_HIDDEN_DIM {values['hidden_dim']}",
        "#define DQN_MODEL_CONTEXT_IS_STATE_AGE_PACKETS 1",
        f"#define DQN_MODEL_INCLUDES_STATE_MCS {1 if values['include_state_mcs'] else 0}",
        f"#define DQN_MODEL_STATE_SCHEMA_LINK_V2 {1 if values['state_schema'] == 'link_v2' else 0}",
        f"#define DQN_MODEL_STATE_SCHEMA_LINK_V3 {1 if values['state_schema'] == 'link_v3' else 0}",
        f"#define DQN_MODEL_STATE_SCHEMA_LINK_V4 {1 if values['state_schema'] == 'link_v4' else 0}",
        f"#define DQN_MODEL_STATE_SCHEMA_LINK_V5 {1 if values['state_schema'] == 'link_v5' else 0}",
        f"#define DQN_MODEL_STATE_SCHEMA_LINK_V6 {1 if values['state_schema'] == 'link_v6' else 0}",
        f"#define DQN_MODEL_STATE_SCHEMA_LINK_V7C {1 if values['state_schema'] == 'link_v7c' else 0}",
        f'#define DQN_MODEL_CSI_FEATURE_CONTRACT_ID "{values["contract_metadata"].get("csi_feature_contract_id", "")}"',
        f'#define DQN_MODEL_CSI_FEATURE_CONTRACT_SHA256 "{values["contract_metadata"].get("csi_feature_contract_sha256", "")}"',
        f'#define DQN_MODEL_CSI_FEATURE_COUNT {values["contract_metadata"].get("csi_feature_count", 0)}',
        f'#define DQN_MODEL_STATE_CONTRACT_ID "{values["contract_metadata"].get("state_contract_id", "")}"',
        f'#define DQN_MODEL_STATE_CONTRACT_SHA256 "{values["contract_metadata"].get("state_contract_sha256", "")}"',
        "",
        format_float_array("dqn_model_state_mean", values["state_mean"]),
        "",
        format_float_array("dqn_model_state_std", values["state_std"]),
        "",
        format_float_array("dqn_model_w1", values["w1"]),
        "",
        format_float_array("dqn_model_b1", values["b1"]),
        "",
        format_float_array("dqn_model_w2", values["w2"]),
        "",
        format_float_array("dqn_model_b2", values["b2"]),
        "",
        format_float_array("dqn_model_w3", values["w3"]),
        "",
        format_float_array("dqn_model_b3", values["b3"]),
        "",
        f"#endif /* {guard} */",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export DQN checkpoint to a C header")
    parser.add_argument("--model", type=Path, required=True, help="DQN .pth checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Generated .h path")
    parser.add_argument(
        "--verify-dataset",
        type=Path,
        help="Optional causal dataset used to verify exported forward-pass parity",
    )
    parser.add_argument(
        "--verify-rows",
        type=int,
        default=512,
        help="Rows used by --verify-dataset",
    )
    args = parser.parse_args()

    if args.verify_rows <= 0:
        raise ValueError("--verify-rows must be positive")

    print("[DQN C Export]")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    values = load_export_values(args.model)
    if args.verify_dataset is not None:
        verify_dataset(args.verify_dataset, values, args.verify_rows)
    write_header(args.output, values)
    print(
        "  Exported network: "
        f"{values['state_dim']} -> {values['hidden_dim']} -> "
        f"{values['hidden_dim']} -> {values['action_dim']} "
        f"(schema={values['state_schema']}, "
        f"state_mcs={'yes' if values['include_state_mcs'] else 'no'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
