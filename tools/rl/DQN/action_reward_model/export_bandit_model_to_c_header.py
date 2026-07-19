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
    schema_amplitude_count,
    validate_dataset_feature_contract,
)
from train_reward_model import parse_iq_raw
from train_bandit_model import ACTION_DIM
from predict_bandit_model import load_bandit_checkpoint
from policy_utils import expected_utility


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
    action_feature_dim = int(checkpoint.get("action_feature_dim", 9))
    hidden_dim = int(checkpoint.get("hidden_dim", 128))
    input_dim = state_dim + action_feature_dim
    utility = checkpoint["utility_params"]
    if utility.get("objective", "utility") != "utility":
        raise ValueError(
            "Firmware bandit export currently implements only objective='utility'; "
            f"checkpoint objective is {utility.get('objective')!r}"
        )

    values = {
        "state_dim": state_dim,
        "action_feature_dim": action_feature_dim,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "state_schema": checkpoint["state_schema"],
        "amplitude_count": schema_amplitude_count(checkpoint["state_schema"]),
        "include_state_mcs": bool(checkpoint["include_state_mcs"]),
        "state_context_feature": checkpoint["state_context_feature"],
        "utility": utility,
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
        "contract_metadata": feature_contract_metadata(checkpoint["state_schema"]),
    }
    values["state_std"] = np.where(
        np.abs(values["state_std"]) < 1e-6, 1.0, values["state_std"]
    ).astype(np.float32)

    if values["state_context_feature"] != "state_age_packets":
        raise ValueError("Firmware bandit export requires a causal checkpoint")
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


def verify_dataset(dataset_path: Path, values: dict, rows: int) -> None:
    validate_dataset_feature_contract(dataset_path, values["state_schema"])
    frame = pd.read_csv(dataset_path, nrows=rows)
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
    numpy_p, numpy_mu = numpy_score_actions_float32(states, values)

    p_err = float(np.max(np.abs(torch_p - numpy_p)))
    mu_err = float(np.max(np.abs(torch_mu - numpy_mu)))

    torch_utility = expected_utility(torch_p, torch_mu, **values["utility"])
    numpy_utility = expected_utility(numpy_p, numpy_mu, **values["utility"])
    action_matches = int(
        np.sum(np.argmax(torch_utility, axis=1) == np.argmax(numpy_utility, axis=1))
    )
    if action_matches != len(states) or p_err > 1e-4 or mu_err > 1e-3:
        raise ValueError(
            f"Export verification failed: action_matches={action_matches}/{len(states)}, "
            f"p_err={p_err:.8g}, mu_err={mu_err:.8g}"
        )
    print(
        f"  Verification: {action_matches}/{len(states)} best actions matched, "
        f"p_err={p_err:.3g}, mu_err={mu_err:.3g}"
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
        f"#define BANDIT_MODEL_INCLUDES_STATE_MCS {1 if values['include_state_mcs'] else 0}",
        "#define BANDIT_MODEL_CONTEXT_IS_STATE_AGE_PACKETS 1",
        f"#define BANDIT_MODEL_STATE_SCHEMA_{values['state_schema'].upper()} 1",
        f'#define BANDIT_MODEL_CSI_FEATURE_CONTRACT_ID "{values["contract_metadata"].get("csi_feature_contract_id", "")}"',
        f'#define BANDIT_MODEL_CSI_FEATURE_CONTRACT_SHA256 "{values["contract_metadata"].get("csi_feature_contract_sha256", "")}"',
        f'#define BANDIT_MODEL_CSI_FEATURE_COUNT {values["contract_metadata"].get("csi_feature_count", 0)}',
        f'#define BANDIT_MODEL_STATE_CONTRACT_ID "{values["contract_metadata"].get("state_contract_id", "")}"',
        f'#define BANDIT_MODEL_STATE_CONTRACT_SHA256 "{values["contract_metadata"].get("state_contract_sha256", "")}"',
        "",
        f"#define BANDIT_MODEL_PAYLOAD_BITS {float(utility['payload_bytes'] * 8):.1f}f",
        f"#define BANDIT_MODEL_LOSS_REWARD {float(utility['loss_reward']):.6f}f",
        f"#define BANDIT_MODEL_UTILITY_SCALE {float(utility['utility_scale']):.6f}f",
        f"#define BANDIT_MODEL_TAIL_ENABLED {1 if tail_enabled else 0}",
        f"#define BANDIT_MODEL_TAIL_TARGET_MS {float(utility['tail_target_ms']):.6f}f",
        f"#define BANDIT_MODEL_TAIL_WEIGHT {float(utility['tail_weight']):.6f}f",
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
    output_path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export bandit checkpoint to a C header")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-dataset", type=Path, default=None)
    parser.add_argument("--verify-rows", type=int, default=512)
    args = parser.parse_args()

    if args.verify_rows <= 0:
        raise ValueError("--verify-rows must be positive")

    print("[Bandit Model C Export]")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    values = load_export_values(args.model)
    values["checkpoint_path"] = str(args.model)
    if args.verify_dataset is not None:
        verify_dataset(args.verify_dataset, values, args.verify_rows)
    write_header(args.output, values)
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
