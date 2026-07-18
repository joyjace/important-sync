#!/usr/bin/env python3
"""Sender/receiver parameter menu with algorithm mode presets.

Runs on any host that has a checkout (lab laptops, the per-device Raspberry
Pis). Shared device configuration lives in device_menu_profile.json and is
synced between hosts; host-local settings (serial ports, ESP-IDF export
script, build directory) live in device_menu_local.json, which stays local
to each machine. A mode preset edits BOTH firmware sources consistently;
each host then builds/flashes only the board attached to it.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SENDER_MAIN = ROOT / "csi_send" / "main" / "app_main.c"
RECEIVER_MAIN = ROOT / "csi_recv" / "main" / "app_main.c"
PROFILE_PATH = Path(__file__).resolve().parent / "device_menu_profile.json"
LOCAL_PATH = Path(__file__).resolve().parent / "device_menu_local.json"

IS_WINDOWS = os.name == "nt"

RATE_CHOICES = [
    "WIFI_PHY_RATE_MCS0_LGI",
    "WIFI_PHY_RATE_MCS1_LGI",
    "WIFI_PHY_RATE_MCS2_LGI",
    "WIFI_PHY_RATE_MCS3_LGI",
    "WIFI_PHY_RATE_MCS4_LGI",
    "WIFI_PHY_RATE_MCS5_LGI",
    "WIFI_PHY_RATE_MCS6_LGI",
    "WIFI_PHY_RATE_MCS7_LGI",
]

LIVE_MCS_ALGO_CHOICES = [
    "MINSTREL_LIKE - sender ACK EWMA + probing",
    "CUSTOM_POLICY - legacy reward-model receiver recommendations",
    "RECEIVER_POLICY - receiver live-policy frames (bandit or DQN)",
]

DEPRECATED_SENDER_KEYS = (
    "custom_reward_throughput_weight",
    "custom_reward_delay_weight",
    "custom_reward_loss_weight",
    "custom_reward_exploration_weight",
)

DEFAULT_PROFILE = {
    "sender": {
        "channel": 11,
        "esp_now_rate": "WIFI_PHY_RATE_MCS0_LGI",
        "send_frequency": 100,
        "packet_pacing_enabled": 1,
        "ack_timing_mode": 2,
        "rate_switch_mode": 2,
        "rate_switch_interval_sec": 10,
        "rate_switch_packet_count": 1000,
        "rate_sweep_group_size": 1,
        "payload_len": 128,
        "tx_power": 84,
        "live_mcs_selection_enabled": 1,
        "live_mcs_algo": 0,
        "live_mcs_min_index": 0,
        "live_mcs_max_index": 7,
        "minstrel_update_every_pkts": 20,
        "minstrel_probe_every_pkts": 10,
        "minstrel_ewma_alpha_num": 1,
        "minstrel_ewma_alpha_den": 4,
        "custom_policy_default_mcs": 0,
        "live_mcs_decision_log_enabled": 1,
        "remote_mcs_recommendation_enabled": 1,
        "remote_mcs_min_confidence": 0,
        "remote_mcs_max_age_ms": 300,
        "dqn_remote_recommendation_enabled": 0,
        "dqn_default_mcs": 4,
        "dqn_remote_min_confidence": 0,
        "dqn_remote_max_age_ms": 50,
        "dqn_remote_max_seq_gap": 0,
        "dqn_failure_stepdown_enabled": 0,
        "dqn_failure_stepdown_count": 3,
        "dqn_log_enabled": 1,
    },
    "receiver": {
        "channel": 11,
        "esp_now_rate": "WIFI_PHY_RATE_MCS0_LGI",
        "force_gain": 1,
        "gain_control": 1,
        "custom_mcs_recommendation_enabled": 0,
        "dqn_mcs_recommendation_enabled": 0,
        "mcs_policy_model": 1,
        "dqn_recommendation_every_n_packets": 1,
        "dqn_warmup_packets": 100,
        "dqn_control_interval_ms": 5,
        "dqn_stale_max_age_packets": 64,
        "dqn_log_enabled": 1,
    },
    "shared": {
        "sender_mac": "1a:00:00:00:00:00",
        "receiver_mac": "1a:00:00:00:00:01",
    },
}

if IS_WINDOWS:
    DEFAULT_LOCAL = {
        "sender_port": "COM3",
        "receiver_port": "COM4",
        "baud": 921600,
        "export_script": str(Path.home() / "esp" / "esp-idf-v5.5.2" / "export.bat"),
        "build_dir": "build_win",
    }
else:
    DEFAULT_LOCAL = {
        "sender_port": "/dev/ttyUSB0",
        "receiver_port": "/dev/ttyUSB0",
        "baud": 921600,
        "export_script": "$HOME/esp/esp-idf/export.sh",
        "build_dir": "build",
    }

# Algorithm mode presets. Applying one rewrites the sender AND receiver
# profile sections so the pair stays consistent; each host then builds and
# flashes only its own attached board. "headers" lists generated model
# headers the receiver build needs: (relative path, required text, hint).
MODES = {
    "minstrel": {
        "label": "Minstrel-like (sender-local ACK statistics, no receiver feedback)",
        "sender": {
            "live_mcs_selection_enabled": 1,
            "live_mcs_algo": 0,
            "remote_mcs_recommendation_enabled": 0,
            "dqn_remote_recommendation_enabled": 0,
        },
        "receiver": {
            "custom_mcs_recommendation_enabled": 0,
            "dqn_mcs_recommendation_enabled": 0,
        },
        "headers": [],
    },
    "bandit": {
        "label": "Contextual bandit (receiver live policy, link_v3c model)",
        "sender": {
            "live_mcs_selection_enabled": 1,
            "live_mcs_algo": 2,
            "remote_mcs_recommendation_enabled": 0,
            "dqn_remote_recommendation_enabled": 1,
        },
        "receiver": {
            "custom_mcs_recommendation_enabled": 0,
            "dqn_mcs_recommendation_enabled": 1,
            "mcs_policy_model": 1,
        },
        "headers": [
            (
                "csi_recv/main/generated_bandit_model.h",
                "BANDIT_MODEL_STATE_SCHEMA_LINK_V3C",
                "tools/rl/DQN/action_reward_model/export_bandit_model_to_c_header.py",
            ),
        ],
    },
    "dqn": {
        "label": "DQN Q-network (receiver live policy)",
        "sender": {
            "live_mcs_selection_enabled": 1,
            "live_mcs_algo": 2,
            "remote_mcs_recommendation_enabled": 0,
            "dqn_remote_recommendation_enabled": 1,
        },
        "receiver": {
            "custom_mcs_recommendation_enabled": 0,
            "dqn_mcs_recommendation_enabled": 1,
            "mcs_policy_model": 0,
        },
        "headers": [
            (
                "csi_recv/main/generated_dqn_model.h",
                "DQN_MODEL_STATE_DIM",
                "tools/rl/DQN/dqn_model/export_dqn_to_c_header.py",
            ),
        ],
    },
    "reward_model": {
        "label": "Legacy reward-model custom policy (receiver recommendations)",
        "sender": {
            "live_mcs_selection_enabled": 1,
            "live_mcs_algo": 1,
            "remote_mcs_recommendation_enabled": 1,
            "dqn_remote_recommendation_enabled": 0,
        },
        "receiver": {
            "custom_mcs_recommendation_enabled": 1,
            "dqn_mcs_recommendation_enabled": 0,
        },
        "headers": [
            (
                "csi_recv/main/generated_reward_model_v2.h",
                "REWARD_MODEL_STATE_DIM",
                "tools/rl/DQN/action_reward_model/export_reward_model_to_c_header.py",
            ),
        ],
    },
    "static": {
        "label": "Static MCS baseline (fixed rate, no adaptation)",
        "sender": {
            "live_mcs_selection_enabled": 0,
            "rate_switch_mode": 2,
        },
        "receiver": {
            "custom_mcs_recommendation_enabled": 0,
            "dqn_mcs_recommendation_enabled": 0,
        },
        "headers": [],
    },
    "random_sweep": {
        "label": "RANDOM_SWEEP data collection (uniform per-packet MCS, propensity 1/8)",
        "sender": {
            "live_mcs_selection_enabled": 0,
            "rate_switch_mode": 3,
            "ack_timing_mode": 2,
        },
        "receiver": {
            "custom_mcs_recommendation_enabled": 0,
            "dqn_mcs_recommendation_enabled": 0,
        },
        "headers": [],
    },
}

MODE_ORDER = ["minstrel", "bandit", "dqn", "reward_model", "static", "random_sweep"]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def replace_define(text: str, name: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*#define\s+{re.escape(name)}\s+)(\S+)(.*)$", re.MULTILINE)
    new_text, count = pattern.subn(rf"\g<1>{value}\g<3>", text, count=1)
    if count == 0:
        raise ValueError(f"Could not find #define {name}")
    return new_text


def replace_mac_array(text: str, symbol: str, mac: str) -> str:
    mac_bytes = parse_mac(mac)
    replacement = "{" + ", ".join(f"0x{b:02x}" for b in mac_bytes) + "}"
    pattern = re.compile(rf"(static\s+const\s+uint8_t\s+{re.escape(symbol)}\[\]\s*=\s*)\{{[^}}]+\}}(;)\s*")
    new_text, count = pattern.subn(rf"\g<1>{replacement}\g<2>\n", text, count=1)
    if count == 0:
        raise ValueError(f"Could not find array for {symbol}")
    return new_text


def parse_mac(mac: str):
    parts = mac.strip().split(":")
    if len(parts) != 6:
        raise ValueError("MAC must have 6 hex bytes separated by ':'")
    out = []
    for part in parts:
        if len(part) != 2:
            raise ValueError("Each MAC byte must have 2 hex chars")
        out.append(int(part, 16))
    return out


def load_profile() -> dict:
    if not PROFILE_PATH.exists():
        return json.loads(json.dumps(DEFAULT_PROFILE))

    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    merged = json.loads(json.dumps(DEFAULT_PROFILE))

    for section in merged:
        if section in profile and isinstance(profile[section], dict):
            merged[section].update(profile[section])

    for key in DEPRECATED_SENDER_KEYS:
        merged["sender"].pop(key, None)

    return merged


def save_profile(profile: dict) -> None:
    PROFILE_PATH.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")


def load_local() -> dict:
    merged = json.loads(json.dumps(DEFAULT_LOCAL))

    if LOCAL_PATH.exists():
        local = json.loads(LOCAL_PATH.read_text(encoding="utf-8"))
        if isinstance(local, dict):
            merged.update(local)
        return merged

    # First run on this host: migrate the old profile["flash"] section if the
    # synced profile still carries one (pre-split layout). Those values were
    # Linux-style, so keep OS defaults on Windows.
    if not IS_WINDOWS and PROFILE_PATH.exists():
        try:
            profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
            flash = profile.get("flash")
            if isinstance(flash, dict):
                for key in ("sender_port", "receiver_port", "baud", "export_script"):
                    if key in flash:
                        merged[key] = flash[key]
        except (ValueError, OSError):
            pass

    save_local(merged)
    return merged


def save_local(local: dict) -> None:
    LOCAL_PATH.write_text(json.dumps(local, indent=2) + "\n", encoding="utf-8")


def validate_sender_profile(sender: dict) -> None:
    if sender["live_mcs_algo"] < 0 or sender["live_mcs_algo"] >= len(LIVE_MCS_ALGO_CHOICES):
        sender["live_mcs_algo"] = 0

    if sender["live_mcs_min_index"] > sender["live_mcs_max_index"]:
        sender["live_mcs_min_index"], sender["live_mcs_max_index"] = (
            sender["live_mcs_max_index"],
            sender["live_mcs_min_index"],
        )

    if sender["minstrel_ewma_alpha_num"] > sender["minstrel_ewma_alpha_den"]:
        sender["minstrel_ewma_alpha_num"] = sender["minstrel_ewma_alpha_den"]

    if sender.get("ack_timing_mode", 2) not in (1, 2):
        sender["ack_timing_mode"] = 2

    if sender.get("rate_sweep_group_size", 1) < 1:
        sender["rate_sweep_group_size"] = 1

    # RANDOM_SWEEP requires stop-and-wait for exact per-packet rate logging.
    if sender.get("rate_switch_mode", 2) == 3 and sender["ack_timing_mode"] != 2:
        print("NOTE: RANDOM_SWEEP requires stop-and-wait; forcing ack_timing_mode=2.")
        sender["ack_timing_mode"] = 2


def validate_receiver_profile(receiver: dict) -> None:
    if receiver["dqn_recommendation_every_n_packets"] <= 0:
        receiver["dqn_recommendation_every_n_packets"] = 1

    if receiver["dqn_warmup_packets"] < 0:
        receiver["dqn_warmup_packets"] = 0

    if receiver.get("dqn_control_interval_ms", 5) <= 0:
        receiver["dqn_control_interval_ms"] = 5

    if receiver.get("dqn_stale_max_age_packets", 64) <= 0:
        receiver["dqn_stale_max_age_packets"] = 64

    if receiver.get("mcs_policy_model", 1) not in (0, 1):
        receiver["mcs_policy_model"] = 1

    # Both recommendation paths on at once is a firmware compile error.
    if receiver.get("custom_mcs_recommendation_enabled", 0) and receiver.get("dqn_mcs_recommendation_enabled", 0):
        print("NOTE: custom and DQN recommendation paths are mutually exclusive; disabling the custom path.")
        receiver["custom_mcs_recommendation_enabled"] = 0


def apply_profile_to_sources(profile: dict) -> None:
    sender_text = read_text(SENDER_MAIN)
    receiver_text = read_text(RECEIVER_MAIN)

    sender = profile["sender"]
    receiver = profile["receiver"]
    shared = profile["shared"]

    validate_sender_profile(sender)
    validate_receiver_profile(receiver)

    sender_text = replace_define(sender_text, "CONFIG_LESS_INTERFERENCE_CHANNEL", str(sender["channel"]))
    sender_text = replace_define(sender_text, "CONFIG_ESP_NOW_RATE", sender["esp_now_rate"])
    sender_text = replace_define(sender_text, "CONFIG_SEND_FREQUENCY", str(sender["send_frequency"]))
    sender_text = replace_define(sender_text, "CONFIG_PACKET_PACING_ENABLED", str(sender["packet_pacing_enabled"]))
    sender_text = replace_define(sender_text, "CONFIG_ACK_TIMING_MODE", str(sender["ack_timing_mode"]))
    sender_text = replace_define(sender_text, "CONFIG_RATE_SWITCH_MODE", str(sender["rate_switch_mode"]))
    sender_text = replace_define(sender_text, "CONFIG_RATE_SWITCH_INTERVAL_SEC", str(sender["rate_switch_interval_sec"]))
    sender_text = replace_define(sender_text, "CONFIG_RATE_SWITCH_PACKET_COUNT", str(sender["rate_switch_packet_count"]))
    sender_text = replace_define(sender_text, "CONFIG_RATE_SWEEP_GROUP_SIZE", str(sender["rate_sweep_group_size"]))
    sender_text = replace_define(sender_text, "CONFIG_ESP_NOW_PAYLOAD_LEN", str(sender["payload_len"]))
    sender_text = replace_define(sender_text, "CONFIG_WIFI_TX_POWER", str(sender["tx_power"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_LIVE_MCS_SELECTION_ENABLED", str(sender["live_mcs_selection_enabled"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_LIVE_MCS_ALGO", str(sender["live_mcs_algo"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_LIVE_MCS_MIN_INDEX", str(sender["live_mcs_min_index"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_LIVE_MCS_MAX_INDEX", str(sender["live_mcs_max_index"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_MINSTREL_UPDATE_EVERY_PKTS", str(sender["minstrel_update_every_pkts"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_MINSTREL_PROBE_EVERY_PKTS", str(sender["minstrel_probe_every_pkts"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_MINSTREL_EWMA_ALPHA_NUM", str(sender["minstrel_ewma_alpha_num"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_MINSTREL_EWMA_ALPHA_DEN", str(sender["minstrel_ewma_alpha_den"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_CUSTOM_POLICY_DEFAULT_MCS", str(sender["custom_policy_default_mcs"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_LIVE_MCS_DECISION_LOG_ENABLED", str(sender["live_mcs_decision_log_enabled"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_REMOTE_MCS_RECOMMENDATION_ENABLED", str(sender["remote_mcs_recommendation_enabled"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_REMOTE_MCS_MIN_CONFIDENCE", str(sender["remote_mcs_min_confidence"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_REMOTE_MCS_MAX_AGE_MS", str(sender["remote_mcs_max_age_ms"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_DQN_REMOTE_RECOMMENDATION_ENABLED", str(sender["dqn_remote_recommendation_enabled"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_DQN_DEFAULT_MCS", str(sender["dqn_default_mcs"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_DQN_REMOTE_MIN_CONFIDENCE", str(sender["dqn_remote_min_confidence"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_DQN_REMOTE_MAX_AGE_MS", str(sender["dqn_remote_max_age_ms"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_DQN_REMOTE_MAX_SEQ_GAP", str(sender["dqn_remote_max_seq_gap"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_DQN_FAILURE_STEPDOWN_ENABLED", str(sender["dqn_failure_stepdown_enabled"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_DQN_FAILURE_STEPDOWN_COUNT", str(sender["dqn_failure_stepdown_count"]))
    sender_text = replace_define(sender_text, "CONFIG_CSI_DQN_LOG_ENABLED", str(sender["dqn_log_enabled"]))
    sender_text = replace_mac_array(sender_text, "CONFIG_CSI_SEND_MAC", shared["sender_mac"])
    sender_text = replace_mac_array(sender_text, "CONFIG_CSI_RECV_MAC", shared["receiver_mac"])

    receiver_text = replace_define(receiver_text, "CONFIG_LESS_INTERFERENCE_CHANNEL", str(receiver["channel"]))
    receiver_text = replace_define(receiver_text, "CONFIG_ESP_NOW_RATE", receiver["esp_now_rate"])
    receiver_text = replace_define(receiver_text, "CONFIG_FORCE_GAIN", str(receiver["force_gain"]))
    receiver_text = replace_define(receiver_text, "CONFIG_GAIN_CONTROL", str(receiver["gain_control"]))
    receiver_text = replace_define(
        receiver_text,
        "CONFIG_MCS_RECOMMENDATION_ENABLED",
        str(receiver["custom_mcs_recommendation_enabled"]),
    )
    receiver_text = replace_define(receiver_text, "CONFIG_CSI_MCS_POLICY_MODEL", str(receiver["mcs_policy_model"]))
    receiver_text = replace_define(receiver_text, "CONFIG_CSI_DQN_MCS_RECOMMENDATION_ENABLED", str(receiver["dqn_mcs_recommendation_enabled"]))
    receiver_text = replace_define(receiver_text, "CONFIG_CSI_DQN_RECOMMENDATION_EVERY_N_PACKETS", str(receiver["dqn_recommendation_every_n_packets"]))
    receiver_text = replace_define(receiver_text, "CONFIG_CSI_DQN_WARMUP_PACKETS", str(receiver["dqn_warmup_packets"]))
    receiver_text = replace_define(receiver_text, "CONFIG_CSI_DQN_CONTROL_INTERVAL_MS", str(receiver["dqn_control_interval_ms"]))
    receiver_text = replace_define(receiver_text, "CONFIG_CSI_DQN_STALE_MAX_AGE_PACKETS", str(receiver["dqn_stale_max_age_packets"]))
    receiver_text = replace_define(receiver_text, "CONFIG_CSI_DQN_LOG_ENABLED", str(receiver["dqn_log_enabled"]))
    receiver_text = replace_mac_array(receiver_text, "CONFIG_CSI_SEND_MAC", shared["sender_mac"])
    receiver_text = replace_mac_array(receiver_text, "CONFIG_CSI_RECV_MAC", shared["receiver_mac"])

    write_text(SENDER_MAIN, sender_text)
    write_text(RECEIVER_MAIN, receiver_text)


# ---------------------------------------------------------------------------
# Mode presets
# ---------------------------------------------------------------------------

def mode_profile_changes(profile: dict, mode: dict) -> list:
    """Return [(section, key, old, new)] for values the mode would change."""
    changes = []
    for section in ("sender", "receiver"):
        for key, new in mode.get(section, {}).items():
            old = profile[section].get(key)
            if old != new:
                changes.append((section, key, old, new))
    return changes


def apply_mode_to_profile(profile: dict, mode: dict) -> None:
    for section in ("sender", "receiver"):
        profile[section].update(mode.get(section, {}))
    validate_sender_profile(profile["sender"])
    validate_receiver_profile(profile["receiver"])


def check_mode_headers(mode: dict) -> bool:
    ok = True
    for rel_path, required_text, hint in mode.get("headers", []):
        path = ROOT / rel_path
        if not path.exists():
            print(f"WARNING: {rel_path} is missing. Export it with {hint}")
            ok = False
        elif required_text not in read_text(path):
            print(f"WARNING: {rel_path} does not contain {required_text}.")
            print(f"         Re-export a matching checkpoint with {hint}")
            ok = False
    return ok


def guess_current_mode(profile: dict) -> str:
    sender = profile["sender"]
    receiver = profile["receiver"]
    for key in MODE_ORDER:
        mode = MODES[key]
        if not mode_profile_changes(profile, mode):
            return key
    if not sender["live_mcs_selection_enabled"]:
        return {2: "static", 3: "random_sweep"}.get(sender["rate_switch_mode"], "custom")
    return "custom"


def select_mode(profile: dict, local: dict) -> None:
    print("\nSelect algorithm mode (configures sender AND receiver consistently)")
    current = guess_current_mode(profile)
    for idx, key in enumerate(MODE_ORDER, start=1):
        marker = "  <- current" if key == current else ""
        print(f"  {idx}. {MODES[key]['label']}{marker}")
    print("  0. Back")

    raw = input("Choose mode: ").strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(MODE_ORDER)):
        return
    key = MODE_ORDER[int(raw) - 1]
    mode = json.loads(json.dumps(MODES[key]))

    if key == "static":
        idx = ask_int("Fixed MCS index", 0, 0, 7)
        mode["sender"]["esp_now_rate"] = RATE_CHOICES[idx]
    elif key == "random_sweep":
        mode["sender"]["rate_sweep_group_size"] = ask_int(
            "Packets per MCS inside each shuffled 8-MCS window",
            profile["sender"].get("rate_sweep_group_size", 1), 1, 1000000)

    changes = mode_profile_changes(profile, mode)
    if not changes:
        print("Profile already matches this mode.")
    else:
        print("\nThis mode will change:")
        for section, field, old, new in changes:
            print(f"  {section}.{field}: {old} -> {new}")

    if not ask_yes_no("Apply this mode to the profile and C sources", True):
        return

    apply_mode_to_profile(profile, mode)
    save_profile(profile)
    check_mode_headers(mode)
    try:
        apply_profile_to_sources(profile)
    except Exception as exc:
        print(f"Apply failed: {exc}")
        return
    print(f"Mode '{key}' applied to profile and both app_main.c files.")
    print("Remember: every host (both Pis) must get these sources before flashing.")

    print("\nBuild + flash on this host now?")
    print("  1. Sender (board attached here)")
    print("  2. Receiver (board attached here)")
    print("  3. Both (single-host bench)")
    print("  0. Skip")
    choice = input("Choose: ").strip()
    if choice == "1":
        run_flash_sender(local)
    elif choice == "2":
        run_flash_receiver(local)
    elif choice == "3":
        run_flash_sender(local)
        run_flash_receiver(local)


# ---------------------------------------------------------------------------
# Build / flash (host-local)
# ---------------------------------------------------------------------------

def run_idf(target_dir: Path, command: str, local: dict) -> int:
    build_dir = str(local.get("build_dir", "")).strip()
    build_arg = f"-B {build_dir} " if build_dir else ""
    export_script = local["export_script"]

    if IS_WINDOWS:
        cmd = f'call "{export_script}" >NUL 2>&1 && cd /d "{target_dir}" && idf.py {build_arg}{command}'
        print(f"\n[run] {target_dir}: idf.py {build_arg}{command}\n")
        proc = subprocess.run(["cmd", "/c", cmd])
    else:
        cmd = f'. {export_script} && idf.py {build_arg}{command}'
        print(f"\n[run] {target_dir}: {cmd}\n")
        proc = subprocess.run(["bash", "-lc", cmd], cwd=str(target_dir))
    return proc.returncode


def run_build_sender(local: dict) -> None:
    code = run_idf(ROOT / "csi_send", "build", local)
    print("Sender build OK" if code == 0 else f"Sender build failed with exit code {code}")


def run_build_receiver(local: dict) -> None:
    code = run_idf(ROOT / "csi_recv", "build", local)
    print("Receiver build OK" if code == 0 else f"Receiver build failed with exit code {code}")


def run_flash_sender(local: dict) -> None:
    command = f'flash -b {local["baud"]} -p {local["sender_port"]}'
    code = run_idf(ROOT / "csi_send", command, local)
    print("Sender flash OK" if code == 0 else f"Sender flash failed with exit code {code}")


def run_flash_receiver(local: dict) -> None:
    command = f'flash -b {local["baud"]} -p {local["receiver_port"]}'
    code = run_idf(ROOT / "csi_recv", command, local)
    print("Receiver flash OK" if code == 0 else f"Receiver flash failed with exit code {code}")


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def ask_int(prompt: str, default: int, min_value: int = None, max_value: int = None) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue

        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}")
            continue
        if max_value is not None and value > max_value:
            print(f"Value must be <= {max_value}")
            continue
        return value


def ask_choice(prompt: str, choices: list, default: str) -> str:
    print(prompt)
    for idx, item in enumerate(choices, start=1):
        marker = " (default)" if item == default else ""
        print(f"  {idx}. {item}{marker}")
    while True:
        raw = input("Choose number (Enter for default): ").strip()
        if raw == "":
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print("Invalid selection.")


def live_mcs_algo_label(value: int) -> str:
    if 0 <= value < len(LIVE_MCS_ALGO_CHOICES):
        return LIVE_MCS_ALGO_CHOICES[value]
    return str(value)


def ask_live_mcs_algo(default: int) -> int:
    default_index = default if 0 <= default < len(LIVE_MCS_ALGO_CHOICES) else 0
    selected = ask_choice("Live MCS algorithm", LIVE_MCS_ALGO_CHOICES, LIVE_MCS_ALGO_CHOICES[default_index])
    return LIVE_MCS_ALGO_CHOICES.index(selected)


def ask_mac(prompt: str, default: str) -> str:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            parse_mac(raw)
            return raw.lower()
        except ValueError as exc:
            print(f"Invalid MAC: {exc}")


def ask_yes_no(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{suffix}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer y or n.")


# ---------------------------------------------------------------------------
# Advanced editing menus
# ---------------------------------------------------------------------------

def edit_sender_full(profile: dict) -> None:
    sender = profile["sender"]
    print("\nEdit sender parameters")
    sender["channel"] = ask_int("Channel", sender["channel"], 1, 14)
    sender["esp_now_rate"] = ask_choice("ESP-NOW PHY rate", RATE_CHOICES, sender["esp_now_rate"])
    sender["send_frequency"] = ask_int("Send frequency (packets/sec)", sender["send_frequency"], 1, 5000)
    sender["packet_pacing_enabled"] = 1 if ask_yes_no("Enable packet pacing", sender["packet_pacing_enabled"] == 1) else 0
    sender["ack_timing_mode"] = ask_int("ACK timing mode (1=async pipeline, 2=stop-and-wait)", sender["ack_timing_mode"], 1, 2)
    sender["rate_switch_mode"] = ask_int("Rate switch mode (0=time, 1=packet, 2=static, 3=random_sweep)", sender["rate_switch_mode"], 0, 3)
    sender["rate_switch_interval_sec"] = ask_int("Rate switch interval seconds", sender["rate_switch_interval_sec"], 1, 3600)
    sender["rate_switch_packet_count"] = ask_int("Rate switch packet count", sender["rate_switch_packet_count"], 1, 1000000)
    sender["rate_sweep_group_size"] = ask_int("Random-sweep packets per MCS step", sender["rate_sweep_group_size"], 1, 1000000)
    sender["payload_len"] = ask_int("ESP-NOW payload length", sender["payload_len"], 4, 250)
    sender["tx_power"] = ask_int("TX power (0.25 dBm units, range 8-84)", sender["tx_power"], 8, 84)
    sender["live_mcs_selection_enabled"] = 1 if ask_yes_no("Enable live MCS selection", sender["live_mcs_selection_enabled"] == 1) else 0
    sender["live_mcs_algo"] = ask_live_mcs_algo(sender["live_mcs_algo"])
    sender["live_mcs_min_index"] = ask_int("Live MCS min index", sender["live_mcs_min_index"], 0, 7)
    sender["live_mcs_max_index"] = ask_int("Live MCS max index", sender["live_mcs_max_index"], 0, 7)
    sender["minstrel_update_every_pkts"] = ask_int("Minstrel update every packets", sender["minstrel_update_every_pkts"], 1, 1000000)
    sender["minstrel_probe_every_pkts"] = ask_int("Minstrel probe every packets", sender["minstrel_probe_every_pkts"], 1, 1000000)
    sender["minstrel_ewma_alpha_den"] = ask_int("EWMA alpha denominator", sender["minstrel_ewma_alpha_den"], 1, 64)
    sender["minstrel_ewma_alpha_num"] = ask_int("EWMA alpha numerator", sender["minstrel_ewma_alpha_num"], 1, sender["minstrel_ewma_alpha_den"])
    sender["custom_policy_default_mcs"] = ask_int("Custom policy initial MCS", sender["custom_policy_default_mcs"], 0, 7)
    sender["live_mcs_decision_log_enabled"] = 1 if ask_yes_no("Enable ACK_POLICY decision logs", sender["live_mcs_decision_log_enabled"] == 1) else 0
    sender["remote_mcs_recommendation_enabled"] = 1 if ask_yes_no("Enable receiver recommendations", sender["remote_mcs_recommendation_enabled"] == 1) else 0
    sender["remote_mcs_min_confidence"] = ask_int("Remote recommendation min confidence", sender["remote_mcs_min_confidence"], 0, 100)
    sender["remote_mcs_max_age_ms"] = ask_int("Remote recommendation max age ms", sender["remote_mcs_max_age_ms"], 1, 5000)
    sender["dqn_remote_recommendation_enabled"] = 1 if ask_yes_no("Enable DQN receiver recommendations", sender["dqn_remote_recommendation_enabled"] == 1) else 0
    sender["dqn_default_mcs"] = ask_int("DQN initial MCS", sender["dqn_default_mcs"], 0, 7)
    sender["dqn_remote_min_confidence"] = ask_int("DQN recommendation min confidence", sender["dqn_remote_min_confidence"], 0, 100)
    sender["dqn_remote_max_age_ms"] = ask_int("DQN recommendation max age ms", sender["dqn_remote_max_age_ms"], 1, 5000)
    sender["dqn_remote_max_seq_gap"] = ask_int("DQN recommendation max sequence gap (0 disables)", sender["dqn_remote_max_seq_gap"], 0, 1000000)
    sender["dqn_failure_stepdown_enabled"] = 1 if ask_yes_no("Enable DQN failure stepdown", sender["dqn_failure_stepdown_enabled"] == 1) else 0
    sender["dqn_failure_stepdown_count"] = ask_int("DQN failure stepdown count", sender["dqn_failure_stepdown_count"], 1, 1000000)
    sender["dqn_log_enabled"] = 1 if ask_yes_no("Enable DQN sender logs", sender["dqn_log_enabled"] == 1) else 0
    validate_sender_profile(sender)


def edit_sender_dqn(profile: dict) -> None:
    sender = profile["sender"]
    while True:
        print("\nEdit sender receiver-policy (DQN frame) fields")
        print(f"1. dqn_remote_recommendation_enabled: {sender['dqn_remote_recommendation_enabled']}")
        print(f"2. dqn_default_mcs: {sender['dqn_default_mcs']}")
        print(f"3. dqn_remote_min_confidence: {sender['dqn_remote_min_confidence']}")
        print(f"4. dqn_remote_max_age_ms: {sender['dqn_remote_max_age_ms']}")
        print(f"5. dqn_remote_max_seq_gap: {sender['dqn_remote_max_seq_gap']}")
        print(f"6. dqn_failure_stepdown_enabled: {sender['dqn_failure_stepdown_enabled']}")
        print(f"7. dqn_failure_stepdown_count: {sender['dqn_failure_stepdown_count']}")
        print(f"8. dqn_log_enabled: {sender['dqn_log_enabled']}")
        print("0. Back")
        choice = input("Select DQN sender field: ").strip()

        if choice == "1":
            sender["dqn_remote_recommendation_enabled"] = 1 if ask_yes_no("Enable DQN receiver recommendations", sender["dqn_remote_recommendation_enabled"] == 1) else 0
        elif choice == "2":
            sender["dqn_default_mcs"] = ask_int("DQN initial MCS", sender["dqn_default_mcs"], 0, 7)
        elif choice == "3":
            sender["dqn_remote_min_confidence"] = ask_int("DQN recommendation min confidence", sender["dqn_remote_min_confidence"], 0, 100)
        elif choice == "4":
            sender["dqn_remote_max_age_ms"] = ask_int("DQN recommendation max age ms", sender["dqn_remote_max_age_ms"], 1, 5000)
        elif choice == "5":
            sender["dqn_remote_max_seq_gap"] = ask_int("DQN recommendation max sequence gap (0 disables)", sender["dqn_remote_max_seq_gap"], 0, 1000000)
        elif choice == "6":
            sender["dqn_failure_stepdown_enabled"] = 1 if ask_yes_no("Enable DQN failure stepdown", sender["dqn_failure_stepdown_enabled"] == 1) else 0
        elif choice == "7":
            sender["dqn_failure_stepdown_count"] = ask_int("DQN failure stepdown count", sender["dqn_failure_stepdown_count"], 1, 1000000)
        elif choice == "8":
            sender["dqn_log_enabled"] = 1 if ask_yes_no("Enable DQN sender logs", sender["dqn_log_enabled"] == 1) else 0
        elif choice == "0":
            return
        else:
            print("Invalid option.")
            continue

        validate_sender_profile(sender)
        save_profile(profile)


def edit_sender_live_mcs(profile: dict) -> None:
    sender = profile["sender"]
    while True:
        algo_name = live_mcs_algo_label(sender["live_mcs_algo"])
        print("\nEdit sender live MCS policy")
        print(f"1. live_mcs_selection_enabled: {sender['live_mcs_selection_enabled']}")
        print(f"2. live_mcs_algo: {sender['live_mcs_algo']} ({algo_name})")
        print(f"3. live_mcs_min_index: {sender['live_mcs_min_index']}")
        print(f"4. live_mcs_max_index: {sender['live_mcs_max_index']}")
        print(f"5. minstrel_update_every_pkts: {sender['minstrel_update_every_pkts']}")
        print(f"6. minstrel_probe_every_pkts: {sender['minstrel_probe_every_pkts']}")
        print(f"7. minstrel_ewma_alpha_den: {sender['minstrel_ewma_alpha_den']}")
        print(f"8. minstrel_ewma_alpha_num: {sender['minstrel_ewma_alpha_num']}")
        print(f"9. custom_policy_initial_mcs: {sender['custom_policy_default_mcs']}")
        print(f"10. live_mcs_decision_log_enabled: {sender['live_mcs_decision_log_enabled']}")
        print(f"11. remote_mcs_recommendation_enabled: {sender['remote_mcs_recommendation_enabled']}")
        print(f"12. remote_mcs_min_confidence: {sender['remote_mcs_min_confidence']}")
        print(f"13. remote_mcs_max_age_ms: {sender['remote_mcs_max_age_ms']}")
        print("14. Edit DQN policy fields")
        print("15. Edit all sender fields")
        print("0. Back")
        choice = input("Select live MCS field: ").strip()

        if choice == "1":
            sender["live_mcs_selection_enabled"] = 1 if ask_yes_no("Enable live MCS selection", sender["live_mcs_selection_enabled"] == 1) else 0
        elif choice == "2":
            sender["live_mcs_algo"] = ask_live_mcs_algo(sender["live_mcs_algo"])
        elif choice == "3":
            sender["live_mcs_min_index"] = ask_int("Live MCS min index", sender["live_mcs_min_index"], 0, 7)
        elif choice == "4":
            sender["live_mcs_max_index"] = ask_int("Live MCS max index", sender["live_mcs_max_index"], 0, 7)
        elif choice == "5":
            sender["minstrel_update_every_pkts"] = ask_int("Minstrel update every packets", sender["minstrel_update_every_pkts"], 1, 1000000)
        elif choice == "6":
            sender["minstrel_probe_every_pkts"] = ask_int("Minstrel probe every packets", sender["minstrel_probe_every_pkts"], 1, 1000000)
        elif choice == "7":
            sender["minstrel_ewma_alpha_den"] = ask_int("EWMA alpha denominator", sender["minstrel_ewma_alpha_den"], 1, 64)
        elif choice == "8":
            sender["minstrel_ewma_alpha_num"] = ask_int("EWMA alpha numerator", sender["minstrel_ewma_alpha_num"], 1, sender["minstrel_ewma_alpha_den"])
        elif choice == "9":
            sender["custom_policy_default_mcs"] = ask_int("Custom policy initial MCS", sender["custom_policy_default_mcs"], 0, 7)
        elif choice == "10":
            sender["live_mcs_decision_log_enabled"] = 1 if ask_yes_no("Enable ACK_POLICY decision logs", sender["live_mcs_decision_log_enabled"] == 1) else 0
        elif choice == "11":
            sender["remote_mcs_recommendation_enabled"] = 1 if ask_yes_no("Enable receiver recommendations", sender["remote_mcs_recommendation_enabled"] == 1) else 0
        elif choice == "12":
            sender["remote_mcs_min_confidence"] = ask_int("Remote recommendation min confidence", sender["remote_mcs_min_confidence"], 0, 100)
        elif choice == "13":
            sender["remote_mcs_max_age_ms"] = ask_int("Remote recommendation max age ms", sender["remote_mcs_max_age_ms"], 1, 5000)
        elif choice == "14":
            edit_sender_dqn(profile)
        elif choice == "15":
            edit_sender_full(profile)
        elif choice == "0":
            return
        else:
            print("Invalid option.")
            continue

        validate_sender_profile(sender)
        save_profile(profile)


def edit_sender(profile: dict) -> None:
    sender = profile["sender"]
    while True:
        print("\nEdit sender parameters (quick)")
        print(f"1. channel: {sender['channel']}")
        print(f"2. esp_now_rate: {sender['esp_now_rate']}")
        print(f"3. send_frequency: {sender['send_frequency']}")
        print(f"4. packet_pacing_enabled: {sender['packet_pacing_enabled']}")
        print(f"5. ack_timing_mode: {sender['ack_timing_mode']} (1=async, 2=stop-and-wait)")
        print(f"6. rate_switch_mode: {sender['rate_switch_mode']} (0=time, 1=packet, 2=static, 3=random_sweep)")
        print(f"7. rate_switch_interval_sec: {sender['rate_switch_interval_sec']}")
        print(f"8. rate_switch_packet_count: {sender['rate_switch_packet_count']}")
        print(f"9. rate_sweep_group_size: {sender['rate_sweep_group_size']}")
        print(f"10. payload_len: {sender['payload_len']}")
        print(f"11. tx_power: {sender['tx_power']} (0.25 dBm units, {sender['tx_power'] * 0.25:.1f} dBm nominal)")
        print(f"12. live_mcs_selection_enabled: {sender['live_mcs_selection_enabled']}")
        algo_name = live_mcs_algo_label(sender["live_mcs_algo"])
        print(f"13. live_mcs_algo: {sender['live_mcs_algo']} ({algo_name})")
        print("14. Edit live MCS policy fields")
        print("15. Edit DQN policy fields")
        print("16. Edit all sender fields")
        print("0. Back")
        choice = input("Select sender field: ").strip()

        if choice == "1":
            sender["channel"] = ask_int("Channel", sender["channel"], 1, 14)
        elif choice == "2":
            sender["esp_now_rate"] = ask_choice("ESP-NOW PHY rate", RATE_CHOICES, sender["esp_now_rate"])
        elif choice == "3":
            sender["send_frequency"] = ask_int("Send frequency (packets/sec)", sender["send_frequency"], 1, 5000)
        elif choice == "4":
            sender["packet_pacing_enabled"] = 1 if ask_yes_no("Enable packet pacing", sender["packet_pacing_enabled"] == 1) else 0
        elif choice == "5":
            sender["ack_timing_mode"] = ask_int("ACK timing mode (1=async pipeline, 2=stop-and-wait)", sender["ack_timing_mode"], 1, 2)
        elif choice == "6":
            sender["rate_switch_mode"] = ask_int("Rate switch mode (0=time, 1=packet, 2=static, 3=random_sweep)", sender["rate_switch_mode"], 0, 3)
        elif choice == "7":
            sender["rate_switch_interval_sec"] = ask_int("Rate switch interval seconds", sender["rate_switch_interval_sec"], 1, 3600)
        elif choice == "8":
            sender["rate_switch_packet_count"] = ask_int("Rate switch packet count", sender["rate_switch_packet_count"], 1, 1000000)
        elif choice == "9":
            sender["rate_sweep_group_size"] = ask_int("Random-sweep packets per MCS step", sender["rate_sweep_group_size"], 1, 1000000)
        elif choice == "10":
            sender["payload_len"] = ask_int("ESP-NOW payload length", sender["payload_len"], 4, 250)
        elif choice == "11":
            sender["tx_power"] = ask_int("TX power (0.25 dBm units, range 8-84)", sender["tx_power"], 8, 84)
        elif choice == "12":
            sender["live_mcs_selection_enabled"] = 1 if ask_yes_no("Enable live MCS selection", sender["live_mcs_selection_enabled"] == 1) else 0
        elif choice == "13":
            sender["live_mcs_algo"] = ask_live_mcs_algo(sender["live_mcs_algo"])
        elif choice == "14":
            edit_sender_live_mcs(profile)
            continue
        elif choice == "15":
            edit_sender_dqn(profile)
            continue
        elif choice == "16":
            edit_sender_full(profile)
        elif choice == "0":
            return
        else:
            print("Invalid option.")
            continue

        validate_sender_profile(sender)
        save_profile(profile)


def edit_receiver_full(profile: dict) -> None:
    receiver = profile["receiver"]
    print("\nEdit receiver parameters")
    receiver["channel"] = ask_int("Channel", receiver["channel"], 1, 14)
    receiver["esp_now_rate"] = ask_choice("ESP-NOW PHY rate", RATE_CHOICES, receiver["esp_now_rate"])
    receiver["force_gain"] = 1 if ask_yes_no("Force gain", receiver["force_gain"] == 1) else 0
    receiver["gain_control"] = 1 if ask_yes_no("Enable gain control", receiver["gain_control"] == 1) else 0
    receiver["custom_mcs_recommendation_enabled"] = 1 if ask_yes_no("Enable legacy reward-model recommendations", receiver["custom_mcs_recommendation_enabled"] == 1) else 0
    receiver["dqn_mcs_recommendation_enabled"] = 1 if ask_yes_no("Enable receiver live-policy recommendations", receiver["dqn_mcs_recommendation_enabled"] == 1) else 0
    receiver["mcs_policy_model"] = ask_int("Live policy model (1=bandit link_v3c, 0=DQN Q-network)", receiver["mcs_policy_model"], 0, 1)
    receiver["dqn_recommendation_every_n_packets"] = ask_int("Live policy recommendation every N packets", receiver["dqn_recommendation_every_n_packets"], 1, 1000000)
    receiver["dqn_warmup_packets"] = ask_int("Live policy warmup packets", receiver["dqn_warmup_packets"], 0, 1000000)
    receiver["dqn_control_interval_ms"] = ask_int("Live policy control interval ms", receiver["dqn_control_interval_ms"], 1, 1000000)
    receiver["dqn_stale_max_age_packets"] = ask_int("Live policy stale max age packets", receiver["dqn_stale_max_age_packets"], 1, 1000000)
    receiver["dqn_log_enabled"] = 1 if ask_yes_no("Enable live policy receiver logs", receiver["dqn_log_enabled"] == 1) else 0
    validate_receiver_profile(receiver)


def edit_receiver_dqn(profile: dict) -> None:
    receiver = profile["receiver"]
    while True:
        print("\nEdit receiver live-policy recommendations")
        print(f"1. dqn_mcs_recommendation_enabled: {receiver['dqn_mcs_recommendation_enabled']}")
        print(f"2. mcs_policy_model: {receiver['mcs_policy_model']} (1=bandit link_v3c, 0=DQN Q-network)")
        print(f"3. dqn_recommendation_every_n_packets: {receiver['dqn_recommendation_every_n_packets']}")
        print(f"4. dqn_warmup_packets: {receiver['dqn_warmup_packets']}")
        print(f"5. dqn_control_interval_ms: {receiver['dqn_control_interval_ms']}")
        print(f"6. dqn_stale_max_age_packets: {receiver['dqn_stale_max_age_packets']}")
        print(f"7. dqn_log_enabled: {receiver['dqn_log_enabled']}")
        print("0. Back")
        choice = input("Select live-policy receiver field: ").strip()

        if choice == "1":
            receiver["dqn_mcs_recommendation_enabled"] = 1 if ask_yes_no("Enable receiver live-policy recommendations", receiver["dqn_mcs_recommendation_enabled"] == 1) else 0
        elif choice == "2":
            receiver["mcs_policy_model"] = ask_int("Live policy model (1=bandit link_v3c, 0=DQN Q-network)", receiver["mcs_policy_model"], 0, 1)
        elif choice == "3":
            receiver["dqn_recommendation_every_n_packets"] = ask_int("Live policy recommendation every N packets", receiver["dqn_recommendation_every_n_packets"], 1, 1000000)
        elif choice == "4":
            receiver["dqn_warmup_packets"] = ask_int("Live policy warmup packets", receiver["dqn_warmup_packets"], 0, 1000000)
        elif choice == "5":
            receiver["dqn_control_interval_ms"] = ask_int("Live policy control interval ms", receiver["dqn_control_interval_ms"], 1, 1000000)
        elif choice == "6":
            receiver["dqn_stale_max_age_packets"] = ask_int("Live policy stale max age packets", receiver["dqn_stale_max_age_packets"], 1, 1000000)
        elif choice == "7":
            receiver["dqn_log_enabled"] = 1 if ask_yes_no("Enable live policy receiver logs", receiver["dqn_log_enabled"] == 1) else 0
        elif choice == "0":
            return
        else:
            print("Invalid option.")
            continue

        validate_receiver_profile(receiver)
        save_profile(profile)


def edit_receiver(profile: dict) -> None:
    receiver = profile["receiver"]
    while True:
        print("\nEdit receiver parameters (quick)")
        print(f"1. channel: {receiver['channel']}")
        print(f"2. esp_now_rate: {receiver['esp_now_rate']}")
        print(f"3. force_gain: {receiver['force_gain']}")
        print(f"4. gain_control: {receiver['gain_control']}")
        print(f"5. custom_mcs_recommendation_enabled: {receiver['custom_mcs_recommendation_enabled']}")
        print(f"6. dqn_mcs_recommendation_enabled: {receiver['dqn_mcs_recommendation_enabled']}")
        print(f"7. mcs_policy_model: {receiver['mcs_policy_model']} (1=bandit, 0=DQN)")
        print("8. Edit live-policy receiver fields")
        print("9. Edit all receiver fields")
        print("0. Back")
        choice = input("Select receiver field: ").strip()

        if choice == "1":
            receiver["channel"] = ask_int("Channel", receiver["channel"], 1, 14)
        elif choice == "2":
            receiver["esp_now_rate"] = ask_choice("ESP-NOW PHY rate", RATE_CHOICES, receiver["esp_now_rate"])
        elif choice == "3":
            receiver["force_gain"] = 1 if ask_yes_no("Force gain", receiver["force_gain"] == 1) else 0
        elif choice == "4":
            receiver["gain_control"] = 1 if ask_yes_no("Enable gain control", receiver["gain_control"] == 1) else 0
        elif choice == "5":
            receiver["custom_mcs_recommendation_enabled"] = 1 if ask_yes_no("Enable legacy reward-model recommendations", receiver["custom_mcs_recommendation_enabled"] == 1) else 0
        elif choice == "6":
            receiver["dqn_mcs_recommendation_enabled"] = 1 if ask_yes_no("Enable receiver live-policy recommendations", receiver["dqn_mcs_recommendation_enabled"] == 1) else 0
        elif choice == "7":
            receiver["mcs_policy_model"] = ask_int("Live policy model (1=bandit link_v3c, 0=DQN Q-network)", receiver["mcs_policy_model"], 0, 1)
        elif choice == "8":
            edit_receiver_dqn(profile)
            continue
        elif choice == "9":
            edit_receiver_full(profile)
        elif choice == "0":
            return
        else:
            print("Invalid option.")
            continue

        validate_receiver_profile(receiver)
        save_profile(profile)


def edit_shared(profile: dict) -> None:
    shared = profile["shared"]
    while True:
        print("\nEdit shared MAC parameters (quick)")
        print(f"1. sender_mac: {shared['sender_mac']}")
        print(f"2. receiver_mac: {shared['receiver_mac']}")
        print("3. Edit both MACs")
        print("0. Back")
        choice = input("Select shared field: ").strip()

        if choice == "1":
            shared["sender_mac"] = ask_mac("Sender MAC", shared["sender_mac"])
            save_profile(profile)
        elif choice == "2":
            shared["receiver_mac"] = ask_mac("Receiver MAC", shared["receiver_mac"])
            save_profile(profile)
        elif choice == "3":
            shared["sender_mac"] = ask_mac("Sender MAC", shared["sender_mac"])
            shared["receiver_mac"] = ask_mac("Receiver MAC", shared["receiver_mac"])
            save_profile(profile)
        elif choice == "0":
            return
        else:
            print("Invalid option.")


def edit_local(local: dict) -> None:
    while True:
        print(f"\nEdit host-local settings (this machine only: {LOCAL_PATH.name}, not synced)")
        print(f"1. sender_port: {local['sender_port']}")
        print(f"2. receiver_port: {local['receiver_port']}")
        print(f"3. baud: {local['baud']}")
        print(f"4. export_script: {local['export_script']}")
        print(f"5. build_dir: {local['build_dir']}")
        print("0. Back")
        choice = input("Select local field: ").strip()

        if choice == "1":
            value = input(f"Sender serial port [{local['sender_port']}]: ").strip()
            if value:
                local["sender_port"] = value
                save_local(local)
        elif choice == "2":
            value = input(f"Receiver serial port [{local['receiver_port']}]: ").strip()
            if value:
                local["receiver_port"] = value
                save_local(local)
        elif choice == "3":
            local["baud"] = ask_int("Flash baud", int(local["baud"]), 115200, 2000000)
            save_local(local)
        elif choice == "4":
            value = input(f"ESP-IDF export script [{local['export_script']}]: ").strip()
            if value:
                local["export_script"] = value
                save_local(local)
        elif choice == "5":
            value = input(f"Build directory [{local['build_dir']}]: ").strip()
            if value:
                local["build_dir"] = value
                save_local(local)
        elif choice == "0":
            return
        else:
            print("Invalid option.")


def show_profile(profile: dict, local: dict) -> None:
    print("\nCurrent shared profile")
    print(json.dumps(profile, indent=2))
    print("\nHost-local settings (not synced)")
    print(json.dumps(local, indent=2))
    print(f"\nDetected mode: {guess_current_mode(profile)}")


def apply_and_save(profile: dict) -> bool:
    try:
        apply_profile_to_sources(profile)
        save_profile(profile)
        print("Applied profile to source files and saved profile JSON.")
        return True
    except Exception as exc:
        print(f"Apply failed: {exc}")
        return False


def run_menu() -> int:
    profile = load_profile()
    local = load_local()

    while True:
        print("\n=== ESP CSI Sender/Receiver Menu ===")
        print(f"(host: {'windows' if IS_WINDOWS else 'linux'}, mode: {guess_current_mode(profile)})")
        print("1. Select algorithm mode (preset for both devices)")
        print("2. Show current profile")
        print("3. Edit sender parameters (advanced)")
        print("4. Edit receiver parameters (advanced)")
        print("5. Edit shared MAC parameters")
        print("6. Edit host-local flash/build settings")
        print("7. Apply profile to C source files")
        print("8. Build sender")
        print("9. Build receiver")
        print("10. Flash sender")
        print("11. Flash receiver")
        print("12. Apply + flash sender")
        print("13. Apply + flash receiver")
        print("14. Apply + flash both")
        print("0. Exit")

        choice = input("Select option: ").strip()

        if choice == "1":
            select_mode(profile, local)
        elif choice == "2":
            show_profile(profile, local)
        elif choice == "3":
            edit_sender(profile)
        elif choice == "4":
            edit_receiver(profile)
        elif choice == "5":
            edit_shared(profile)
        elif choice == "6":
            edit_local(local)
        elif choice == "7":
            apply_and_save(profile)
        elif choice == "8":
            run_build_sender(local)
        elif choice == "9":
            run_build_receiver(local)
        elif choice == "10":
            run_flash_sender(local)
        elif choice == "11":
            run_flash_receiver(local)
        elif choice == "12":
            if apply_and_save(profile):
                run_flash_sender(local)
        elif choice == "13":
            if apply_and_save(profile):
                run_flash_receiver(local)
        elif choice == "14":
            if apply_and_save(profile):
                run_flash_sender(local)
                run_flash_receiver(local)
        elif choice == "0":
            return 0
        else:
            print("Invalid option.")


if __name__ == "__main__":
    try:
        sys.exit(run_menu())
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(130)
