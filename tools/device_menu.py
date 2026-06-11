#!/usr/bin/env python3
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SENDER_MAIN = ROOT / "csi_send" / "main" / "app_main.c"
RECEIVER_MAIN = ROOT / "csi_recv" / "main" / "app_main.c"
PROFILE_PATH = Path(__file__).resolve().parent / "device_menu_profile.json"

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

DEFAULT_PROFILE = {
    "sender": {
        "channel": 11,
        "esp_now_rate": "WIFI_PHY_RATE_MCS0_LGI",
        "send_frequency": 100,
        "packet_pacing_enabled": 1,
        "rate_switch_mode": 2,
        "rate_switch_interval_sec": 10,
        "rate_switch_packet_count": 1000,
        "payload_len": 128,
        "tx_power": 84,
    },
    "receiver": {
        "channel": 11,
        "esp_now_rate": "WIFI_PHY_RATE_MCS0_LGI",
        "force_gain": 1,
        "gain_control": 1,
    },
    "shared": {
        "sender_mac": "1a:00:00:00:00:00",
        "receiver_mac": "1a:00:00:00:00:01",
    },
    "flash": {
        "sender_port": "/dev/ttyUSB0",
        "receiver_port": "/dev/ttyUSB0",
        "baud": 921600,
        "export_script": "$HOME/esp/esp-idf/export.sh",
    },
}


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

    return merged


def save_profile(profile: dict) -> None:
    PROFILE_PATH.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")


def apply_profile_to_sources(profile: dict) -> None:
    sender_text = read_text(SENDER_MAIN)
    receiver_text = read_text(RECEIVER_MAIN)

    sender = profile["sender"]
    receiver = profile["receiver"]
    shared = profile["shared"]

    sender_text = replace_define(sender_text, "CONFIG_LESS_INTERFERENCE_CHANNEL", str(sender["channel"]))
    sender_text = replace_define(sender_text, "CONFIG_ESP_NOW_RATE", sender["esp_now_rate"])
    sender_text = replace_define(sender_text, "CONFIG_SEND_FREQUENCY", str(sender["send_frequency"]))
    sender_text = replace_define(sender_text, "CONFIG_PACKET_PACING_ENABLED", str(sender["packet_pacing_enabled"]))
    sender_text = replace_define(sender_text, "CONFIG_RATE_SWITCH_MODE", str(sender["rate_switch_mode"]))
    sender_text = replace_define(sender_text, "CONFIG_RATE_SWITCH_INTERVAL_SEC", str(sender["rate_switch_interval_sec"]))
    sender_text = replace_define(sender_text, "CONFIG_RATE_SWITCH_PACKET_COUNT", str(sender["rate_switch_packet_count"]))
    sender_text = replace_define(sender_text, "CONFIG_ESP_NOW_PAYLOAD_LEN", str(sender["payload_len"]))
    sender_text = replace_define(sender_text, "CONFIG_WIFI_TX_POWER", str(sender["tx_power"]))
    sender_text = replace_mac_array(sender_text, "CONFIG_CSI_SEND_MAC", shared["sender_mac"])
    sender_text = replace_mac_array(sender_text, "CONFIG_CSI_RECV_MAC", shared["receiver_mac"])

    receiver_text = replace_define(receiver_text, "CONFIG_LESS_INTERFERENCE_CHANNEL", str(receiver["channel"]))
    receiver_text = replace_define(receiver_text, "CONFIG_ESP_NOW_RATE", receiver["esp_now_rate"])
    receiver_text = replace_define(receiver_text, "CONFIG_FORCE_GAIN", str(receiver["force_gain"]))
    receiver_text = replace_define(receiver_text, "CONFIG_GAIN_CONTROL", str(receiver["gain_control"]))
    receiver_text = replace_mac_array(receiver_text, "CONFIG_CSI_SEND_MAC", shared["sender_mac"])
    receiver_text = replace_mac_array(receiver_text, "CONFIG_CSI_RECV_MAC", shared["receiver_mac"])

    write_text(SENDER_MAIN, sender_text)
    write_text(RECEIVER_MAIN, receiver_text)


def run_idf(target_dir: Path, command: str, export_script: str) -> int:
    cmd = f'. {export_script} && idf.py {command}'
    print(f"\n[run] {target_dir}: {cmd}\n")
    proc = subprocess.run(["bash", "-lc", cmd], cwd=str(target_dir))
    return proc.returncode


def run_build_sender(profile: dict) -> None:
    code = run_idf(ROOT / "csi_send", "build", profile["flash"]["export_script"])
    print("Sender build OK" if code == 0 else f"Sender build failed with exit code {code}")


def run_build_receiver(profile: dict) -> None:
    code = run_idf(ROOT / "csi_recv", "build", profile["flash"]["export_script"])
    print("Receiver build OK" if code == 0 else f"Receiver build failed with exit code {code}")


def run_flash_sender(profile: dict) -> None:
    flash = profile["flash"]
    command = f'flash -b {flash["baud"]} -p {flash["sender_port"]}'
    code = run_idf(ROOT / "csi_send", command, flash["export_script"])
    print("Sender flash OK" if code == 0 else f"Sender flash failed with exit code {code}")


def run_flash_receiver(profile: dict) -> None:
    flash = profile["flash"]
    command = f'flash -b {flash["baud"]} -p {flash["receiver_port"]}'
    code = run_idf(ROOT / "csi_recv", command, flash["export_script"])
    print("Receiver flash OK" if code == 0 else f"Receiver flash failed with exit code {code}")


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


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
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


def edit_sender_full(profile: dict) -> None:
    sender = profile["sender"]
    print("\nEdit sender parameters")
    sender["channel"] = ask_int("Channel", sender["channel"], 1, 14)
    sender["esp_now_rate"] = ask_choice("ESP-NOW PHY rate", RATE_CHOICES, sender["esp_now_rate"])
    sender["send_frequency"] = ask_int("Send frequency (packets/sec)", sender["send_frequency"], 1, 5000)
    sender["packet_pacing_enabled"] = 1 if ask_yes_no("Enable packet pacing", sender["packet_pacing_enabled"] == 1) else 0
    sender["rate_switch_mode"] = ask_int("Rate switch mode (0=time, 1=packet, 2=static)", sender["rate_switch_mode"], 0, 2)
    sender["rate_switch_interval_sec"] = ask_int("Rate switch interval seconds", sender["rate_switch_interval_sec"], 1, 3600)
    sender["rate_switch_packet_count"] = ask_int("Rate switch packet count", sender["rate_switch_packet_count"], 1, 1000000)
    sender["payload_len"] = ask_int("ESP-NOW payload length", sender["payload_len"], 4, 250)
    sender["tx_power"] = ask_int("TX power (0.25 dBm units, range 8-84)", sender["tx_power"], 8, 84)


def edit_sender(profile: dict) -> None:
    sender = profile["sender"]
    while True:
        print("\nEdit sender parameters (quick)")
        print(f"1. channel: {sender['channel']}")
        print(f"2. esp_now_rate: {sender['esp_now_rate']}")
        print(f"3. send_frequency: {sender['send_frequency']}")
        print(f"4. packet_pacing_enabled: {sender['packet_pacing_enabled']}")
        print(f"5. rate_switch_mode: {sender['rate_switch_mode']}")
        print(f"6. rate_switch_interval_sec: {sender['rate_switch_interval_sec']}")
        print(f"7. rate_switch_packet_count: {sender['rate_switch_packet_count']}")
        print(f"8. payload_len: {sender['payload_len']}")
        print(f"9. tx_power: {sender['tx_power']} (0.25 dBm units, {sender['tx_power'] * 0.25:.1f} dBm nominal)")
        print("10. Edit all sender fields")
        print("0. Back")
        choice = input("Select sender field: ").strip()

        if choice == "1":
            sender["channel"] = ask_int("Channel", sender["channel"], 1, 14)
            save_profile(profile)
        elif choice == "2":
            sender["esp_now_rate"] = ask_choice("ESP-NOW PHY rate", RATE_CHOICES, sender["esp_now_rate"])
            save_profile(profile)
        elif choice == "3":
            sender["send_frequency"] = ask_int("Send frequency (packets/sec)", sender["send_frequency"], 1, 5000)
            save_profile(profile)
        elif choice == "4":
            sender["packet_pacing_enabled"] = 1 if ask_yes_no("Enable packet pacing", sender["packet_pacing_enabled"] == 1) else 0
            save_profile(profile)
        elif choice == "5":
            sender["rate_switch_mode"] = ask_int("Rate switch mode (0=time, 1=packet, 2=static)", sender["rate_switch_mode"], 0, 2)
            save_profile(profile)
        elif choice == "6":
            sender["rate_switch_interval_sec"] = ask_int("Rate switch interval seconds", sender["rate_switch_interval_sec"], 1, 3600)
            save_profile(profile)
        elif choice == "7":
            sender["rate_switch_packet_count"] = ask_int("Rate switch packet count", sender["rate_switch_packet_count"], 1, 1000000)
            save_profile(profile)
        elif choice == "8":
            sender["payload_len"] = ask_int("ESP-NOW payload length", sender["payload_len"], 4, 250)
            save_profile(profile)
        elif choice == "9":
            sender["tx_power"] = ask_int("TX power (0.25 dBm units, range 8-84)", sender["tx_power"], 8, 84)
            save_profile(profile)
        elif choice == "10":
            edit_sender_full(profile)
            save_profile(profile)
        elif choice == "0":
            return
        else:
            print("Invalid option.")


def edit_receiver_full(profile: dict) -> None:
    receiver = profile["receiver"]
    print("\nEdit receiver parameters")
    receiver["channel"] = ask_int("Channel", receiver["channel"], 1, 14)
    receiver["esp_now_rate"] = ask_choice("ESP-NOW PHY rate", RATE_CHOICES, receiver["esp_now_rate"])
    receiver["force_gain"] = 1 if ask_yes_no("Force gain", receiver["force_gain"] == 1) else 0
    receiver["gain_control"] = 1 if ask_yes_no("Enable gain control", receiver["gain_control"] == 1) else 0


def edit_receiver(profile: dict) -> None:
    receiver = profile["receiver"]
    while True:
        print("\nEdit receiver parameters (quick)")
        print(f"1. channel: {receiver['channel']}")
        print(f"2. esp_now_rate: {receiver['esp_now_rate']}")
        print(f"3. force_gain: {receiver['force_gain']}")
        print(f"4. gain_control: {receiver['gain_control']}")
        print("5. Edit all receiver fields")
        print("0. Back")
        choice = input("Select receiver field: ").strip()

        if choice == "1":
            receiver["channel"] = ask_int("Channel", receiver["channel"], 1, 14)
            save_profile(profile)
        elif choice == "2":
            receiver["esp_now_rate"] = ask_choice("ESP-NOW PHY rate", RATE_CHOICES, receiver["esp_now_rate"])
            save_profile(profile)
        elif choice == "3":
            receiver["force_gain"] = 1 if ask_yes_no("Force gain", receiver["force_gain"] == 1) else 0
            save_profile(profile)
        elif choice == "4":
            receiver["gain_control"] = 1 if ask_yes_no("Enable gain control", receiver["gain_control"] == 1) else 0
            save_profile(profile)
        elif choice == "5":
            edit_receiver_full(profile)
            save_profile(profile)
        elif choice == "0":
            return
        else:
            print("Invalid option.")


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


def edit_flash(profile: dict) -> None:
    flash = profile["flash"]
    while True:
        print("\nEdit flashing parameters (quick)")
        print(f"1. sender_port: {flash['sender_port']}")
        print(f"2. receiver_port: {flash['receiver_port']}")
        print(f"3. baud: {flash['baud']}")
        print(f"4. export_script: {flash['export_script']}")
        print("5. Edit all flash fields")
        print("0. Back")
        choice = input("Select flash field: ").strip()

        if choice == "1":
            sender_port = input(f"Sender serial port [{flash['sender_port']}]: ").strip()
            if sender_port:
                flash["sender_port"] = sender_port
                save_profile(profile)
        elif choice == "2":
            receiver_port = input(f"Receiver serial port [{flash['receiver_port']}]: ").strip()
            if receiver_port:
                flash["receiver_port"] = receiver_port
                save_profile(profile)
        elif choice == "3":
            flash["baud"] = ask_int("Flash baud", int(flash["baud"]), 115200, 2000000)
            save_profile(profile)
        elif choice == "4":
            export_script = input(f"ESP-IDF export script [{flash['export_script']}]: ").strip()
            if export_script:
                flash["export_script"] = export_script
                save_profile(profile)
        elif choice == "5":
            sender_port = input(f"Sender serial port [{flash['sender_port']}]: ").strip()
            if sender_port:
                flash["sender_port"] = sender_port

            receiver_port = input(f"Receiver serial port [{flash['receiver_port']}]: ").strip()
            if receiver_port:
                flash["receiver_port"] = receiver_port

            flash["baud"] = ask_int("Flash baud", int(flash["baud"]), 115200, 2000000)

            export_script = input(f"ESP-IDF export script [{flash['export_script']}]: ").strip()
            if export_script:
                flash["export_script"] = export_script
            save_profile(profile)
        elif choice == "0":
            return
        else:
            print("Invalid option.")


def show_profile(profile: dict) -> None:
    print("\nCurrent profile")
    print(json.dumps(profile, indent=2))


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

    while True:
        print("\n=== ESP CSI Sender/Receiver Menu ===")
        print("1. Show current profile")
        print("2. Edit sender parameters")
        print("3. Edit receiver parameters")
        print("4. Edit shared MAC parameters")
        print("5. Edit flash/build parameters")
        print("6. Apply profile to C source files")
        print("7. Build sender")
        print("8. Build receiver")
        print("9. Flash sender")
        print("10. Flash receiver")
        print("11. Apply + flash sender")
        print("12. Apply + flash receiver")
        print("13. Apply + flash both")
        print("0. Exit")

        choice = input("Select option: ").strip()

        if choice == "1":
            show_profile(profile)
        elif choice == "2":
            edit_sender(profile)
        elif choice == "3":
            edit_receiver(profile)
        elif choice == "4":
            edit_shared(profile)
        elif choice == "5":
            edit_flash(profile)
        elif choice == "6":
            apply_and_save(profile)
        elif choice == "7":
            run_build_sender(profile)
        elif choice == "8":
            run_build_receiver(profile)
        elif choice == "9":
            run_flash_sender(profile)
        elif choice == "10":
            run_flash_receiver(profile)
        elif choice == "11":
            if apply_and_save(profile):
                run_flash_sender(profile)
        elif choice == "12":
            if apply_and_save(profile):
                run_flash_receiver(profile)
        elif choice == "13":
            if apply_and_save(profile):
                run_flash_sender(profile)
                run_flash_receiver(profile)
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