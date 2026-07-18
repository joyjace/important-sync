# Sender/Receiver Device Menu (v2)

`tools/device_menu.py` manages parameters for `csi_send/main/app_main.c` and
`csi_recv/main/app_main.c`, and can build/flash each board. It does **not**
modify `csi_recv_router`.

## Run

From the repository root on any host (lab laptop or a device Raspberry Pi):

```bash
python3 tools/device_menu.py
```

## Algorithm mode presets

Menu option 1 selects a complete algorithm mode. One preset configures **both**
the sender and the receiver consistently (all coupled flags at once), saves the
shared profile, and rewrites both C sources:

| # | Mode | Sender | Receiver |
|---|------|--------|----------|
| 1 | `minstrel` | live, MINSTREL_LIKE, no remote recs | pure CSI logger |
| 2 | `bandit` | live, RECEIVER_POLICY frames | live path, bandit model (`link_v3c`) |
| 3 | `dqn` | live, RECEIVER_POLICY frames | live path, DQN Q-network |
| 4 | `reward_model` | live, CUSTOM_POLICY recs | legacy reward-model path |
| 5 | `static` | fixed MCS (asks which), no adaptation | pure CSI logger |
| 6 | `random_sweep` | RANDOM_SWEEP collection (asks group size) | pure CSI logger |

Before applying, the menu shows exactly which values will change. For model
modes it verifies the required generated header is present and matches
(`generated_bandit_model.h` must be a `link_v3c` export, etc.) and prints the
export command if not.

After applying, it offers to build+flash **the board attached to this host**
(sender / receiver / both). On the two-Pi setup: apply the mode once, sync the
sources to both Pis (git or `sync_selected_files.sh` → `important-sync`), then
on each Pi build+flash only its own device.

## Shared vs host-local configuration

- `tools/device_menu_profile.json` — the device configuration (channel, rates,
  algorithm flags, MACs). **Synced** between hosts so the pair stays
  consistent.
- `tools/device_menu_local.json` — this machine's serial ports, ESP-IDF export
  script, and build directory. **Never synced** (gitignored and excluded from
  `sync_selected_files.sh`). Created on first run with per-OS defaults:
  Linux/Pi → `/dev/ttyUSB0`, `export.sh`, `build`; Windows → `COM3`/`COM4`,
  `export.bat`, `build_win`. Edit via menu option 6.

## What the menu writes to the C sources

Sender (`csi_send/main/app_main.c`): channel, ESP-NOW rate, send frequency,
pacing, ACK timing mode, rate-switch mode (`0`=time, `1`=packet, `2`=static,
`3`=random_sweep) with interval/packet-count/sweep-group-size, payload length,
TX power, all `CONFIG_CSI_LIVE_MCS_*` / `CONFIG_CSI_MINSTREL_*` /
`CONFIG_CSI_REMOTE_MCS_*` / `CONFIG_CSI_DQN_*` knobs, and both MAC arrays.

Receiver (`csi_recv/main/app_main.c`): channel, ESP-NOW rate, gain options,
`CONFIG_MCS_RECOMMENDATION_ENABLED` (legacy reward-model path),
`CONFIG_CSI_DQN_MCS_RECOMMENDATION_ENABLED` (live-policy path),
`CONFIG_CSI_MCS_POLICY_MODEL` (`1` = bandit, `0` = DQN Q-network), the
live-policy cadence/warmup/staleness knobs, and both MAC arrays.

The two receiver recommendation paths are mutually exclusive (firmware
compile error otherwise); the menu enforces this.

## Sender live algorithm values

- `live_mcs_algo = 0` — MINSTREL_LIKE: sender-local ACK EWMA + probing.
- `live_mcs_algo = 1` — CUSTOM_POLICY: legacy reward-model receiver
  recommendations (`mcs_reco` frames).
- `live_mcs_algo = 2` — RECEIVER_POLICY: follows the receiver's live-policy
  frames (`dqn_reco` frames). The receiver-side model behind those frames is
  chosen by `mcs_policy_model` (bandit or DQN) — this is the mode both the
  bandit and DQN presets use.

## Comparing algorithms

Collect paired runs with different presets, then compare the saved
`ack_data.csv` folders:

```bash
python3 tools/compare_mcs_algorithms.py \
  --recursive tools/Scenarios/live_ab \
  --warmup-packets 300 \
  --output-dir tools/results/mcs_algorithm_comparison
```

See `tools/MCS_ALGORITHM_COMPARISON.md` for the full A/B protocol.

## Build/flash commands used

Linux/Pi: `. <export.sh> && idf.py -B <build_dir> flash -b <baud> -p <port>`
Windows: `call <export.bat> && idf.py -B build_win flash -b <baud> -p <port>`

with the per-device project directory (`csi_send` or `csi_recv`) and the
port/baud/build-dir from `device_menu_local.json`.
