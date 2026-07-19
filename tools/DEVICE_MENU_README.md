# Sender/Receiver Device Menu

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

| # | Mode | Menu label | Sender | Receiver |
|---|------|------------|--------|----------|
| 1 | `minstrel` | Minstrel-like (sender-local ACK statistics, no receiver feedback) | `MINSTREL_LIKE`, no remote recommendations | pure CSI logger |
| 2 | `bandit` | Contextual bandit (receiver live policy, contract checked in firmware) | `RECEIVER_POLICY` (`dqn_reco` frames) with blackout safeguards | live-policy path, contract-checked `link_v3c` bandit |
| 3 | `dqn` | DQN Q-network (receiver live policy) | `RECEIVER_POLICY` (`dqn_reco` frames) with blackout safeguards | live-policy path, DQN Q-network |
| 4 | `reward_model` | v2.6 amplitude reward-model champion (rollback/control) | `CUSTOM_POLICY` (`mcs_reco` frames) | custom reward-model path, v2.6 amplitude header |
| 5 | `reward_model_v7c_canary` | v3.1 robust full-CSI reward-model canary | `CUSTOM_POLICY` (`mcs_reco` frames) with age/sequence/failure safeguards | custom reward-model path, v7c staging header and low-SNR guard |
| 6 | `static` | Static MCS baseline (fixed rate, no adaptation) | fixed MCS (asks which), no adaptation | pure CSI logger |
| 7 | `random_sweep` | RANDOM_SWEEP data collection (uniform per-packet MCS, propensity 1/8) | randomized collection (asks group size) | pure CSI logger |

Before applying, the menu shows exactly which values will change. For model
modes it verifies the required generated header is present and has the expected
model marker, and prints the export command if not. The firmware then enforces
its compiled feature/state contract for contract-bound models; the bandit menu
label explicitly identifies this check.

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
`CONFIG_MCS_RECOMMENDATION_ENABLED` (custom reward-model path used by both the
v2.6 control and v3.1 canary), the v7c header selector and low-SNR guard,
`CONFIG_CSI_DQN_MCS_RECOMMENDATION_ENABLED` (live-policy path),
`CONFIG_CSI_MCS_POLICY_MODEL` (`1` = bandit, `0` = DQN Q-network), the
live-policy cadence/warmup/staleness knobs, and both MAC arrays.

The two receiver recommendation paths are mutually exclusive (firmware
compile error otherwise); the menu enforces this.

## Sender live algorithm values

- `live_mcs_algo = 0` — MINSTREL_LIKE: sender-local ACK EWMA + probing.
- `live_mcs_algo = 1` — CUSTOM_POLICY: reward-model receiver recommendations
  (`mcs_reco` frames). Both the v2.6 rollback/control and v3.1 full-CSI canary
  use this path.
- `live_mcs_algo = 2` — RECEIVER_POLICY: follows the receiver's live-policy
  frames (`dqn_reco` frames). The receiver-side model behind those frames is
  chosen by `mcs_policy_model` (bandit or DQN) — this is the mode both the
  bandit and DQN presets use.

The v3.1 canary does **not** use the DQN recommendation path. Its preset enables
the receiver custom reward-model path, selects the v7c staging header, and caps
recommendations at MCS1 when SNR is below 15 dB. On the sender it enables the
CUSTOM_POLICY recommendation-age check, the shared sequence-gap guard, and an
emergency stepdown after eight consecutive failures. Selecting the v2.6 preset
restores the amplitude model as the rollback/control without overwriting either
generated header. Both reward-model presets pin stop-and-wait ACK timing and the
MCS0..MCS7 range so a stale custom profile cannot bypass the low-SNR or
non-finite-score safety stepdown.

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
