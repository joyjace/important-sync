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
| 2 | `bandit` | v3.3 matched two-head bandit (seed 11, fresh CSI only) | `RECEIVER_POLICY` (`dqn_reco` frames) with blackout safeguards | exact-hash-checked `link_v5` bandit plus deployment seal |
| 3 | `dqn` | v3.3 matched gamma-zero DQN (seed 11, fresh CSI only) | `RECEIVER_POLICY` (`dqn_reco` frames) with blackout safeguards | exact-hash-checked `link_v5` DQN, seed 11 |
| 4 | `reward_model` | v2.6 amplitude reward-model champion (rollback/control) | `CUSTOM_POLICY` (`mcs_reco` frames) | custom reward-model path, v2.6 amplitude header |
| 5 | `reward_model_broad_amp_canary` | v3.2 broad-data amplitude reward-model canary (seed 42) | `CUSTOM_POLICY` (`mcs_reco` frames) with age/sequence/failure safeguards | custom reward-model path, exact-hash-checked broad-data `link_v5` header |
| 6 | `reward_model_v7c_canary` | v3.1 robust full-CSI reward-model canary | `CUSTOM_POLICY` (`mcs_reco` frames) with age/sequence/failure safeguards | custom reward-model path, v7c staging header and low-SNR guard |
| 7 | `reward_model_v3_3_amp_seed11` | v3.3 expanded matched amplitude reward model (seed 11) | `CUSTOM_POLICY` (`mcs_reco` frames) with age/sequence/failure safeguards | exact-hash-checked `link_v5` v3.3 header, no SNR guard |
| 8 | `reward_model_v3_3_full_seed11` | v3.3 expanded matched FullCSI reward model (seed 11) | `CUSTOM_POLICY` (`mcs_reco` frames) with age/sequence/failure safeguards | exact-hash-checked `link_v7c` v3.3 header, no SNR guard |
| 9 | `static` | Static MCS baseline (fixed rate, no adaptation) | fixed MCS (asks which), no adaptation | pure CSI logger |
| 10 | `random_sweep` | RANDOM_SWEEP data collection (uniform per-packet MCS, propensity 1/8) | randomized collection (asks group size) | pure CSI logger |
| 11 | `rssi_heuristic` | RSSI threshold heuristic (receiver recommendations, no learned model) | `CUSTOM_POLICY` (`mcs_reco` frames) with age/sequence/failure safeguards | fixed RSSI thresholds; no learned checkpoint |

The RSSI preset is appended as option 11 so the original option numbers remain
unchanged.

Before applying, the menu shows exactly which values will change. For model
modes it verifies each required generated artifact and prints the export command
if one is missing or mismatched. The DQN and selected reward models pin their
generated-header hash and checkpoint identities, respectively. The bandit pins
its checkpoint, complete header, and qualification-gated deployment-record
hashes. The firmware then enforces the compiled feature/state contract.

After applying, it offers to build+flash **the board attached to this host**
(sender / receiver / both). On the two-Pi setup: apply the mode once, sync the
sources to both Pis (git or `sync_selected_files.sh` → `important-sync`), then
on each Pi build+flash only its own device.

Both application consoles run at 921600 baud. Clean sender builds obtain that
setting from `csi_send/sdkconfig.defaults`; before a sender build or flash, the
menu also migrates only the console/monitor entries in an existing generated
`csi_send/sdkconfig`. This is necessary because ESP-IDF otherwise keeps an old
115200 setting instead of applying changed defaults. The host-local `baud`
field remains the separate esptool flashing baud.

Before receiver builds or flashes, the menu also migrates an existing generated
`csi_recv/sdkconfig` from the 1 MiB single-app partition to the repository's
1.5 MiB single-app default. This preserves unrelated settings while providing
safe firmware-size headroom for the generated model headers. Intentional OTA,
custom, TEE, or encrypted-NVS partition layouts are left untouched.

Every receiver build or flash started from the menu rechecks the artifact for
the active receiver policy. A stale or mismatched DQN header is therefore
blocked even when using the direct build/flash menu options rather than
reselecting the mode first.

## Shared vs host-local configuration

- `tools/device_menu_profile.json` — the device configuration (channel, rates,
  algorithm flags, MACs). **Synced** between hosts so the pair stays
  consistent.
- `tools/device_menu_local.json` — this machine's serial ports, ESP-IDF export
  script, build directory, and optional post-flash command. **Never synced**
  (gitignored and excluded from `sync_selected_files.sh`). Created on first run
  with per-OS defaults: Linux/Pi → `/dev/ttyUSB0`, `export.sh`, `build`;
  Windows → `COM3`/`COM4`, `export.bat`, `build_win`. Edit via menu option 6.

### Auto-running a capture script right after flashing

`post_flash_command` (empty/disabled by default) is a host-local shell command
run immediately after this host's own flash succeeds (exit code 0) -- before
control returns to the menu. On a capture Pi, point it at the script that
starts reading the serial port and ships the data off, e.g.:

```
post_flash_command: /home/gje1671/Desktop/run_and_send.sh
```

so there is no manual gap between "board finished rebooting after flash" and
"capture started," which is when early packets are otherwise easiest to miss.
Set it via local-settings option 6 (enter `off` to disable it again). It only
fires on a successful flash and only on the host it's configured on -- each Pi
in the two-Pi setup has its own `device_menu_local.json`, so the sender Pi and
receiver Pi can each point at their own post-flash script (or leave it unset).

## What the menu writes to the C sources

Sender (`csi_send/main/app_main.c`): channel, ESP-NOW rate, send frequency,
pacing, ACK timing mode, rate-switch mode (`0`=time, `1`=packet, `2`=static,
`3`=random_sweep) with interval/packet-count/sweep-group-size, payload length,
TX power, all `CONFIG_CSI_LIVE_MCS_*` / `CONFIG_CSI_MINSTREL_*` /
`CONFIG_CSI_REMOTE_MCS_*` / `CONFIG_CSI_DQN_*` knobs, and both MAC arrays.

Receiver (`csi_recv/main/app_main.c`): channel, ESP-NOW rate, gain options,
`CONFIG_MCS_RECOMMENDATION_ENABLED` (custom recommendation path),
`CONFIG_MCS_RECOMMENDATION_USE_MODEL` (`0` = RSSI heuristic, `1` = learned
reward model), the five-way
`CONFIG_CSI_REWARD_MODEL_VARIANT` selector and low-SNR guard,
`CONFIG_CSI_DQN_MCS_RECOMMENDATION_ENABLED` (live-policy path),
`CONFIG_CSI_MCS_POLICY_MODEL` (`1` = bandit, `0` = DQN Q-network), the
live-policy cadence/warmup/staleness knobs, and both MAC arrays.

The two receiver recommendation paths are mutually exclusive (firmware
compile error otherwise); the menu enforces this.

## Sender live algorithm values

- `live_mcs_algo = 0` — MINSTREL_LIKE: sender-local ACK EWMA + probing.
- `live_mcs_algo = 1` — CUSTOM_POLICY: reward-model receiver recommendations
  or the RSSI heuristic (`mcs_reco` frames). The v2.6 rollback, v3.2 broad-data
  amplitude canary, v3.1 full-CSI canary, both v3.3 candidates, and RSSI preset
  all use this transport path.
- `live_mcs_algo = 2` — RECEIVER_POLICY: follows the receiver's live-policy
  frames (`dqn_reco` frames). The receiver-side model behind those frames is
  chosen by `mcs_policy_model` (bandit or DQN) — this is the mode both the
  bandit and DQN presets use.

The reward-model variant values are `0` = v2.6 rollback, `1` = v3.1 full CSI,
`2` = v3.2 broad-data amplitude, `3` = v3.3 amplitude seed 11, and `4` = v3.3
FullCSI seed 11. Each has its own generated header; changing the mode never
overwrites another model. The v3.2 and v3.3 modes check their exact checkpoint
SHA-256 before they can be applied. The v3.3 pair deliberately uses the same
training seed and leaves the SNR guard off to match the offline paired
evaluation.

The v3.1 preset selects the v7c staging header and caps recommendations at MCS1
when SNR is below 15 dB. All reward-model presets pin stop-and-wait ACK timing,
MCS0..MCS7, recommendation-age/sequence checks, and emergency stepdown after
eight consecutive failures. At startup, the receiver emits
`CSI_MODEL_VARIANT` so a capture records which artifact was flashed.

### Matched bandit preset

The `bandit` preset now deploys the validation-selected seed-11 candidate from
`v3_3_expanded_matched_bandit_v1`. It uses the receiver-only, 132-feature
`link_v5` state (no current/source MCS), requires exact-fresh 114-scalar HT20
CSI with a valid first word, and was trained with seeds 11, 42, and 73. Seed 11
had the lowest validation macro run loss: `0.480523`, versus `0.483012` and
`0.484476`. The locked qualification data was opened only after this selection
was sealed and did not change it.

This retraining uses the exact same observed rows and outcomes as the matched
DQN and action-reward experiments:

- TRAIN: 800,000 rows, SHA-256
  `445e9fc1378b02d7435f3be50793b18db2f7ec6481cddefdd12e81309b3de8c3`.
- VALIDATION: 187,139 disjoint rows, SHA-256
  `e9b424b75d31082e14ee817bda110d8e080142ea01138d8c6579bd1201b38859`.

The evidence is matched, but the learning targets are intentionally appropriate
to each model family rather than literally identical. The bandit predicts a
delivery score and a robust conditional location for log service time, then
ranks actions with a two-head bounded-utility plug-in score using a train-global
q95 scale (`8.432273`). The DQN/action-reward models instead learn the stored
reward/value target, whose normalization was calculated per source. Therefore
the deployed bandit does **not** optimize exactly the same reward reported by
the evaluator, and its head outputs should not be read as proven calibration or
an exact mathematical expectation.

Across the nine pooled locked holdouts, the preselected bandit averaged a
composite score of `0.9306`, mean service time `1.5406 ms`, and mean reward
`0.7428`. The preselected DQN averaged `0.9500`, `1.3830 ms`, and `0.7706`.
Thus the bandit is a provenance- and deployment-qualified comparison candidate;
"qualified" is not a statistical performance threshold, and these offline
results do not claim that it beat the DQN. They are one-step replay diagnostics
with mean logged-action coverage of about 16.6% for the bandit and 17.4% for
DQN. Seven of the nine pooled holdouts use marginal empirical action frequency
where state-conditional logging propensity is unknown, so their fields named
`snips_*` are descriptive marginal-reweighted estimates rather than guaranteed
unbiased off-policy estimates. The matched training corpus is also only partly
randomized. The nine related holdouts have no confidence interval or
significance claim, and the analysis does not simulate policy-induced state
changes, CSI blackouts, stale recommendations, or sender fallback. The live
counterbalanced experiment remains the end-to-end comparison.

The reproducible workflow is:

```bash
.venv/bin/python tools/rl/DQN/action_reward_model/run_v3_3_matched_bandit.py --device cpu
.venv/bin/python tools/rl/DQN/action_reward_model/qualify_v3_3_matched_bandit.py --device cpu
.venv/bin/python tools/rl/DQN/action_reward_model/export_bandit_model_to_c_header.py \
  --model tools/rl/DQN/experiments/v3_3_expanded_matched_bandit_v1/models/seed_11/bandit_model.pth \
  --output csi_recv/main/generated_bandit_model.h \
  --verify-dataset tools/rl/DQN/datasets/v3_3_expanded_matched_v1/holdout/validation_phase_run2_plus_broad9_disjoint_exact_fresh_link_v7c_ht20_v1_utility.csv \
  --verify-rows 512 \
  --candidate-selection tools/rl/DQN/experiments/v3_3_expanded_matched_bandit_v1/candidate_selection.json \
  --qualification-complete tools/rl/DQN/experiments/v3_3_expanded_matched_bandit_v1/qualification_v1/qualification_complete.json \
  --allow-exact-fresh-only-export
```

Device Menu pins checkpoint SHA-256
`4f6d4021a47f85f19742673e675ab4952607f76c45875abbfaae0e5375ce2886`,
header SHA-256
`6bc793b157dcd7274cbf789818f7c538acf5579ad10233a07a64a8d65613332b`,
and deployment-record SHA-256
`02633beec49e180d7f95b1c10155247435bea4ec43e88b7474c39e86602926b3`.
The deterministic, repository-relative `bandit_firmware_deployment/v1` record
binds the validation selection, completed 27-prediction/42-evaluation
qualification, and sealed preselected bandit-versus-DQN comparison before the
exporter replaces either deployment artifact.

The two-head bandit scores the network eight times per decision, so a clean
build does not by itself prove that the ESP32-C5 meets the every-packet/50 ms
control deadline. Before collecting the comparison runs, do a short on-device
smoke run and inspect the existing `DQN_INFERENCE` latency/queue counters and
`DQN_QUEUE_DROPPED` output. There should be no sustained queue growth or stale
recommendation rejection. These are controller health logs, not route-event
markers.

### RSSI heuristic preset

The `rssi_heuristic` preset enables the existing non-learned receiver policy.
It maps RSSI to MCS as follows:

| RSSI | Selected MCS |
|---|---:|
| `>= -45 dBm` | 7 |
| `-50 .. -46 dBm` | 6 |
| `-55 .. -51 dBm` | 5 |
| `-60 .. -56 dBm` | 4 |
| `-66 .. -61 dBm` | 3 |
| `-72 .. -67 dBm` | 2 |
| `-78 .. -73 dBm` | 1 |
| `< -78 dBm` | 0 |

It evaluates every 20 received data packets and sends feedback when the MCS
changes or once the sender sequence has advanced by at least 100 since the last
feedback. At startup the receiver emits
`CSI_MODEL_VARIANT,rssi_heuristic,rssi_thresholds_v1,no_checkpoint,0` and the
seven thresholds in `CSI_RSSI_THRESHOLDS_DBM`. Attach the capture process before
reset/boot if that startup line must be present in `csi_data_log.txt`; otherwise
copy the Device Menu profile into the run metadata. The legacy
`REWARD_INFERENCE` telemetry name is shared by the reward-model and RSSI worker
for capture-parser compatibility; it does not imply that RSSI uses a model.

For fair live comparisons, record the complete controller settings. The
reward-model/RSSI custom path evaluates every 20 packets and accepts feedback
for up to 300 ms; the DQN/bandit live-policy path evaluates every fresh packet
after warmup and uses a 50 ms recommendation lifetime plus its loss-stepdown
safeguard. Keep radio, traffic, route, and counterbalanced run-order settings
identical, but describe live results as **controller-stack** comparisons. They
do not isolate neural architecture because cadence, feedback transport, and
fallback behavior differ. Treat the matched offline qualification as a
supporting one-step diagnostic, not as a causal replacement for the live
controller comparison.

The v3.3 seed-11 headers can be regenerated without replacing any existing
deployment artifact:

```bash
tools/venv/bin/python3 tools/rl/DQN/action_reward_model/export_reward_model_to_c_header.py \
  --model tools/rl/DQN/experiments/v3_3_expanded_matched_batchtrace_v1/models/seed_11/amp/action_reward_model.pth \
  --output csi_recv/main/generated_reward_model_v3_3_amp_seed11.h

tools/venv/bin/python3 tools/rl/DQN/action_reward_model/export_reward_model_to_c_header.py \
  --model tools/rl/DQN/experiments/v3_3_expanded_matched_batchtrace_v1/models/seed_11/full/action_reward_model.pth \
  --output csi_recv/main/generated_reward_model_v3_3_full_seed11.h
```

The `dqn` preset is pinned to checkpoint SHA-256
`8e865618b96b5efd11f6033b6fe150a3443f227eb0f07378b19a02bf45c3ac43`.
The menu also verifies the complete generated header SHA-256
`74ae4bbdd3beb8def2627a5e88137548dca58d7a0fd236a58d73eaa8bf68040a`,
so changing a weight array while retaining the checkpoint marker is rejected.
It replaces the previous ambiguous DQN slot with the qualified 132-feature
`link_v5` receiver-only network. Because its training rows contain exact-fresh
CSI only, the receiver runs it only when a new CSI frame arrives. It also
requires exactly 114 HT20 CSI scalars and a valid first word, matching the
training input contract. During a blackout, the sender rejects
aged/sequence-stale recommendations and performs its existing ACK-failure
stepdown. The receiver boot log reports the checkpoint hash, checkpoint schema,
transition contract, state dimension, and fresh-only flag in
`CSI_MODEL_VARIANT,dqn,...`.

Regenerate the deployed DQN header with:

```bash
.venv/bin/python tools/rl/DQN/dqn_model/export_dqn_to_c_header.py \
  --model tools/rl/DQN/experiments/v3_3_expanded_matched_dqn_gamma0_v1/models/seed_11/dqn_model.pth \
  --output csi_recv/main/generated_dqn_model.h \
  --verify-dataset tools/rl/DQN/datasets/v3_3_expanded_matched_v1/holdout/validation_phase_run2_plus_broad9_disjoint_exact_fresh_link_v7c_ht20_v1_utility.csv \
  --verify-rows 512 \
  --allow-exact-fresh-only-export
```

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
