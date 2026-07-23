# CSI_RECV

The receiver has two independent MCS recommendation paths:

- The existing reward-model/custom-policy path.
- The live-policy path, disabled by default and using its own model, task, queue, feedback frame, and logs. Its model is selected at compile time by `CONFIG_CSI_MCS_POLICY_MODEL` in `main/app_main.c`:
  - `1` (default): qualified v3.3 two-head contextual bandit from `main/generated_bandit_model.h`, using the receiver-only `link_v5` schema.
  - `0`: qualified v3.3 gamma-zero DQN from `main/generated_dqn_model.h`, using the receiver-only `link_v5` schema.

## Receiver App Partition

The full-CSI canary requires ESP-IDF's 1500K single-app partition. This is
selected in `sdkconfig.defaults`. If a host already has a generated
`csi_recv/sdkconfig` that still selects the 1 MiB table, Device Menu migrates
only the partition selector before a receiver build or flash. The manual
equivalent is:

```bash
cd csi_recv
mv sdkconfig sdkconfig.before-large-partition
. /home/$(id -un)/esp/esp-idf/export.sh
idf.py reconfigure
```

The old configuration remains recoverable in
`sdkconfig.before-large-partition`. The generated replacement should contain
`CONFIG_PARTITION_TABLE_SINGLE_APP_LARGE=y` and
`CONFIG_PARTITION_TABLE_FILENAME="partitions_singleapp_large.csv"`.

## DQN Live Recommendations

In the normal project workflow, configure this from the repository root with:

```bash
python tools/device_menu.py
```

Choose **Select algorithm mode**, then **v3.3 matched gamma-zero DQN (seed 11,
fresh CSI only)**. The preset enables
`CONFIG_CSI_DQN_MCS_RECOMMENDATION_ENABLED`, selects policy model `0`, disables
the custom reward path, and couples the sender to receiver-policy feedback.

Applying the profile updates the source constants used by the normal
`idf.py build` flow.

Optional controls:

- `CONFIG_CSI_DQN_RECOMMENDATION_EVERY_N_PACKETS`: fresh-CSI update cadence, default `1`.
- `CONFIG_CSI_DQN_WARMUP_PACKETS`: CSI gain-control warmup, default `100`.
- `CONFIG_CSI_DQN_CONTROL_INTERVAL_MS`: stale-state control-loop interval, default `5`.
- `CONFIG_CSI_DQN_STALE_MAX_AGE_PACKETS`: cap for stale age/gap counters, default `64`.
- `CONFIG_CSI_DQN_LOG_ENABLED`: emit `DQN_INFERENCE` and `DQN_QUEUE_DROPPED` rows.

Inference runs in a FreeRTOS worker task, not in the Wi-Fi CSI callback. The
deployed checkpoint is marked exact-fresh-only, so the task blocks until a real
new CSI frame arrives and never evaluates it on synthesized stale states.
During a blackout, the sender's recommendation age/sequence checks and
ACK-failure stepdown provide the fallback. The stale control-loop options remain
available for bandit or future trajectory-qualified DQN artifacts. Enabling both
receiver recommendation paths at once is a compile-time error.

### Export A Checkpoint

The DQN slot is pinned to checkpoint SHA-256
`8e865618b96b5efd11f6033b6fe150a3443f227eb0f07378b19a02bf45c3ac43`.
It uses `state_age_packets`, the receiver-only `link_v5` schema with no
source/current-MCS input, two hidden layers, eight actions, and no dropout or
LayerNorm. `link_v5` uses CSI amplitudes plus receiver-side
metadata such as RSSI, SNR, gain, channel, packet length, and stale CSI
age/gap/missing counters. It does not use ACK-history features.

`link_v3` and `link_v4` checkpoints include ACK-history conditioning. Keep
those for offline diagnostics unless sender-side ACK/loss telemetry is added
to the receiver state.

From `examples/get-started` (use your workspace's Python environment):

```bash
.venv/bin/python tools/rl/DQN/dqn_model/export_dqn_to_c_header.py \
  --model tools/rl/DQN/experiments/v3_3_expanded_matched_dqn_gamma0_v1/models/seed_11/dqn_model.pth \
  --output csi_recv/main/generated_dqn_model.h \
  --verify-dataset tools/rl/DQN/datasets/v3_3_expanded_matched_v1/holdout/validation_phase_run2_plus_broad9_disjoint_exact_fresh_link_v7c_ht20_v1_utility.csv \
  --verify-rows 512 \
  --allow-exact-fresh-only-export
```

The exporter validates the architecture and checks NumPy/PyTorch output
parity before replacing `main/generated_dqn_model.h`. The generated header also
requires exactly 114 HT20 CSI scalars with a valid first word, matching the
training data; the receiver rejects other layouts instead of evaluating them.

### Export A Bandit Checkpoint

The default bandit policy (`CONFIG_CSI_MCS_POLICY_MODEL == 1`) is the
validation-selected seed-11 candidate from
`v3_3_expanded_matched_bandit_v1`. It has checkpoint SHA-256
`4f6d4021a47f85f19742673e675ab4952607f76c45875abbfaae0e5375ce2886`
and uses the same 132-feature, receiver-only `link_v5` input contract as the
matched DQN: no source/current-MCS feature, exact-fresh 114-scalar HT20 CSI,
and a valid first word.

Training used the same exact TRAIN (800,000 rows) and disjoint VALIDATION
(187,139 rows) artifacts as the v3.3 DQN and action-reward experiments. The
bandit's family-appropriate objective is different: binary delivery plus robust
log-service-time heads are combined into a bounded-utility plug-in score using
a train-global normalization, rather than the evaluator's stored, per-source
normalized reward. This score is not a claim that the two heads are calibrated
or that it is the exact expected utility. The locked qualification was run only
after seed 11 was selected on validation; here "qualified" means the candidate
and provenance were sealed for deployment, not that a statistical performance
threshold was passed.

Export is deliberately gated on both the sealed selection and completed
qualification. From the repository root:

```bash
.venv/bin/python tools/rl/DQN/action_reward_model/export_bandit_model_to_c_header.py \
  --model tools/rl/DQN/experiments/v3_3_expanded_matched_bandit_v1/models/seed_11/bandit_model.pth \
  --output csi_recv/main/generated_bandit_model.h \
  --verify-dataset tools/rl/DQN/datasets/v3_3_expanded_matched_v1/holdout/validation_phase_run2_plus_broad9_disjoint_exact_fresh_link_v7c_ht20_v1_utility.csv \
  --verify-rows 512 \
  --candidate-selection tools/rl/DQN/experiments/v3_3_expanded_matched_bandit_v1/candidate_selection.json \
  --qualification-complete tools/rl/DQN/experiments/v3_3_expanded_matched_bandit_v1/qualification_v1/qualification_complete.json \
  --allow-exact-fresh-only-export
```

This writes the header and `main/generated_bandit_model.h.deployment.json`
using per-file atomic replacement. Device Menu pins the complete header and
deployment-record hashes and blocks receiver build/flash if either artifact is
missing or changes, including an interrupted partial update. The record is
deterministic and repository-relative, so the
same sealed inputs reproduce it from another checkout root. It also binds the
sealed selected bandit-versus-DQN comparison, not merely the bandit's own
qualification. The generated header was verified on 512 validation rows
against the checkpoint before installation. The firmware also enforces the
state contract and reports the checkpoint schema, objective, hash, state size,
and fresh-only flag in its `CSI_MODEL_VARIANT` boot line.

Because the bandit runs one network pass for each of eight candidate actions,
confirm its real ESP32-C5 latency before the formal experiment. Use the existing
`DQN_INFERENCE` and `DQN_QUEUE_DROPPED` health rows to check that inference stays
inside the sender's 50 ms recommendation lifetime without sustained queue
drops. This hardware timing check cannot be established by an offline model or
firmware build alone.

## Reward-Model Custom Recommendation Path

The custom path is shared by the v2.6 amplitude control, the v3.2 broad-data
amplitude canary, the staged v3.1 full-CSI canary, and the separate v3.3
amplitude/FullCSI seed-11 candidates. Its principal controls
in `main/app_main.c` are:

- `CONFIG_MCS_RECOMMENDATION_ENABLED`
- `CONFIG_MCS_RECOMMENDATION_EVERY_N_PACKETS`
- `CONFIG_MCS_RECOMMENDATION_USE_MODEL`
- `CONFIG_CSI_REWARD_MODEL_VARIANT`
- `CONFIG_CSI_REWARD_MODEL_SNR_GUARD_ENABLED`

The variant is `0` for `main/generated_reward_model_v2.h`, `1` for
`main/generated_reward_model_linkv7c_canary.h`, and `2` for
`main/generated_reward_model_linkv5_broad_canary.h`. Variant `3` selects
`main/generated_reward_model_v3_3_amp_seed11.h`, and variant `4` selects
`main/generated_reward_model_v3_3_full_seed11.h`. It uses the custom-policy
feedback frame and is separate from the DQN/live-policy controller. Prefer the
coupled presets in `tools/device_menu.py` so sender and receiver flags cannot
drift apart.

For the current receiver-only deployment, prefer the `link_v5` action-reward
model exported into `main/generated_reward_model_v2.h`. The firmware state
builder supports gain-compensated `link_v5` reward headers with 132 features
and no source/current-MCS conditioning:

```bash
REWARD_RUN="v2_6_linkv5_receiveronly_reward_model_holdout_f202_rooftop02"
REWARD="tools/rl/DQN/models/$REWARD_RUN/action_reward_model.pth"
REWARD_HEADER="tools/rl/DQN/models/$REWARD_RUN/generated_reward_model_linkv5.h"

python3 tools/rl/DQN/action_reward_model/export_reward_model_to_c_header.py \
  --model "$REWARD" \
  --output "$REWARD_HEADER"
```

Only after reviewing the generated header, install it with:

```bash
cp "$REWARD_HEADER" csi_recv/main/generated_reward_model_v2.h
```

The selected broad-data model is seed 42 from the completed v3.2 ablation. It
uses the same receiver-only `link_v5` schema as v2.6 but was trained on the
broad LOS/NLOS/RapidSweep training set. Reproduce its checked deployment header
without replacing v2.6:

```bash
tools/venv/bin/python3 \
  tools/rl/DQN/action_reward_model/export_reward_model_to_c_header.py \
  --model tools/rl/DQN/models/v3_2_broad_exact_reward_ablation_v1/seed_42/amp/action_reward_model.pth \
  --output csi_recv/main/generated_reward_model_linkv5_broad_canary.h
```

The menu requires checkpoint SHA-256
`b70a3dd63bbe6964b5128412d6325becc2ed00d15bbdfafeab97b949c4ba7964`
for this mode. Select **v3.2 broad-data amplitude reward-model canary (seed
42)** to stage it, **v3.1 robust full-CSI reward-model canary** for the phase
model, **v3.3 expanded matched amplitude reward model (seed 11)** or **v3.3
expanded matched FullCSI reward model (seed 11)** for the new paired candidates,
or **v2.6 amplitude reward-model champion (rollback/control)** to return to the
original model. These choices only change the selector; none overwrites another
artifact. Both v3.3 presets leave the SNR guard disabled to match their offline
paired evaluation. The receiver prints `CSI_MODEL_VARIANT` at startup to record
the selected model in live captures.

Set `CONFIG_MCS_RECOMMENDATION_USE_MODEL=0` to use the fixed RSSI heuristic
instead of a learned reward model. The coupled `rssi_heuristic` Device Menu
preset also configures the sender to follow `mcs_reco` feedback and preserves
the sequence-age and ACK-failure safeguards. The thresholds are `-45, -50,
-55, -60, -66, -72, -78 dBm` for MCS7 through MCS1 respectively, with lower
RSSI selecting MCS0. Its startup identity is
`CSI_MODEL_VARIANT,rssi_heuristic,rssi_thresholds_v1,no_checkpoint,0`.

The worker emits a `REWARD_INFERENCE_HEADER` followed by bounded
`REWARD_INFERENCE` health rows at most once per second during normal operation
and immediately on an error/overrun. Rows include inference latency, queue
depth, total drops, invalid inputs, queue-full events, state-build failures, coalesced jobs,
non-finite model scores, and inference overruns. Any non-finite score fails
closed to a forced MCS0 recommendation with confidence zero and an explicit
`model_nonfinite_*` status.
