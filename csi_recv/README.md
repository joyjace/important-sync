# CSI_RECV

The receiver has two independent MCS recommendation paths:

- The existing reward-model/custom-policy path.
- The live-policy path, disabled by default and using its own model, task, queue, feedback frame, and logs. Its model is selected at compile time by `CONFIG_CSI_MCS_POLICY_MODEL` in `main/app_main.c`:
  - `1` (default): two-head contextual bandit from `main/generated_bandit_model.h`, using the compact `link_v3c` (57-amplitude) state schema.
  - `0`: DQN Q-network from `main/generated_dqn_model.h`, using the `link_v2`..`link_v6` schema family.

## Receiver App Partition

The full-CSI canary requires ESP-IDF's 1500K single-app partition. This is
selected in `sdkconfig.defaults`. If a host already has a generated
`csi_recv/sdkconfig` that still selects the 1 MiB table, migrate it once before
building:

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

Then open receiver parameters and enable:

- `CONFIG_CSI_DQN_MCS_RECOMMENDATION_ENABLED`

Applying the profile updates the source constants used by the normal
`idf.py build` flow.

Optional controls:

- `CONFIG_CSI_DQN_RECOMMENDATION_EVERY_N_PACKETS`: fresh-CSI update cadence, default `1`.
- `CONFIG_CSI_DQN_WARMUP_PACKETS`: CSI gain-control warmup, default `100`.
- `CONFIG_CSI_DQN_CONTROL_INTERVAL_MS`: stale-state control-loop interval, default `5`.
- `CONFIG_CSI_DQN_STALE_MAX_AGE_PACKETS`: cap for stale age/gap counters, default `64`.
- `CONFIG_CSI_DQN_LOG_ENABLED`: emit `DQN_INFERENCE` and `DQN_QUEUE_DROPPED` rows.

Inference runs in a FreeRTOS worker task, not in the Wi-Fi CSI callback. The
task keeps the latest CSI state and continues to run every
`CONFIG_CSI_DQN_CONTROL_INTERVAL_MS` while no new CSI arrives. During those
quiet periods it increments CSI age, packet gap, and missing-packet estimate,
then sends a fresh DQN feedback frame to the sender. Enabling both receiver
recommendation paths at once is a compile-time error.

### Export A Checkpoint

For receiver-only deployment, use a checkpoint trained with `state_age_packets`,
the `link_v5` state schema, source-MCS input, two hidden layers, eight actions,
and no dropout or LayerNorm. `link_v5` uses CSI amplitudes plus receiver-side
metadata such as RSSI, SNR, gain, channel, packet length, and stale CSI
age/gap/missing counters. It does not use ACK-history features.

`link_v3` and `link_v4` checkpoints include ACK-history conditioning. Keep
those for offline diagnostics unless sender-side ACK/loss telemetry is added
to the receiver state.

From `examples/get-started` (use your workspace's Python environment):

```bash
python3 tools/rl/DQN/dqn_model/export_dqn_to_c_header.py \
  --model tools/rl/DQN/models/v2_6_linkv5_receiveronly_lossfallback_keeptelemetry_p030_double_cql001_gamma06_lr0001_qreg001_quick01/dqn_mcs_model.pth \
  --output csi_recv/main/generated_dqn_model.h \
  --verify-dataset tools/rl/DQN/datasets/v2_4_linkv3_holdout_f202_rooftop02/rl_dqn_dataset_LOS_1m_F202_run02_utility.csv \
  --verify-rows 512
```

The exporter validates the architecture and checks NumPy/PyTorch output
parity before replacing `main/generated_dqn_model.h`.

### Export A Bandit Checkpoint

For the default bandit policy (`CONFIG_CSI_MCS_POLICY_MODEL == 1`), export a
two-head bandit checkpoint trained with `--state-schema link_v3c`:

```bash
python3 tools/rl/DQN/action_reward_model/export_bandit_model_to_c_header.py \
  --model tools/rl/DQN/models/<bandit_run>/bandit_model.pth \
  --output csi_recv/main/generated_bandit_model.h
```

Checkpoints trained before the schema rename stored the compact layout as
`link_v3`; the loader maps them to `link_v3c` automatically.

## Reward-Model Custom Recommendation Path

The custom path is used by both the v2.6 amplitude control and the staged v3.1
full-CSI canary. Its principal controls in `main/app_main.c` are:

- `CONFIG_MCS_RECOMMENDATION_ENABLED`
- `CONFIG_MCS_RECOMMENDATION_EVERY_N_PACKETS`
- `CONFIG_MCS_RECOMMENDATION_USE_MODEL`
- `CONFIG_CSI_REWARD_MODEL_USE_V7C_CANARY`
- `CONFIG_CSI_REWARD_MODEL_SNR_GUARD_ENABLED`

With the canary selector off, the path includes
`main/generated_reward_model_v2.h`; with it on, it includes
`main/generated_reward_model_linkv7c_canary.h`. It uses the custom-policy
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

Do not overwrite the v2.6 header to try the full-CSI checkpoint. Export v7c to
the separate canary header and select the **v3.1 robust full-CSI reward-model
canary** menu mode; the **v2.6 amplitude reward-model champion
(rollback/control)** mode switches back without replacing either artifact.

The worker emits a `REWARD_INFERENCE_HEADER` followed by bounded
`REWARD_INFERENCE` health rows at most once per second during normal operation
and immediately on an error/overrun. Rows include inference latency, queue
depth, total drops, invalid inputs, queue-full events, state-build failures, coalesced jobs,
non-finite model scores, and inference overruns. Any non-finite score fails
closed to a forced MCS0 recommendation with confidence zero and an explicit
`model_nonfinite_*` status.
