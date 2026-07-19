# Live A/B Test: Bandit Policy vs Minstrel vs Static

Compares the deployed two-head bandit policy (receiver-driven, CSI-based)
against the sender-local Minstrel-like controller and a static-MCS baseline,
on live hardware, in the same positions.

## Arms

All arms use the same two boards, payload (128 B), rate (200 pkt/s), and
channel. Only the listed defines change between arms. The receiver always
keeps `CONFIG_MCS_RECOMMENDATION_ENABLED 0` (the legacy reco path caused the
collision loss floor — never re-enable it).

### Arm A — Bandit policy (receiver-driven)

`csi_recv/main/app_main.c`:

```c
#define CONFIG_CSI_DQN_MCS_RECOMMENDATION_ENABLED 1   // enables the live reco task
#define CONFIG_CSI_MCS_POLICY_MODEL 1                 // 1 = generated_bandit_model.h (default)
```

`csi_send/main/app_main.c`:

```c
#define CONFIG_CSI_LIVE_MCS_SELECTION_ENABLED 1
#define CONFIG_CSI_LIVE_MCS_ALGO 2                    // receiver-driven policy
#define CONFIG_CSI_DQN_REMOTE_RECOMMENDATION_ENABLED 1
```

The exported model must be current: rerun
`action_reward_model/export_bandit_model_to_c_header.py` after any retrain.

### Arm B — Minstrel-like (sender-local)

`csi_recv`: `CONFIG_CSI_DQN_MCS_RECOMMENDATION_ENABLED 0` (passive CSI logger).

`csi_send`:

```c
#define CONFIG_CSI_LIVE_MCS_SELECTION_ENABLED 1
#define CONFIG_CSI_LIVE_MCS_ALGO 0                    // Minstrel-like
#define CONFIG_CSI_DQN_REMOTE_RECOMMENDATION_ENABLED 0
```

### Arm C — Static MCS3 (strongest fixed baseline from offline eval)

`csi_recv`: same as Arm B (passive).

`csi_send`:

```c
#define CONFIG_CSI_LIVE_MCS_SELECTION_ENABLED 0
#define CONFIG_RATE_SWITCH_MODE 2                     // STATIC
#define CONFIG_ESP_NOW_RATE WIFI_PHY_RATE_MCS3_LGI
```

(Optionally add a static MCS7 arm the same way, to show the high-rate
failure mode in NLOS.)

## Run schedule

Position drift is the enemy — interleave arms per scenario rather than
running each arm in a block:

- Scenarios: at least one LOS position (3–5 m) and one NLOS cliff position
  (the 3m/5m NLOS spots from the RapidSweep collection).
- Per scenario, run **A B C C B A** (2–3 min each). The mirrored order
  cancels slow environmental drift; boards stay untouched between runs
  within a scenario.
- Between runs, only reflash the sender (arms B↔C differ on sender only;
  A also needs the reco-enabled receiver — flash the receiver once at the
  start of the A runs and once when leaving them).

Capture each run exactly like a collection run:

```bash
python3 tools/csi_data_read_parse_SSH.py \
  -p <recv_port> -b 921600 -sp <send_port> -sb 115200 \
  -s runs/<scenario>_<arm>_<i>/csi_data.csv \
  -l runs/<scenario>_<arm>_<i>/csi_log.txt \
  -a runs/<scenario>_<arm>_<i>/ack_data.csv \
  -ap runs/<scenario>_<arm>_<i>/ack_pdr.csv \
  --send-log runs/<scenario>_<arm>_<i>/send_log.txt
```

## Analysis

```bash
python3 tools/rl/DQN/shared/compare_ab_runs.py \
  bandit_1=runs/nlos5m_A_1/ack_data.csv \
  bandit_2=runs/nlos5m_A_2/ack_data.csv \
  minstrel_1=runs/nlos5m_B_1/ack_data.csv \
  minstrel_2=runs/nlos5m_B_2/ack_data.csv \
  static3_1=runs/nlos5m_C_1/ack_data.csv \
  static3_2=runs/nlos5m_C_2/ack_data.csv \
  --utility-scale 8.35
```

`--utility-scale 8.35` is the training scale of `bandit_rapidsweep_v2_censored`
(printed at training time), so `mean_reward` is directly comparable to the
offline SNIPS numbers.

## Sanity checks per run (before trusting results)

1. **Collision check** — `compare_ab_runs.py` prints
   `loss_burst_period_delivered`; a value near 20 with burstiness >> 1 in the
   *bandit arm* means the recommendation traffic is colliding with data.
   The reco task sends packet-synchronized (right after a data packet, in
   the sender's pacing gap), which should avoid this; if it appears, raise
   `CONFIG_CSI_DQN_RECOMMENDATION_EVERY_N_PACKETS` (2–4) and retest.
2. **Inference latency** — the receiver's `DQN_INFERENCE` console lines log
   per-decision latency in µs. It must stay well under the 5 ms packet
   interval (the 8-action bandit inference is ~1–2 ms on the C5).
3. **Recommendation uptake** — the sender's `MCS_CHANGE` lines with reason
   `dqn_recommendation` confirm the sender applies the recommendations
   (`dqn_hold` means gates rejected them — check
   `CONFIG_CSI_DQN_REMOTE_MAX_AGE_MS` if so).
4. **Warmup** — the receiver needs ~100 packets of gain-calibration before
   recommendations start (`CONFIG_CSI_DQN_WARMUP_PACKETS`); discard the
   first 5 seconds of each run if you want to be strict.

## What "the bandit wins" looks like

- NLOS cliff scenario: bandit ≈ static MCS3 on reward/PDR, both far above
  Minstrel if Minstrel probes into the dead MCS5–7 region (its throughput
  prior favors high rates); bandit should hold low MCS with near-zero losses.
- LOS scenario: bandit ≈ Minstrel ≈ static-high on median, bandit equal or
  better on mean reward without the probing dips Minstrel takes.
- The decisive comparison is **the same model, no per-scenario retuning,
  winning or tying in both scenarios** — that is what neither Minstrel's
  prior nor any static can do.
