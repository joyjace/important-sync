# Randomized MCS Collection (RANDOM_SWEEP)

This is the collection mode for RL/bandit training data. It replaces the 5k-packet
fixed-MCS block sweeps with a **shuffled round-robin** that changes MCS every packet,
and replaces the lossy `CSI_DATA` serial path with a compact CRC-protected format.

## Why

The old block sweeps have two problems that no amount of model tuning can fix:

1. **Confounding / no counterfactuals.** Each MCS occupied one contiguous ~25 s time
   window, so the action was perfectly correlated with time. For any channel instant
   only one action was ever observed, the conditional propensity was degenerate, and
   IPS/SNIPS numbers from `evaluate_policy_delay.py` were biased. With RANDOM_SWEEP,
   every window of 8 packets (40 ms at 200 Hz — inside channel coherence) contains all
   8 MCS in a fresh random order: the logged action is independent of the channel with
   known per-packet propensity **1/8**, so offline evaluation and counterfactual reward
   modeling become honest.

2. **CSI loss from the serial path, not RF.** The legacy `CSI_DATA` printer emitted
   ~1 KB lines with 234 separate `ets_printf` calls from the Wi-Fi callback:
   at 921,600 baud that caps out near ~85 frames/s (only ~37% of packets got CSI),
   blocks the Wi-Fi task while printing, and other console output spliced into lines
   mid-frame (the `wrong_field_count` / `data_len_mismatch` errors in
   `csi_data_log.txt`). The new path queues frames to a dedicated task, writes each
   line with a single `fwrite`, base64-encodes the payload (~404-byte lines,
   ~227 frames/s capacity at 921,600 baud), and adds a CRC16 so the host *knows*
   a line is intact instead of guessing.

Side benefits: during high-MCS outages the interleaved low-MCS packets keep the CSI
stream fresh (no more blind stretches), and `state_mcs_index` varies independently of
the action, removing the fixed-sweep "keep current MCS" shortcut (no more
`--ignore-state-mcs` workarounds).

## Firmware configuration

### Sender (`csi_send/main/app_main.c`)

```c
#define CONFIG_RATE_SWITCH_MODE             3  // RANDOM_SWEEP
#define CONFIG_RATE_SWEEP_GROUP_SIZE        1  // packets per MCS step (keep 1)
```

Requirements enforced at compile time: `CONFIG_ACK_TIMING_MODE == 2` (stop-and-wait,
already the default) so no packet is in flight when the rate changes and the logged
`configured_rate` is exact per packet. Live MCS selection must stay disabled
(`CONFIG_CSI_LIVE_MCS_SELECTION_ENABLED 0`) during collection.

At startup the sender prints a provenance line that the host logs:

```text
RATE_POLICY,shuffled_round_robin,group_size=1,actions=8,propensity=0.125000
```

There are no `ACK_RESET` segment resets in this mode; the per-packet MCS is recovered
from `configured_rate` in every `ACK_STATUS` row exactly as before.

### Receiver (`csi_recv/main/app_main.c`)

```c
#define CONFIG_CSI_OUTPUT_ASYNC_COMPACT  1   // default; 0 restores legacy CSI_DATA
```

Current serial format (one line per frame, single write, CRC16-CCITT over the text
before the CRC field):

```text
CSI_B64,<ver>,<seq>,<mac>,<rssi>,<rate>,<noise_floor>,<fft_gain>,<agc_gain>,
<channel>,<timestamp>,<sig_len>,<rx_state>,<len>,<first_word_invalid>,
<gain_f32_hex>,i8,<base64 raw int8 payload>,<crc16 hex>
```

Firmware emits version `2`. `gain_f32_hex` is the exact IEEE-754 binary32 bit
pattern as eight hexadecimal digits; for example, `3F800000` represents `1.0f`.
The payload is the **raw** CSI buffer. The host reconstructs the exact float32
gain, performs float32 multiplication, and truncates toward zero just like the
firmware's `(int16_t)(gain * value)` operation. The normalized CSV records
`b64_version=2` and `gain_compensation_exact=1` so qualification tooling can
verify this boundary.

The parser remains backward-compatible with version `1` lines containing a
six-decimal gain. Those captures remain usable for historical dataset analysis,
but their CSV rows are marked `gain_compensation_exact=0` because the original
float bits cannot be recovered. Contract-bound trainers reject their sidecar,
so they cannot qualify a model artifact for live deployment. Legacy
`CSI_DATA`/`c5c6` rows already contain device-compensated integers and count as
exact. If frames ever outpace the writer, the device reports
`CSI_DROP,<seq>,<total>` lines instead of silently losing them.

### Baud rate (optional but recommended)

At the current 921,600 baud the budget is ~227 frames/s against 200 packets/s — it
fits, but with limited headroom. For margin, raise the receiver console to 2 Mbaud
(`CONFIG_ESP_CONSOLE_UART_BAUDRATE=2000000` via menuconfig, if your USB-serial
adapter supports it) and pass `-b 2000000` to the host reader.

## Collecting a run

Same command as before — the parser auto-detects `CSI_B64` and legacy `CSI_DATA`:

```bash
python3 tools/csi_data_read_parse_SSH.py \
  -p /dev/ttyUSB1 -b 921600 \
  -sp /dev/ttyUSB0 -sb 921600
```

On Ctrl-C the summary now also reports `CSI CRC mismatches` and
`CSI frames dropped on device` — both should be ~0 on a healthy setup.

Add the logging policy to the run metadata JSON so it flows into datasets as
`meta_*` columns:

```json
{
  "scenario_id": "S-xx",
  "logging_policy": "shuffled_round_robin",
  "rate_sweep_group_size": 1,
  "action_propensity": 0.125,
  "...": "existing fields (distance_m, environment, motion, ...)"
}
```

## Scenario priorities

Train where the MCS decision actually matters. From the existing block-sweep data,
the "cliff" regimes are NLOS 4–8 m and LOS 7–12 m (e.g. NLOS 8 m: MCS0–3 ≈ 100% PDR
but MCS6 ≈ 15%, MCS7 ≈ 7%). Suggested mix per collection session:

- 6–8 runs in cliff regimes (mid-range NLOS, far LOS),
- 3–4 easy runs (close LOS) for policy coverage,
- a few mobility runs (walk, human crossing),
- optionally 1–2 legacy block-sweep runs per key scenario for continuity with the
  old baseline curves (interleaving changes queueing behavior slightly, so absolute
  service-time distributions are not directly comparable to block runs).

## Dataset building and evaluation

Build datasets exactly as before (`--state-alignment previous_csi`). Two things
change downstream:

1. **Propensity-correct evaluation is now valid.** Use the uniform logging model:

   ```bash
   python3 shared/evaluate_policy_delay.py \
     --predictions predictions/... \
     --logged-propensity uniform
   ```

   Expected coverage (predicted == logged) is ~12.5% by construction; the matched
   subset is now an unbiased sample, so the SNIPS/replay numbers are meaningful.

2. **Counterfactual reward models no longer need shortcut guards.** With actions
   randomized per packet, `state_mcs_index` is no longer a proxy for the action, and
   every state neighborhood contains all 8 actions within ±20 ms.

## After collection: training

Preferred learner on randomized data: the two-head bandit model
(`rl/DQN/action_reward_model/train_bandit_model.py`) — see the "Two-Head
Bandit Workflow" section in `rl/DQN/action_reward_model/README.md` for the
full train / predict / evaluate / export command sequence. For the DQN
comparison, train `rl/DQN/dqn_model/train_dqn.py` on the same data with
`--state-schema link_v3c` and gamma in {0, 0.9} as the key ablation.

## First-run sanity checklist

- `ack_data.csv`: per-MCS counts equal within ±group size over any window; ~5k per
  MCS per 40k-packet run.
- MCS transitions ≈ number of packets (was: 8 per run).
- CSI rows ≈ number of delivered data packets (was: ~37%).
- `csi_data_log.txt`: near-zero malformed lines; `CSI CRC mismatches: 0`.
- `RATE_POLICY` line present in `send_serial_log.txt` (or console).
- `state_age_packets` in built datasets: median 1, no long blind stretches during
  high-MCS loss bursts.
