# CSI_SEND

This sender firmware now supports live MCS selection in addition to legacy fixed/time/packet sweep modes.

## Live MCS Selection

The implementation is in `main/app_main.c`. In the normal project workflow,
configure it from the repository root with:

```bash
python tools/device_menu.py
```

Then open sender parameters and choose the live MCS policy.
Applying the profile updates the source constants used by the normal
`idf.py build` flow.

The sender application console is configured for 921600 baud in
`sdkconfig.defaults`, matching the receiver and the capture tool's default
`--send-baud`. When building through `tools/device_menu.py`, an existing local
`sdkconfig` is migrated automatically so an older 115200 configuration cannot
continue throttling the per-packet ACK log.

Key options:

- CONFIG_CSI_LIVE_MCS_SELECTION_ENABLED
  - Enable live per-packet adaptation.
  - Disable to use CONFIG_RATE_SWITCH_MODE behavior instead:
    0 = TIME_BASED, 1 = PACKET_BASED, 2 = STATIC (fixed rate),
    3 = RANDOM_SWEEP (shuffled round-robin, uniform per-packet MCS with
    propensity 1/8 — the mode used for unbiased dataset collection; requires
    stop-and-wait ACK timing).
- CONFIG_CSI_LIVE_MCS_ALGO
  - 0: MINSTREL_LIKE (EWMA success + expected throughput + periodic probing)
  - 1: CUSTOM_POLICY (receiver recommendations only)
  - 2: DQN_POLICY (dedicated DQN receiver recommendations + loss fallback)
- CONFIG_CSI_LIVE_MCS_MIN_INDEX, CONFIG_CSI_LIVE_MCS_MAX_INDEX
- CONFIG_CSI_MINSTREL_UPDATE_EVERY_PKTS
- CONFIG_CSI_MINSTREL_PROBE_EVERY_PKTS
- CONFIG_CSI_MINSTREL_EWMA_ALPHA_NUM, CONFIG_CSI_MINSTREL_EWMA_ALPHA_DEN
- CONFIG_CSI_CUSTOM_POLICY_DEFAULT_MCS
  - Initial MCS used by CUSTOM_POLICY until the first valid receiver recommendation is applied.
  - Default is MCS0 so the receiver can hear initial packets and bootstrap recommendations.
- CONFIG_CSI_LIVE_MCS_DECISION_LOG_ENABLED

## Receiver-Driven MCS Selection

Sender can accept compact recommendation frames from receiver and apply them to the next packet. In CUSTOM_POLICY mode, this is the only mechanism that changes MCS after initialization; local ACK statistics are logged but do not choose a replacement MCS.

Controls in main/app_main.c:

- CONFIG_CSI_REMOTE_MCS_RECOMMENDATION_ENABLED
	- 1: accept receiver recommendations
	- 0: ignore receiver recommendations
- CONFIG_CSI_REMOTE_MCS_MIN_CONFIDENCE
	- minimum confidence (0-100) required to apply recommendation
- CONFIG_CSI_REMOTE_MCS_MAX_AGE_MS
	- recommendation freshness limit in milliseconds

Current logic applies the latest receiver recommendation to later packets when it passes freshness/confidence checks.

## DQN Receiver-Driven Selection

Select `DQN_POLICY - receiver DQN recommendations only` in
`tools/device_menu.py`. Its controls are separate from the custom-policy
controls:

- `CONFIG_CSI_DQN_REMOTE_RECOMMENDATION_ENABLED`
- `CONFIG_CSI_DQN_DEFAULT_MCS`
- `CONFIG_CSI_DQN_REMOTE_MIN_CONFIDENCE`
- `CONFIG_CSI_DQN_REMOTE_MAX_AGE_MS`
- `CONFIG_CSI_DQN_REMOTE_MAX_SEQ_GAP` (default `50`: reject recommendations computed from CSI more than 50 packets old; `0` disables. The 50 ms age check stamps arrival time at the sender, so it cannot catch recommendations delayed in flight during an outage)
- `CONFIG_CSI_DQN_FAILURE_STEPDOWN_ENABLED` (default enabled; keep it enabled whenever `MAX_SEQ_GAP` is nonzero — stale-branch recommendations carry the old CSI seq and are rejected during a blackout, so the local stepdown is what ramps the rate down)
- `CONFIG_CSI_DQN_FAILURE_STEPDOWN_COUNT` (default `8` consecutive ACK failures per one-step ramp: full MCS7→0 in ~320 ms of total blackout at 200 pkt/s, while random loss below ~40% almost never triggers it)
- `CONFIG_CSI_DQN_LOG_ENABLED`

The sender applies each fresh DQN recommendation once. In DQN mode, sender ACK
statistics are logged but do not choose a replacement MCS, except for the
emergency local stepdown after the configured consecutive-failure count.
Minstrel and Custom behavior are unchanged.

## Decision CSV Logging

When CONFIG_CSI_LIVE_MCS_DECISION_LOG_ENABLED is enabled, the sender emits:

- ACK_POLICY_HEADER
- ACK_POLICY rows per packet with:
	- used_mcs, next_mcs, best_mcs, is_probe
	- used_ewma_pdr, used_ewma_service_us
	- selected_reward, which is 0.0 in CUSTOM_POLICY receiver-only mode

The sender also emits MCS change rows whenever the configured ESP-NOW MCS changes:

- MCS_CHANGE_HEADER
- MCS_CHANGE rows with:
	- seq, old_mcs, new_mcs
	- old_rate, new_rate
	- reason
		- initial
		- minstrel_best
		- minstrel_probe
		- receiver_recommendation
		- dqn_recommendation
		- dqn_failure_stepdown
		- dqn_hold
	- ts_us

With DQN logging enabled, `DQN_FEEDBACK` records the source sequence, target
sequence, selected MCS, confidence, Q margin, CSI RSSI/SNR, and feedback age.

## Notes About Minstrel

ESP-IDF/ESP-NOW does not expose Linux mac80211 Minstrel directly. The provided `MINSTREL_LIKE` mode is a firmware-level approximation designed for A/B comparison in this project.
