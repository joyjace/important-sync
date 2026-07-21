# `link_v7c_ht20_v1` CSI Feature Contract

This contract is the shared boundary for offline dataset generation and live
receiver inference. It fixes the representation problems in experimental
`link_v6` without changing legacy model layouts; its versioned dataset,
training, export, and firmware paths are opt-in.

## Input

- Exactly 114 gain-compensated signed `int16` scalars: 57 HT20 complex bins.
- Collector order is `[imaginary, real]`; the canonical complex value is
  `H = real + j*imaginary`.
- Raw bin 28 is DC/null and is excluded.
- Dataset construction requires `data_len=114`, `first_word=0`, a normalized
  ESP32-C5 collector format (`c5c6` or CRC-protected `b64`), and
  `iq_pairs=57` when that derived field is present.
- `aggregate` CSI deduplication is rejected because averaged RF metadata plus
  one raw CSI frame is not a coherent physical observation.

## Output

The transform emits 166 `float32` values in this fixed order:

1. 56 active-bin amplitudes: raw bins 0..27, then 29..56.
2. 54 corrected adjacent-phase real components. Pairs do not cross DC.
3. 54 corrected adjacent-phase imaginary components in the same pair order.
4. Phase-valid fraction.
5. Weighted differential-phase coherence.

An adjacent phase pair is valid only when both amplitudes are at least
`max(2.0, 0.10 * median(active amplitudes))`. Invalid pairs are `(0, 0)`.
Valid differential phasors are weighted by the weaker endpoint. Their common
phasor is removed. Adjacent differences already reject packet-wide phase
rotation; removing their common phasor additionally rejects a packet-wide
linear phase ramp while preserving local frequency structure.

The canonical contract SHA-256 is:

```text
df4f262b3fdf57f2f693b40b8584c08d5193ba092290d7f771e1a52575c8603a
```

Any semantic or ordering change requires a new contract ID and digest.

Dataset CSVs store the three vector sections separately as
`iq_active_amplitudes`, `iq_phase_diff_real`, and `iq_phase_diff_imag`, followed
by the two scalar quality columns. Concatenating them in that order reconstructs
the 166-value vector without storing a duplicate copy. `iq_raw` and the legacy
summary columns deliberately retain the original 57-bin amplitude view so the
existing amplitude-only pipeline remains a valid baseline.

## Build A Qualification Dataset

From `tools/rl/DQN`:

```bash
python3 shared/build_all_dqn_datasets.py \
  --csi-feature-contract link_v7c_ht20_v1 \
  --state-alignment previous_csi \
  --require-metadata \
  --reward-mode utility \
  --overwrite
```

Non-legacy batch filenames include `link_v7c_ht20_v1`, preventing collisions
with amplitude datasets. Every v7 CSV gets an adjacent
`.feature_contract.json` file containing the exact contract, artifact and
source hashes, build settings, capture summaries, and normalized scenario
metadata. Contract-aware merges verify all parent hashes and refuse mixed or
tampered inputs.

The builder still permits `same_packet` for representation diagnostics, but
its sidecar is explicitly marked `qualification.status=noncausal`. Only
`previous_csi` artifacts with exact gain-compensation provenance are marked as
deployment candidates for the later model qualification stage.

## Receiver-Only Model State

Training and firmware bind the 166 CSI features to the separate state contract
`link_v7c_receiveronly_v1` (SHA-256
`8c88cd04a5e2c28366f567a5928e8df6df685f05284e5683a6a4d3cfd0beb790`). It
contains exactly 181 `float32` values: the 166 values above followed by, in
order:

1. `rssi`, `snr`, `fft_gain`, `agc_gain`, `channel`, and `sig_len`.
2. `log1p_state_age_packets`, `log1p_state_packet_gap`,
   `log1p_state_missing_packets`, and `state_is_stale`.
3. Mean, standard deviation, p10, p50, and p90 over the same 56 active-bin
   amplitudes used by this contract.

The state never contains source MCS. Train `link_v7c` models with
`--ignore-state-mcs`; an incompatible checkpoint is rejected by the trainer,
exporter, firmware compile-time checks, or receiver startup checks.

Compact `CSI_B64` version 2 records the gain as its exact eight-digit
IEEE-754 binary32 bit pattern. The collector reconstructs that float32, rounds
the multiplication as float32, and truncates toward zero, matching live
firmware. Normalized rows carry `b64_version=2` and
`gain_compensation_exact=1`. Version 1 B64 captures remain accepted by the
dataset builder for historical analysis, but their original six-decimal gain
is irreversible and the builder records an explicit qualification blocker.
The contract-bound trainers fail closed on that blocker, so such data cannot
produce a live-deployable v7c checkpoint. `c5c6`/legacy `CSI_DATA` captures are
already device-compensated and therefore satisfy this boundary.

The sidecar keeps the existing `qualification.status=candidate` vocabulary for
compatible downstream tooling and adds `qualification.deployment_candidate`.
A candidate requires causal `previous_csi` alignment and exact compensation for
every B64 frame. Contract-aware merges propagate all parent blockers and cannot
promote a causal-but-inexact parent back to candidate status.

## Qualification And Deployment Runbook

The Python transform, portable C99 transform, causal 181-value training state,
model exporters, and guarded firmware path are implemented. This makes a model
deployable, but does not qualify a particular checkpoint. Keep the current
amplitude champion in control until the following gates pass.

### Current staged canary (2026-07-19)

The workspace contains a phase-regularized h128 checkpoint at
`tools/rl/DQN/models/v3_1_linkv7c_phasereg075_c050_receiveronly_reward_model_holdout_f202_rooftop02/action_reward_model.pth`
(SHA-256 `8472e769f2bba57cb1978c20f733e65ff0cced480cdc2471bfa66ecdcd134e22`).
Its exported staging header is
`csi_recv/main/generated_reward_model_linkv7c_canary.h` (SHA-256
`8d0d8c023bc12f24b091e75e5820949f93a46e8549b5795426e7643c05ec4459`).
The v2.6 amplitude header remains untouched.

On the same fresh-state scenario holdouts, using empirical logged propensity:

| Policy / holdout | Coverage | SNIPS reward | p95 service | capped mean | composite |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2.6 amplitude / healthy | 16.57% | 0.8115 | 4.999 ms | 0.9743 ms | 0.8123 |
| v3.1 v7c / healthy | 11.45% | 0.7994 | 4.633 ms | 1.0022 ms | 0.8189 |
| v2.6 amplitude / failure | 18.46% | 0.8093 | 4.733 ms | 1.0291 ms | 0.9725 |
| v3.1 v7c + live guard / failure | 15.73% | 0.8100 | 4.185 ms | 0.9926 ms | 0.9867 |

The guard mirrors firmware exactly: when SNR is below 15 dB, it caps the
reward-model recommendation at MCS1. Reproduce the guarded offline result with
`predict_reward_model.py --low-snr-threshold-db 15 --low-snr-max-mcs 1`.
The unguarded prediction CSVs are retained separately, so this effect is never
hidden inside the neural-network score.

A phase-zero ablation was also run by replacing all 110 phase-derived state
values—the 108 differential-phase values plus valid fraction and coherence—with
their checkpoint training means (all corresponding normalized inputs are
exactly zero), while leaving amplitude, metadata, model weights, and the live
guard unchanged:

| Input / holdout | Coverage | SNIPS reward | p95 service | capped mean | composite |
| --- | ---: | ---: | ---: | ---: | ---: |
| full v7c / healthy | 11.45% | 0.7994 | 4.633 ms | 1.0022 ms | 0.8189 |
| phase-masked v7c / healthy | 13.77% | 0.7966 | 4.831 ms | 1.0206 ms | 0.7715 |
| full v7c + guard / failure | 15.73% | 0.8100 | 4.185 ms | 0.9926 ms | 0.9867 |
| phase-masked v7c + guard / failure | 16.33% | 0.8185 | 3.691 ms | 0.9247 ms | 0.9988 |

Phase therefore adds held-out value on the healthy scenario but not on the
failure scenario. This mixed ablation does not establish a general incremental
phase benefit and is an additional reason not to promote the checkpoint yet.
The ablation changes 32.77% of guarded healthy actions and 16.16% of guarded
failure actions. It is reproducible with
`predict_reward_model.py --mask-v7c-phase`; each prediction row records
`phase_masked`, so it cannot be confused with the deployable full-input result.

This is a canary, not a silent champion replacement: healthy reward is slightly
lower even though healthy tail/composite and difficult-link results pass the
staging gates. In `tools/device_menu.py`, select **v3.1 robust full-CSI
reward-model canary** to build it. Select **v2.6 amplitude reward-model champion
(rollback/control)** to return to the prior header. The live canary build is
`0xf25c0` bytes and leaves `0xda40` bytes (5%) in the default 1 MiB app
partition.

### 1. Prove the data artifact

- Use randomized per-packet action logging, causal `previous_csi` alignment,
  exact gain compensation, and scenario metadata.
- Inspect every adjacent `.feature_contract.json`. It must report
  `qualification.deployment_candidate=true` with no blocking reasons.
- Divide train and holdout by complete run/scenario/day/device groups before
  merging. The trainer's shuffled row evaluation split is useful for progress
  monitoring only; it is not the qualification holdout.
- Keep the holdout CSV and sidecar untouched after choosing them. Record their
  SHA-256 values with the experiment results.

### 2. Run a paired amplitude/full-CSI ablation

Train the same two-head architecture, seed, rows, objective, and hyperparameters
twice. Use `link_v5` for the receiver-only amplitude control and `link_v7c` for
the candidate; both require `--ignore-state-mcs`.

```bash
python3 action_reward_model/train_bandit_model.py \
  --dataset datasets/<train>.csv \
  --output models/<run>/amplitude/bandit_model.pth \
  --state-schema link_v5 --ignore-state-mcs \
  --epochs 20 --batch-size 512 --hidden-dim 128 --seed 42

python3 action_reward_model/train_bandit_model.py \
  --dataset datasets/<train>.csv \
  --output models/<run>/full_csi/bandit_model.pth \
  --state-schema link_v7c --ignore-state-mcs \
  --epochs 20 --batch-size 512 --hidden-dim 128 --seed 42
```

Repeat across multiple seeds. Score both checkpoints on the same
scenario-held-out datasets with `predict_bandit_model.py`, then evaluate
randomized captures with `evaluate_policy_delay.py` and uniform logged
propensities (`--logged-propensity uniform`). Do not qualify on
training/evaluation loss or action agreement alone. Require full CSI to improve
or tie the amplitude control on SNIPS/replay reward, delivery rate, and tail
service time across the difficult holdouts. A phase-zero or within-run
phase-shuffle ablation should remove that gain; otherwise phase has not shown
incremental value.

When matching the current direct reward-model champion, keep the architecture
and sample budget identical and regularize only the new differential-phase
section. For example:

```bash
python3 action_reward_model/train_reward_model.py \
  --dataset datasets/<train>.csv \
  --output models/<run>/action_reward_model.pth \
  --state-schema link_v7c --ignore-state-mcs \
  --target-column reward --objective maximize \
  --observed-only --fresh-state-only \
  --epochs 15 --batch-size 512 --hidden-dim 128 \
  --max-train-rows 400000 --loss huber \
  --phase-dropout 0.75 --phase-consistency-weight 0.5 \
  --seed 42
```

Phase dropout replaces all 110 normalized phase-derived values—the 108
differential-phase values plus valid fraction and coherence—with their training
mean for a random fraction of frames. The consistency loss limits the score
change between full and phase-masked states. These are safeguards against
scenario-specific phase shortcuts, not evidence that phase adds value; the
held-out and phase-ablation gates above remain mandatory.

### 3. Export without bypassing contract checks

Export to a staging header first and verify it against a deployment-candidate
holdout:

```bash
python3 action_reward_model/export_bandit_model_to_c_header.py \
  --model models/<run>/full_csi/bandit_model.pth \
  --output models/<run>/full_csi/generated_bandit_model.h \
  --verify-dataset datasets/<holdout>.csv \
  --verify-rows 512
```

The header must identify `link_v7c_ht20_v1`,
`link_v7c_receiveronly_v1`, 166 CSI values, and 181 total state values. Install
it as `csi_recv/main/generated_bandit_model.h` only after review, then rebuild
from a clean firmware build directory. The receiver deliberately fails to
compile or aborts at startup when an exported feature/state ID, digest, count,
or source-MCS setting disagrees with firmware.

### 4. Shadow before allowing control

Enable receiver inference with `CONFIG_CSI_DQN_MCS_RECOMMENDATION_ENABLED=1`
and `CONFIG_CSI_MCS_POLICY_MODEL=1`, but keep
`CONFIG_CSI_DQN_REMOTE_RECOMMENDATION_ENABLED=0` on the sender. Let the sender
run the chosen baseline controller while the candidate only emits telemetry.
On receiver startup, require exactly:

```text
CSI_MODEL,bandit,link_v7c_ht20_v1,link_v7c_receiveronly_v1,166,181
```

Across representative LOS, NLOS-cliff, movement, and induced-blackout runs,
check that outputs remain finite, queue drops and invalid-state rejections stay
at zero, action distributions are plausible, and p99 `DQN_INFERENCE` latency
is comfortably below the 5 ms packet interval. A shadow run validates runtime
health and decisions, not end-to-end control benefit.

### 5. Canary, A/B, and rollback

For the live v3.1 canary, use the custom reward-model recommendation channel,
not the receiver DQN live-policy channel. The **v3.1 robust full-CSI
reward-model canary** preset in `tools/device_menu.py` configures the coupled
settings:

- Sender: `CONFIG_CSI_LIVE_MCS_ALGO=1` (`CUSTOM_POLICY`),
  `CONFIG_CSI_REMOTE_MCS_RECOMMENDATION_ENABLED=1`, and
  `CONFIG_CSI_DQN_REMOTE_RECOMMENDATION_ENABLED=0`; stop-and-wait ACK timing
  and the full MCS0..MCS7 range are pinned by the preset.
- Receiver: `CONFIG_MCS_RECOMMENDATION_ENABLED=1`,
  `CONFIG_CSI_DQN_MCS_RECOMMENDATION_ENABLED=0`, and
  `CONFIG_CSI_REWARD_MODEL_VARIANT=1`.
- Guard: `CONFIG_CSI_REWARD_MODEL_SNR_GUARD_ENABLED=1`, with the 15 dB
  threshold and MCS1 low-SNR cap used for offline qualification.

On receiver startup, require:

```text
CSI_MODEL,reward_model,link_v7c_ht20_v1,link_v7c_receiveronly_v1,166,181
```

The receiver then emits `REWARD_INFERENCE` health rows at most once per second
during normal operation and immediately on an error or latency overrun. Monitor
`inference_us`, `queue_depth`, `total_drops`, `invalid_inputs`, `queue_full`, `state_failures`,
`coalesced`, `model_nonfinite`, and `inference_overruns`. Non-finite scores are
not converted into an action: firmware counts the event and forces a fresh
MCS0/confidence-zero safety recommendation, which the sender accepts even when
a positive ordinary confidence threshold is configured.

Retain the sender recommendation-age check
(`CONFIG_CSI_REMOTE_MCS_MAX_AGE_MS`, 300 ms in the default profile),
sequence-gap guard, and emergency failure stepdown
(`CONFIG_CSI_DQN_REMOTE_MAX_SEQ_GAP=50`,
`CONFIG_CSI_DQN_FAILURE_STEPDOWN_ENABLED=1`, and a failure count of 8). The
historical `DQN` names on the latter safeguards are shared by the custom
reward-model path. Start in bounded, recoverable scenarios, then follow
[`tools/AB_TEST_PROTOCOL.md`](../../../AB_TEST_PROTOCOL.md) with mirrored runs
against the amplitude champion, Minstrel-like control, and strong static MCS
baselines.

Stop and roll back immediately on a contract/startup failure, non-finite score,
repeated queue drops, inference overruns, stale recommendation uptake, failure
to step down during a blackout, or a material PDR/tail-latency regression. Keep
the last accepted amplitude firmware binary and model header available so
rollback is a reflash/configuration change rather than an emergency retrain.

Record the dataset/sidecar hashes, checkpoint hash, exported-header hash,
firmware commit, build configuration, board pair, and live-run IDs for every
canary. Passing one environment is not sufficient to promote the model.

## Contract Tests

Run the contract tests from the repository root using the model-training
environment (the repository `.venv` in this workspace):

```bash
.venv/bin/python -m unittest discover -s tools/rl/DQN/shared/tests -v
```

These tests include Python/C feature parity, threshold and DC edge cases,
dataset/sidecar tamper rejection, exact B64 gain compensation, and end-to-end
B64-to-181-value receiver-state parity. A host C99 compiler is required for the
portable-runtime tests.
