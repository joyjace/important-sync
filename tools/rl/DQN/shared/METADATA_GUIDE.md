# Scenario Metadata Guide

Scenario metadata adds experiment context to generated dataset CSVs. The metadata columns are useful for analysis and grouping, but they are not used as model input by the current training scripts.

## Metadata JSON

Create one metadata file per scenario run:

```json
{
  "scenario": "LOS_1m_F202_run02",
  "distance_m": 1.0,
  "obstacle_type": "none",
  "movement_type": "static",
  "channel_condition": "LOS",
  "location": "lab",
  "notes": "Line-of-sight baseline"
}
```

The builder also accepts the singleton-array representation used by existing
captures:

```json
[
  {
    "scenario_id": "S-11",
    "distance_m": 1,
    "environment": "LOS",
    "motion": "static"
  }
]
```

Fields are optional. Common fields are `scenario`, `distance_m`, `obstacle_type`, `movement_type`, `channel_condition`, `date`, `location`, and `notes`.

## Build A Dataset With Metadata

Run from:

```bash
cd /home/gje1671/Downloads/esp-csi-master/examples/get-started/tools/rl/DQN
```

```bash
python3 shared/build_dqn_dataset.py \
  --csi-csv ../../Scenarios/LOS/LOS_1m_F202_run02/csi_data.csv \
  --ack-csv ../../Scenarios/LOS/LOS_1m_F202_run02/ack_data.csv \
  --output datasets/rl_dqn_dataset_LOS_1m_F202_run02_robust_delay.csv \
  --reward-mode robust_delay \
  --state-alignment previous_csi \
  --metadata-json ../../Scenarios/LOS/LOS_1m_F202_run02/metadata.json
```

The output CSV will include columns such as `meta_scenario`, `meta_distance_m`, `meta_obstacle_type`, `meta_movement_type`, and `meta_channel_condition`.

For v7 qualification builds, `build_all_dqn_datasets.py` automatically
discovers either `metadata.json` or the historical spelling `matadata.json` in
each scenario folder. Add `--require-metadata` so a missing file is an error;
on the legacy feature default, this flag also explicitly opts into metadata
columns and keeps old invocations unchanged.

## Train After Metadata Is Added

DQN example:

```bash
python3 dqn_model/train_dqn.py \
  --dataset datasets/rl_dqn_dataset_LOS_1m_F202_run02_robust_delay.csv \
  --output models/dqn_mcs_model_LOS_1m_F202_robust_delay.pth \
  --epochs 30
```

Action-reward example:

```bash
python3 action_reward_model/train_reward_model.py \
  --dataset datasets/rl_dqn_dataset_LOS_1m_F202_run02_robust_delay.csv \
  --output models/action_reward_model_LOS_1m_F202_service_ms.pth \
  --target-column service_ms \
  --objective minimize \
  --epochs 30
```
