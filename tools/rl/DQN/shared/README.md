# Shared Dataset And Evaluation Tools

These utilities are shared by both `action_reward_model/` and `dqn_model/`.

Run examples from:

```bash
cd /home/gje1671/Downloads/esp-csi-master/examples/get-started/tools/rl/DQN
```

## Files

| File | Purpose |
| --- | --- |
| `build_dqn_dataset.py` | Build one dataset CSV from one CSI/ACK scenario folder. |
| `build_all_dqn_datasets.py` | Build datasets for every scenario under `tools/Scenarios`. |
| `merge_dqn_datasets.py` | Merge per-scenario dataset CSVs into one training CSV. |
| `csi_link_v7c.py` | Canonical compact HT20 amplitude + differential-phase transform. |
| `CSI_LINK_V7C_CONTRACT.md` | Versioned input/output, provenance, and qualification contract. |
| `build_model_augmented_stale_dataset.py` | Add modeled all-action stale trajectories from genuine ACK-loss streaks. |
| `audit_dqn_dataset.py` | Report loss/action coverage and validate modeled temporal links. |
| `evaluate_policy_delay.py` | Evaluate recommendation CSVs with replay, IPS, and SNIPS metrics. |
| `standardize_scenario_names.py` | Create or apply a canonical scenario naming manifest. |
| `METADATA_GUIDE.md` | Add scenario metadata columns to generated datasets. |

## Build One Dataset

```bash
python3 shared/build_dqn_dataset.py \
  --csi-csv ../../Scenarios/LOS/LOS_1m_F202_run02/csi_data.csv \
  --ack-csv ../../Scenarios/LOS/LOS_1m_F202_run02/ack_data.csv \
  --output datasets/rl_dqn_dataset_LOS_1m_F202_run02_robust_delay.csv \
  --reward-mode robust_delay \
  --state-alignment previous_csi
```

## Build All Datasets

```bash
python3 shared/build_all_dqn_datasets.py \
  --reward-mode robust_delay \
  --state-alignment previous_csi \
  --merge-groups \
  --merge-balance equal_rows \
  --overwrite
```

To build strict full-CSI qualification datasets without colliding with legacy
amplitude outputs:

```bash
python3 shared/build_all_dqn_datasets.py \
  --csi-feature-contract link_v7c_ht20_v1 \
  --reward-mode utility \
  --state-alignment previous_csi \
  --require-metadata \
  --overwrite
```

This writes a verified `.feature_contract.json` beside each dataset. See
[`CSI_LINK_V7C_CONTRACT.md`](CSI_LINK_V7C_CONTRACT.md) before adding the new
state to training or firmware.

## Merge Existing Datasets

```bash
python3 shared/merge_dqn_datasets.py \
  --input-glob 'datasets/rl_dqn_dataset_LOS_*_robust_delay.csv' \
  --output datasets/rl_dqn_dataset_all_LOS_robust_delay.csv \
  --balance equal_distance_rows \
  --seed 42
```

## Evaluate Predictions

```bash
python3 shared/evaluate_policy_delay.py \
  --predictions predictions/dqn_mcs_recommendations_LOS_1m_F202_utility.csv \
  --logged-propensity empirical
```

The evaluator accepts prediction files from both `dqn_model/predict_dqn.py` and `action_reward_model/predict_reward_model.py`.
