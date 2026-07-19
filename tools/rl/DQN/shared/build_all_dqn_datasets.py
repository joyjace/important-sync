#!/usr/bin/env python3
"""
Build DQN datasets for every scenario folder with CSI and ACK logs.

Typical usage:
    python build_all_dqn_datasets.py

Runs placed anywhere under tools/Scenarios/skip/ are ignored by default.

Build both classic reward variants:
    python build_all_dqn_datasets.py --reward-mode both

Rebuild existing outputs:
    python build_all_dqn_datasets.py --overwrite

Build per-run datasets and grouped merged datasets:
    python build_all_dqn_datasets.py --reward-mode robust_delay --merge-groups --overwrite
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

try:
    from .build_dqn_dataset import (
        CSI_FEATURE_CONTRACTS,
        LEGACY_CSI_FEATURE_CONTRACT,
        LINK_V7C_CONTRACT_SHA256,
        LINK_V7C_CONTRACT_ID,
        build_dataset,
        combine_parent_qualifications,
        feature_contract_sidecar_path,
        sha256_file,
        validate_feature_contract_record,
    )
except ImportError:
    from build_dqn_dataset import (
        CSI_FEATURE_CONTRACTS,
        LEGACY_CSI_FEATURE_CONTRACT,
        LINK_V7C_CONTRACT_SHA256,
        LINK_V7C_CONTRACT_ID,
        build_dataset,
        combine_parent_qualifications,
        feature_contract_sidecar_path,
        sha256_file,
        validate_feature_contract_record,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
DQN_ROOT = SCRIPT_DIR.parent
TOOLS_ROOT = DQN_ROOT.parents[1]
DEFAULT_SCENARIOS_DIR = TOOLS_ROOT / "Scenarios"
DEFAULT_OUTPUT_DIR = DQN_ROOT / "datasets"
DEFAULT_SKIP_DIR_NAME = "skip"
KNOWN_CONDITIONS = {"LOS", "NLOS"}
REWARD_SUFFIXES = ("fast_latency", "robust_delay", "log_delay", "utility", "delay", "pdr")


class ScenarioRun:
    def __init__(self, path: Path, scenario_id: str, condition: str | None, distance_label: str | None):
        self.path = path
        self.scenario_id = scenario_id
        self.condition = condition
        self.distance_label = distance_label


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def scenario_id_for_path(path: Path, scenarios_dir: Path) -> str:
    rel_parts = path.relative_to(scenarios_dir).parts
    if len(rel_parts) >= 2 and rel_parts[0] in KNOWN_CONDITIONS and path.name.startswith(f"{rel_parts[0]}_"):
        return sanitize_name(path.name)
    return sanitize_name("_".join(rel_parts))


def condition_for_path(path: Path, scenarios_dir: Path) -> str | None:
    rel_parts = path.relative_to(scenarios_dir).parts
    if rel_parts and rel_parts[0] in KNOWN_CONDITIONS:
        return rel_parts[0]
    return None


def normalize_distance_label(raw_value: str) -> str:
    numeric = float(raw_value)
    if numeric.is_integer():
        return f"{int(numeric)}m"
    return f"{str(numeric).replace('.', 'p')}m"


def distance_label_from_name(name: str) -> str | None:
    """Infer a distance label from common scenario naming schemes."""
    patterns = (
        r"(?i)(?:^|[_-])Distance[_-]?(\d+(?:\.\d+)?)m?(?:[_-]|$)",
        r"(?i)(?:^|[_-])(\d+(?:\.\d+)?)m(?:[_-]|$)",
        r"^(\d+)(?:[_-]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return normalize_distance_label(match.group(1))
    return None


def should_skip_path(path: Path, scenarios_dir: Path, skip_dir_name: str | None) -> bool:
    if not skip_dir_name:
        return False
    rel_parts = path.relative_to(scenarios_dir).parts
    return skip_dir_name in rel_parts


def discover_scenarios(scenarios_dir: Path, skip_dir_name: str | None) -> list[ScenarioRun]:
    """Return scenario directories that contain the expected input CSVs."""
    if not scenarios_dir.exists():
        raise SystemExit(f"Scenarios directory not found: {scenarios_dir}")

    scenario_runs = []
    for csi_csv in sorted(scenarios_dir.rglob("csi_data.csv")):
        path = csi_csv.parent
        if should_skip_path(path, scenarios_dir, skip_dir_name):
            continue

        ack_csv = path / "ack_data.csv"
        if not ack_csv.exists():
            continue

        scenario_runs.append(
            ScenarioRun(
                path=path,
                scenario_id=scenario_id_for_path(path, scenarios_dir),
                condition=condition_for_path(path, scenarios_dir),
                distance_label=distance_label_from_name(path.name),
            )
        )

    return scenario_runs


def reward_modes_from_arg(reward_mode: str) -> list[str]:
    if reward_mode == "both":
        return ["delay", "pdr"]
    if reward_mode == "all":
        return ["delay", "pdr", "fast_latency", "log_delay", "robust_delay", "utility"]
    return [reward_mode]


def metadata_path_for_scenario(path: Path) -> Path | None:
    """Find capture metadata, including the historical misspelled filename."""
    for filename in ("metadata.json", "matadata.json"):
        candidate = path / filename
        if candidate.exists():
            return candidate
    return None


def build_all(
    scenarios_dir: Path,
    output_dir: Path,
    reward_mode: str,
    dedup: str,
    skip_dir_name: str | None,
    include_substrings: list[str] | None,
    overwrite: bool,
    dry_run: bool,
    fast_target_ms: float,
    fast_ok_ms: float,
    robust_cap_quantile: float,
    robust_scale_ms: float,
    robust_loss_penalty_ms: float,
    utility_payload_bytes: int,
    utility_goodput_quantile: float,
    utility_loss_reward: float,
    utility_tail_target_ms: float,
    utility_tail_weight: float,
    state_alignment: str,
    stale_augment_ages: str | None,
    stale_augment_source: str,
    merge_groups: bool,
    merge_balance: str,
    merge_shuffle: bool,
    seed: int,
    csi_feature_contract: str = LEGACY_CSI_FEATURE_CONTRACT,
    require_metadata: bool = False,
) -> int:
    scenario_runs = discover_scenarios(scenarios_dir, skip_dir_name)
    include_substrings = include_substrings or []
    if include_substrings:
        scenario_runs = [
            scenario
            for scenario in scenario_runs
            if any(text in scenario.scenario_id for text in include_substrings)
        ]
    reward_modes = reward_modes_from_arg(reward_mode)

    print("[DQN Batch Dataset Builder]")
    print(f"  Scenarios dir: {scenarios_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Skip dir: {skip_dir_name or 'none'}")
    if include_substrings:
        print(f"  Include scenario substring(s): {', '.join(include_substrings)}")
    print(f"  Reward mode(s): {', '.join(reward_modes)}")
    print(f"  Dedup strategy: {dedup}")
    print(f"  State alignment: {state_alignment}")
    print(f"  CSI feature contract: {csi_feature_contract}")
    if stale_augment_ages:
        print(f"  Synthetic stale ages: {stale_augment_ages} ({stale_augment_source})")
    print(f"  Merge grouped datasets: {'yes' if merge_groups else 'no'}")
    if "fast_latency" in reward_modes:
        print(f"  Fast latency thresholds: target <= {fast_target_ms:.3f} ms, ok <= {fast_ok_ms:.3f} ms")
    if "robust_delay" in reward_modes:
        print(
            "  Robust delay settings: "
            f"cap_q={robust_cap_quantile:.3f}, scale={robust_scale_ms:.3f} ms, "
            f"loss_penalty={robust_loss_penalty_ms:.3f} ms"
        )
    if "utility" in reward_modes:
        print(
            "  Utility settings: "
            f"payload={utility_payload_bytes} bytes, q={utility_goodput_quantile:.3f}, "
            f"loss_reward={utility_loss_reward:.3f}, "
            f"tail_target={utility_tail_target_ms:.3f} ms, tail_weight={utility_tail_weight:.3f}"
        )
    print(f"  Matched scenarios: {len(scenario_runs)}")

    if not scenario_runs:
        print("  No scenario folders found with csi_data.csv and ack_data.csv")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    built = 0
    skipped = 0
    failed = 0
    dataset_paths_by_mode: dict[str, list[tuple[ScenarioRun, Path]]] = {mode: [] for mode in reward_modes}
    contract_tag = "" if csi_feature_contract == LEGACY_CSI_FEATURE_CONTRACT else f"_{csi_feature_contract}"

    for scenario in scenario_runs:
        metadata_path = metadata_path_for_scenario(scenario.path)
        if require_metadata and metadata_path is None:
            failed += len(reward_modes)
            print(f"[FAIL] {scenario.scenario_id}: required metadata.json/matadata.json is missing")
            continue
        build_metadata_path = (
            metadata_path
            if csi_feature_contract == LINK_V7C_CONTRACT_ID or require_metadata
            else None
        )
        for mode in reward_modes:
            output_csv = output_dir / f"rl_dqn_dataset_{scenario.scenario_id}{contract_tag}_{mode}.csv"
            dataset_paths_by_mode[mode].append((scenario, output_csv))

            if output_csv.exists() and not overwrite:
                if csi_feature_contract == LINK_V7C_CONTRACT_ID:
                    try:
                        load_verified_parent_contracts([output_csv])
                    except Exception as exc:
                        failed += 1
                        print(
                            f"[FAIL] {scenario.scenario_id} ({mode}) -> existing "
                            f"v7 dataset is not verifiable: {exc}"
                        )
                        continue
                print(f"[SKIP] {scenario.scenario_id} ({mode}) -> {output_csv.name} already exists")
                skipped += 1
                continue

            print(f"[BUILD] {scenario.scenario_id} ({mode}) -> {output_csv.name}")
            if build_metadata_path is not None:
                print(f"  Metadata: {build_metadata_path.name}")

            if dry_run:
                built += 1
                continue

            try:
                build_dataset(
                    scenario.path / "csi_data.csv",
                    scenario.path / "ack_data.csv",
                    output_csv,
                    dedup_strategy=dedup,
                    reward_mode=mode,
                    metadata_json=build_metadata_path,
                    fast_target_ms=fast_target_ms,
                    fast_ok_ms=fast_ok_ms,
                    robust_cap_quantile=robust_cap_quantile,
                    robust_scale_ms=robust_scale_ms,
                    robust_loss_penalty_ms=robust_loss_penalty_ms,
                    utility_payload_bytes=utility_payload_bytes,
                    utility_goodput_quantile=utility_goodput_quantile,
                    utility_loss_reward=utility_loss_reward,
                    utility_tail_target_ms=utility_tail_target_ms,
                    utility_tail_weight=utility_tail_weight,
                    state_alignment=state_alignment,
                    stale_augment_ages=stale_augment_ages,
                    stale_augment_source=stale_augment_source,
                    csi_feature_contract=csi_feature_contract,
                )
                built += 1
            except Exception as exc:
                failed += 1
                print(f"[FAIL] {scenario.scenario_id} ({mode}): {exc}")

    merge_failed = 0
    if merge_groups and not dry_run:
        merge_failed = merge_grouped_datasets(
            dataset_paths_by_mode=dataset_paths_by_mode,
            output_dir=output_dir,
            balance=merge_balance,
            shuffle=merge_shuffle,
            seed=seed,
            csi_feature_contract=csi_feature_contract,
        )

    print("[DQN Batch Dataset Builder Complete]")
    print(f"  Built: {built}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    if merge_groups:
        print(f"  Merge failures: {merge_failed}")

    return 1 if failed or merge_failed else 0


def scenario_name_from_dataset_path(path_obj: Path) -> str:
    name = path_obj.stem
    prefix = "rl_dqn_dataset_"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    for suffix in REWARD_SUFFIXES:
        suffix = f"_{suffix}"
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    contract_suffix = f"_{LINK_V7C_CONTRACT_ID}"
    if name.endswith(contract_suffix):
        name = name[: -len(contract_suffix)]
    return name


def load_and_tag_dataset(path_obj: Path, scenario: ScenarioRun) -> pd.DataFrame:
    df = pd.read_csv(path_obj)
    df = df.copy()
    df["source_file"] = path_obj.name
    df["source_scenario"] = scenario.scenario_id
    df["source_condition"] = scenario.condition or "unknown"
    df["source_distance"] = scenario.distance_label or "unknown"
    return df


def load_verified_parent_contracts(paths: list[Path]):
    """Verify every v7 parent artifact and return its common contract."""
    common_contract = None
    parents = []
    for path in paths:
        sidecar_path = feature_contract_sidecar_path(path)
        if not sidecar_path.exists():
            raise ValueError(f"Missing feature-contract sidecar for {path}")
        with sidecar_path.open("r", encoding="utf-8") as stream:
            sidecar = json.load(stream)
        recorded_hash = sidecar.get("artifact", {}).get("sha256")
        actual_hash = sha256_file(path)
        if recorded_hash != actual_hash:
            raise ValueError(f"Dataset hash does not match sidecar: {path}")
        contract = sidecar.get("feature_contract")
        validate_feature_contract_record(
            contract,
            expected_id=LINK_V7C_CONTRACT_ID,
            expected_sha256=LINK_V7C_CONTRACT_SHA256,
        )
        if common_contract is None:
            common_contract = contract
        elif contract != common_contract:
            raise ValueError(f"Feature-contract mismatch at {path}")
        parents.append(
            {
                "artifact_path": str(path.resolve()),
                "artifact_sha256": actual_hash,
                "manifest_path": str(sidecar_path.resolve()),
                "manifest_sha256": sha256_file(sidecar_path),
                "qualification": sidecar.get("qualification", {}),
            }
        )
    return common_contract, parents


def write_merged_dataset(
    items: list[tuple[ScenarioRun, Path]],
    output_csv: Path,
    balance: str,
    shuffle: bool,
    seed: int,
    csi_feature_contract: str = LEGACY_CSI_FEATURE_CONTRACT,
) -> None:
    existing_items = [(scenario, path) for scenario, path in items if path.exists()]
    if not existing_items:
        print(f"[MERGE SKIP] {output_csv.name}: no existing input datasets")
        return

    paths = [path for _scenario, path in existing_items]
    frames = [load_and_tag_dataset(path, scenario) for scenario, path in existing_items]
    original_rows = [len(frame) for frame in frames]
    common_contract = None
    parents = []
    if csi_feature_contract == LINK_V7C_CONTRACT_ID:
        common_contract, parents = load_verified_parent_contracts(paths)

    if balance == "equal_rows":
        min_rows = min(len(df) for df in frames)
        frames = [df.sample(n=min_rows, random_state=seed).reset_index(drop=True) for df in frames]

    merged = pd.concat(frames, ignore_index=True)
    if shuffle:
        merged = merged.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    if csi_feature_contract == LINK_V7C_CONTRACT_ID:
        for parent, original_count, selected_frame in zip(parents, original_rows, frames):
            parent["original_row_count"] = int(original_count)
            parent["selected_row_count"] = int(len(selected_frame))
        merged_qualification = combine_parent_qualifications(parents)
        sidecar = {
            "manifest_schema": "dqn_csi_feature_contract_merge/v1",
            "artifact": {
                "path": str(output_csv.resolve()),
                "sha256": sha256_file(output_csv),
                "row_count": int(len(merged)),
                "columns": list(merged.columns),
            },
            "feature_contract": common_contract,
            "merge": {
                "balance": balance,
                "shuffle": bool(shuffle),
                "seed": int(seed),
            },
            "parents": parents,
            "qualification": merged_qualification,
            "producer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(__file__),
            },
        }
        sidecar_path = feature_contract_sidecar_path(output_csv)
        temporary_path = sidecar_path.with_name(f"{sidecar_path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(sidecar, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary_path.replace(sidecar_path)
    print(f"[MERGE] {output_csv.name}: {len(existing_items)} files, {len(merged)} rows")


def merge_grouped_datasets(
    dataset_paths_by_mode: dict[str, list[tuple[ScenarioRun, Path]]],
    output_dir: Path,
    balance: str,
    shuffle: bool,
    seed: int,
    csi_feature_contract: str = LEGACY_CSI_FEATURE_CONTRACT,
) -> int:
    print("[DQN Grouped Dataset Merger]")
    failed = 0

    contract_tag = "" if csi_feature_contract == LEGACY_CSI_FEATURE_CONTRACT else f"_{csi_feature_contract}"
    for mode, items in dataset_paths_by_mode.items():
        try:
            write_merged_dataset(
                items,
                output_dir / f"rl_dqn_dataset_all_LOS_NLOS{contract_tag}_{mode}.csv",
                balance,
                shuffle,
                seed,
                csi_feature_contract,
            )

            for condition in sorted(KNOWN_CONDITIONS):
                condition_items = [
                    item for item in items if item[0].condition == condition
                ]
                write_merged_dataset(
                    condition_items,
                    output_dir / f"rl_dqn_dataset_all_{condition}{contract_tag}_{mode}.csv",
                    balance,
                    shuffle,
                    seed,
                    csi_feature_contract,
                )

            distances = sorted(
                {
                    scenario.distance_label
                    for scenario, _path in items
                    if scenario.distance_label
                },
                key=distance_sort_key,
            )
            unknown_distance = [scenario.scenario_id for scenario, _path in items if not scenario.distance_label]
            if unknown_distance:
                print(
                    "[WARN] Distance could not be inferred for: "
                    + ", ".join(sorted(unknown_distance))
                )

            for distance in distances:
                distance_items = [
                    item for item in items if item[0].distance_label == distance
                ]
                write_merged_dataset(
                    distance_items,
                    output_dir / f"rl_dqn_dataset_all_distance_{distance}{contract_tag}_{mode}.csv",
                    balance,
                    shuffle,
                    seed,
                    csi_feature_contract,
                )

                for condition in sorted(KNOWN_CONDITIONS):
                    condition_distance_items = [
                        item
                        for item in distance_items
                        if item[0].condition == condition
                    ]
                    write_merged_dataset(
                        condition_distance_items,
                        output_dir / f"rl_dqn_dataset_all_{condition}_distance_{distance}{contract_tag}_{mode}.csv",
                        balance,
                        shuffle,
                        seed,
                        csi_feature_contract,
                    )
        except Exception as exc:
            failed += 1
            print(f"[MERGE FAIL] {mode}: {exc}")

    return failed


def distance_sort_key(label: str) -> tuple[float, str]:
    match = re.match(r"(\d+(?:p\d+)?)m$", label)
    if not match:
        return (float("inf"), label)
    return (float(match.group(1).replace("p", ".")), label)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build DQN datasets for every folder in tools/Scenarios"
    )
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=DEFAULT_SCENARIOS_DIR,
        help="Directory containing scenario run folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated DQN dataset CSV files",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["delay", "pdr", "fast_latency", "log_delay", "robust_delay", "utility", "both", "all"],
        default="delay",
        help="Reward dataset(s) to build",
    )
    parser.add_argument(
        "--dedup",
        choices=["keep_first", "keep_latest", "aggregate"],
        default="keep_first",
        help="Deduplication strategy for duplicate sequence IDs",
    )
    parser.add_argument(
        "--csi-feature-contract",
        choices=CSI_FEATURE_CONTRACTS,
        default=LEGACY_CSI_FEATURE_CONTRACT,
        help=(
            "CSI transform for every output. Non-legacy contract names are included "
            "in filenames and produce verified feature-contract sidecars."
        ),
    )
    parser.add_argument(
        "--require-metadata",
        action="store_true",
        help="Fail a scenario when neither metadata.json nor matadata.json exists",
    )
    parser.add_argument(
        "--skip-dir-name",
        default=DEFAULT_SKIP_DIR_NAME,
        help="Name of a direct child folder under scenarios-dir to ignore",
    )
    parser.add_argument(
        "--no-skip-dir",
        action="store_true",
        help="Disable skipping a reserved folder under scenarios-dir",
    )
    parser.add_argument(
        "--include-substring",
        action="append",
        default=[],
        help=(
            "Build only scenario IDs containing this text; can be repeated and "
            "matches any supplied value"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild datasets even when the output CSV already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be built without writing datasets",
    )
    parser.add_argument(
        "--fast-target-ms",
        type=float,
        default=0.25,
        help="Full-reward service-time threshold for fast_latency datasets",
    )
    parser.add_argument(
        "--fast-ok-ms",
        type=float,
        default=0.50,
        help="Partial-reward service-time threshold for fast_latency datasets",
    )
    parser.add_argument(
        "--robust-cap-quantile",
        type=float,
        default=0.99,
        help="Cap service_ms at this quantile for robust_delay datasets",
    )
    parser.add_argument(
        "--robust-scale-ms",
        type=float,
        default=0.25,
        help="Scale term in -log1p(service_ms/scale) for robust_delay datasets",
    )
    parser.add_argument(
        "--robust-loss-penalty-ms",
        type=float,
        default=5.0,
        help="Extra cost added past the cap for undelivered packets in robust_delay mode",
    )
    parser.add_argument(
        "--utility-payload-bytes",
        type=int,
        default=128,
        help="Payload bytes used to compute goodput for utility datasets",
    )
    parser.add_argument(
        "--utility-goodput-quantile",
        type=float,
        default=0.95,
        help="Delivered log-goodput quantile mapped to reward +1 in utility datasets",
    )
    parser.add_argument(
        "--utility-loss-reward",
        type=float,
        default=-1.0,
        help="Reward assigned to undelivered packets in utility datasets before clipping",
    )
    parser.add_argument(
        "--utility-tail-target-ms",
        type=float,
        default=0.0,
        help="Optional tail-delay target for utility datasets; 0 disables the tail penalty",
    )
    parser.add_argument(
        "--utility-tail-weight",
        type=float,
        default=0.0,
        help="Optional tail-delay penalty weight for utility datasets",
    )
    parser.add_argument(
        "--state-alignment",
        choices=["same_packet", "previous_csi"],
        default="same_packet",
        help=(
            "State/action alignment passed to each dataset build; previous_csi "
            "uses only CSI available before the target packet"
        ),
    )
    parser.add_argument(
        "--stale-augment-ages",
        default=None,
        help=(
            "Comma-separated ages for synthetic no-new-CSI rows, e.g. '2,4,8,16'. "
            "Requires --state-alignment previous_csi."
        ),
    )
    parser.add_argument(
        "--stale-augment-source",
        choices=["loss_only", "loss_or_missing", "all"],
        default="loss_only",
        help="Rows to duplicate for synthetic stale-state augmentation",
    )
    parser.add_argument(
        "--merge-groups",
        action="store_true",
        help=(
            "After per-run datasets are available, write grouped merged datasets for "
            "LOS, NLOS, each distance, and each condition+distance"
        ),
    )
    parser.add_argument(
        "--merge-balance",
        choices=["none", "equal_rows"],
        default="none",
        help="Balance rows across source files when writing grouped merged datasets",
    )
    parser.add_argument(
        "--no-merge-shuffle",
        action="store_true",
        help="Disable final row shuffling for grouped merged datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for grouped merge balancing/shuffling",
    )

    args = parser.parse_args()

    return build_all(
        scenarios_dir=args.scenarios_dir,
        output_dir=args.output_dir,
        reward_mode=args.reward_mode,
        dedup=args.dedup,
        skip_dir_name=None if args.no_skip_dir else args.skip_dir_name,
        include_substrings=args.include_substring,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        fast_target_ms=args.fast_target_ms,
        fast_ok_ms=args.fast_ok_ms,
        robust_cap_quantile=args.robust_cap_quantile,
        robust_scale_ms=args.robust_scale_ms,
        robust_loss_penalty_ms=args.robust_loss_penalty_ms,
        utility_payload_bytes=args.utility_payload_bytes,
        utility_goodput_quantile=args.utility_goodput_quantile,
        utility_loss_reward=args.utility_loss_reward,
        utility_tail_target_ms=args.utility_tail_target_ms,
        utility_tail_weight=args.utility_tail_weight,
        state_alignment=args.state_alignment,
        stale_augment_ages=args.stale_augment_ages,
        stale_augment_source=args.stale_augment_source,
        merge_groups=args.merge_groups,
        merge_balance=args.merge_balance,
        merge_shuffle=not args.no_merge_shuffle,
        seed=args.seed,
        csi_feature_contract=args.csi_feature_contract,
        require_metadata=args.require_metadata,
    )


if __name__ == "__main__":
    raise SystemExit(main())
