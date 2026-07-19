#!/usr/bin/env python3
"""
Merge multiple DQN dataset CSV files into one training dataset.

Typical usage:
    python merge_dqn_datasets.py \
        --input-glob "datasets/rl_dqn_dataset_*_*_delay.csv" \
        --output datasets/rl_dqn_dataset_all_scenarios_delay.csv

Balanced merge (same row count per scenario file):
    python merge_dqn_datasets.py \
        --input-glob "datasets/rl_dqn_dataset_*_*_delay.csv" \
        --output datasets/rl_dqn_dataset_all_scenarios_delay_balanced.csv \
        --balance equal_rows
"""

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd

try:
    from .build_dqn_dataset import (
        combine_parent_qualifications,
        feature_contract_sidecar_path,
        sha256_file,
        validate_feature_contract_record,
    )
except ImportError:
    from build_dqn_dataset import (
        combine_parent_qualifications,
        feature_contract_sidecar_path,
        sha256_file,
        validate_feature_contract_record,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
DQN_ROOT = SCRIPT_DIR.parent
REQUIRED_COLUMNS = {
    "seq",
    "mcs_index",
    "delivered",
    "service_ms",
    "reward",
    "rssi",
    "snr",
    "fft_gain",
    "agc_gain",
    "channel",
    "sig_len",
    "iq_mean",
    "iq_std",
    "iq_p10",
    "iq_p50",
    "iq_p90",
    "iq_raw",
}


def scenario_name_from_path(path_obj: Path, feature_contract_id: str | None = None) -> str:
    """Extract scenario-ish name from known filename format."""
    name = path_obj.stem
    prefix = "rl_dqn_dataset_"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    for suffix in ("_fast_latency", "_robust_delay", "_log_delay", "_utility", "_delay", "_pdr"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if feature_contract_id:
        contract_suffix = f"_{feature_contract_id}"
        if name.endswith(contract_suffix):
            name = name[: -len(contract_suffix)]
    return name


def distance_label_from_scenario(scenario: str) -> str | None:
    """Extract labels such as 1m or 2p5m from a scenario name."""
    match = re.search(r"(?:^|_)(\d+(?:p\d+)?)m(?:_|$)", scenario)
    return f"{match.group(1)}m" if match else None


def load_and_tag_csv(
    path_obj: Path,
    feature_contract_id: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path_obj)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path_obj}: missing required columns: {missing}")

    df = df.copy()
    df["source_file"] = path_obj.name
    scenario = scenario_name_from_path(path_obj, feature_contract_id)
    df["source_scenario"] = scenario
    df["source_distance"] = distance_label_from_scenario(scenario) or "unknown"
    return df


def load_verified_contract_lineage(paths: list[Path]):
    """Validate adjacent sidecars; wholly manifest-free legacy input is allowed."""
    sidecar_paths = [feature_contract_sidecar_path(path) for path in paths]
    present = [sidecar.exists() for sidecar in sidecar_paths]
    if not any(present):
        return None, []
    if not all(present):
        missing = [str(path) for path, exists in zip(paths, present) if not exists]
        raise ValueError(
            "Refusing to mix contract-tracked and untracked datasets; missing sidecars: "
            + ", ".join(missing)
        )

    common_contract = None
    lineage = []
    for path, sidecar_path in zip(paths, sidecar_paths):
        with sidecar_path.open("r", encoding="utf-8") as stream:
            sidecar = json.load(stream)
        actual_hash = sha256_file(path)
        if sidecar.get("artifact", {}).get("sha256") != actual_hash:
            raise ValueError(f"Dataset hash does not match sidecar: {path}")
        contract = sidecar.get("feature_contract")
        validate_feature_contract_record(contract)
        if common_contract is None:
            common_contract = contract
        elif contract != common_contract:
            raise ValueError(f"Feature-contract mismatch at {path}")
        lineage.append(
            {
                "artifact_path": str(path.resolve()),
                "artifact_sha256": actual_hash,
                "manifest_path": str(sidecar_path.resolve()),
                "manifest_sha256": sha256_file(sidecar_path),
                "qualification": sidecar.get("qualification", {}),
            }
        )
    return common_contract, lineage


def balance_equal_distance_rows(frames: list[pd.DataFrame], seed: int) -> list[pd.DataFrame]:
    """
    Give every distance the same total rows and balance scenarios inside it.

    The common distance budget is limited by the distance whose balanced
    scenario files provide the fewest rows.
    """
    frames_by_distance: dict[str, list[pd.DataFrame]] = {}
    for frame in frames:
        distance = str(frame["source_distance"].iloc[0])
        if distance == "unknown":
            scenario = str(frame["source_scenario"].iloc[0])
            raise ValueError(
                f"Cannot use equal_distance_rows: distance is unknown for {scenario}"
            )
        frames_by_distance.setdefault(distance, []).append(frame)

    balanced_capacities = {
        distance: len(group) * min(len(frame) for frame in group)
        for distance, group in frames_by_distance.items()
    }
    rows_per_distance = min(balanced_capacities.values())
    print(
        "  Balancing mode: equal_distance_rows "
        f"(sampling {rows_per_distance} total rows per distance)"
    )

    balanced_frames = []
    for distance in sorted(frames_by_distance):
        group = frames_by_distance[distance]
        rows_per_file, remainder = divmod(rows_per_distance, len(group))
        file_budget = (
            str(rows_per_file)
            if remainder == 0
            else f"{rows_per_file} or {rows_per_file + 1}"
        )
        print(
            f"    {distance}: {len(group)} scenario file(s), "
            f"{file_budget} rows per file"
        )
        for index, frame in enumerate(group):
            sample_rows = rows_per_file + (1 if index < remainder else 0)
            balanced_frames.append(
                frame.sample(
                    n=sample_rows,
                    random_state=seed + index,
                ).reset_index(drop=True)
            )
    return balanced_frames


def merge_datasets(
    input_glob: str,
    output_csv: Path,
    balance: str,
    shuffle: bool,
    seed: int,
    exclude_substrings: list[str] | None = None,
) -> None:
    paths = sorted(Path(p) for p in glob.glob(input_glob))
    exclude_substrings = exclude_substrings or []
    if exclude_substrings:
        paths = [
            path
            for path in paths
            if not any(exclude in path.name for exclude in exclude_substrings)
        ]
    if not paths:
        raise SystemExit(f"No files matched: {input_glob}")

    print("[DQN Dataset Merger]")
    print(f"  Pattern: {input_glob}")
    if exclude_substrings:
        print(f"  Excluding files containing: {', '.join(exclude_substrings)}")
    print(f"  Matched files: {len(paths)}")
    for p in paths:
        print(f"    - {p}")

    common_contract, lineage = load_verified_contract_lineage(paths)
    feature_contract_id = (
        common_contract.get("feature_contract_id")
        if common_contract is not None
        else None
    )
    frames = [load_and_tag_csv(p, feature_contract_id) for p in paths]
    original_rows = [len(frame) for frame in frames]

    if balance == "equal_rows":
        min_rows = min(len(df) for df in frames)
        print(f"  Balancing mode: equal_rows (sampling {min_rows} from each file)")
        frames = [df.sample(n=min_rows, random_state=seed).reset_index(drop=True) for df in frames]
    elif balance == "equal_distance_rows":
        frames = balance_equal_distance_rows(frames, seed)
    else:
        print("  Balancing mode: none")

    merged = pd.concat(frames, ignore_index=True)

    if shuffle:
        merged = merged.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    if common_contract is not None:
        selected_rows_by_file = {
            str(frame["source_file"].iloc[0]): len(frame)
            for frame in frames
            if not frame.empty
        }
        for parent, path, original_count in zip(lineage, paths, original_rows):
            parent["original_row_count"] = int(original_count)
            parent["selected_row_count"] = int(selected_rows_by_file.get(path.name, 0))
        merged_qualification = combine_parent_qualifications(lineage)
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
                "exclude_substrings": list(exclude_substrings),
            },
            "parents": lineage,
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

    print(f"  Output: {output_csv}")
    print(f"  Total rows: {len(merged)}")
    print("  Rows per distance:")
    for distance, cnt in merged["source_distance"].value_counts().sort_index().items():
        print(f"    {distance}: {cnt}")
    print("  Rows per source file:")
    for fname, cnt in merged["source_file"].value_counts().sort_index().items():
        print(f"    {fname}: {cnt}")

    print("  Mean service_ms by source file:")
    service_stats = merged.groupby("source_file")["service_ms"].mean().sort_index()
    for fname, val in service_stats.items():
        print(f"    {fname}: {val:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge DQN scenario datasets")
    parser.add_argument(
        "--input-glob",
        default=str(DQN_ROOT / "datasets" / "rl_dqn_dataset_*_*_delay.csv"),
        help="Glob for input CSV files",
    )
    parser.add_argument(
        "--output",
        default=str(DQN_ROOT / "datasets" / "rl_dqn_dataset_all_scenarios_delay.csv"),
        help="Output merged CSV path",
    )
    parser.add_argument(
        "--balance",
        choices=["none", "equal_rows", "equal_distance_rows"],
        default="none",
        help=(
            "Balance mode: equal_rows balances every file; "
            "equal_distance_rows balances total rows per distance and scenarios within it"
        ),
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable final row shuffling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling/shuffling",
    )
    parser.add_argument(
        "--exclude-substring",
        action="append",
        default=[],
        help="Exclude matched files whose filename contains this text; can be repeated",
    )

    args = parser.parse_args()
    merge_datasets(
        input_glob=args.input_glob,
        output_csv=Path(args.output),
        balance=args.balance,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        exclude_substrings=args.exclude_substring,
    )
