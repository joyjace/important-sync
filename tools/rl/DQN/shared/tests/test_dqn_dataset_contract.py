"""Dataset-builder integration tests for the link-v7c CSI contract."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterable, Mapping


SHARED_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHARED_DIR))

import build_dqn_dataset as dataset_builder  # noqa: E402
import csi_link_v7c as link_v7c  # noqa: E402
import merge_dqn_datasets as dataset_merger  # noqa: E402


CSI_COLUMNS = (
    "seq",
    "rssi",
    "noise_floor",
    "fft_gain",
    "agc_gain",
    "channel",
    "sig_len",
    "data_json",
    "host_time",
    "format",
    "mac",
    "rate",
    "local_timestamp",
    "rx_state",
    "data_len",
    "first_word",
    "iq_pairs",
    "b64_version",
    "gain_compensation_exact",
)

ACK_COLUMNS = ("seq", "mcs_index", "delivered", "service_us")


def _write_csv(
    path: Path,
    columns: Iterable[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iq_frame(imaginary: int = 4, real: int = 3) -> list[int]:
    """Return one strong, deterministic 57-bin ``[imaginary, real]`` frame."""

    values = [component for _ in range(57) for component in (imaginary, real)]
    values[56:58] = [0, 0]  # Raw bin 28 is DC.
    return values


def _csi_row(seq: int, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seq": seq,
        "rssi": -48,
        "noise_floor": -96,
        "fft_gain": 2,
        "agc_gain": 3,
        "channel": 6,
        "sig_len": 128,
        "data_json": json.dumps(_iq_frame()),
        "host_time": f"2026-01-02T03:04:{seq:02d}.000000+00:00",
        "format": "c5c6",
        "mac": "aa:bb:cc:dd:ee:01",
        "rate": 11,
        "local_timestamp": 1000 + seq,
        "rx_state": 0,
        "data_len": 114,
        "first_word": 0,
        "iq_pairs": 57,
        "b64_version": "",
        "gain_compensation_exact": "",
    }
    row.update(overrides)
    return row


def _ack_row(
    seq: int,
    *,
    mcs_index: int = 2,
    delivered: int = 1,
    service_us: int = 250,
) -> dict[str, int]:
    return {
        "seq": seq,
        "mcs_index": mcs_index,
        "delivered": delivered,
        "service_us": service_us,
    }


class TestDqnDatasetFeatureContract(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="dqn_dataset_contract_test_"
        )
        self.directory = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _build(
        self,
        csi_rows: list[dict[str, Any]],
        ack_rows: list[dict[str, Any]],
        *,
        output_name: str = "dataset.csv",
        csi_columns: Iterable[str] = CSI_COLUMNS,
        **kwargs: Any,
    ) -> tuple[Path, Path, Path]:
        csi_csv = self.directory / f"{Path(output_name).stem}_csi.csv"
        ack_csv = self.directory / f"{Path(output_name).stem}_ack.csv"
        output_csv = self.directory / output_name
        _write_csv(csi_csv, csi_columns, csi_rows)
        _write_csv(ack_csv, ACK_COLUMNS, ack_rows)
        with contextlib.redirect_stdout(io.StringIO()):
            dataset_builder.build_dataset(
                csi_csv=csi_csv,
                ack_csv=ack_csv,
                output_csv=output_csv,
                **kwargs,
            )
        return csi_csv, ack_csv, output_csv

    def test_v7_same_packet_writes_features_provenance_and_sidecar(self) -> None:
        metadata_json = self.directory / "metadata.json"
        metadata = {
            "scenario_id": "contract_fixture",
            "distance_m": 1.25,
        }
        # Capture tooling writes singleton arrays today; this must remain a
        # supported provenance representation rather than being silently lost.
        metadata_json.write_text(json.dumps([metadata]), encoding="utf-8")

        csi_rows = [
            _csi_row(1),
            _csi_row(
                2,
                format="b64",
                b64_version=2,
                gain_compensation_exact=1,
                mac="aa:bb:cc:dd:ee:02",
                local_timestamp=2002,
                rx_state=1,
            ),
        ]
        ack_rows = [_ack_row(1), _ack_row(2, mcs_index=4, service_us=375)]
        csi_csv, ack_csv, output_csv = self._build(
            csi_rows,
            ack_rows,
            csi_feature_contract=link_v7c.CONTRACT_ID,
            metadata_json=metadata_json,
            state_alignment="same_packet",
        )

        rows = _read_csv(output_csv)
        self.assertEqual(len(rows), 2)
        for row in rows:
            amplitudes = json.loads(row["iq_active_amplitudes"])
            legacy_amplitudes = json.loads(row["iq_raw"])
            phase_real = json.loads(row["iq_phase_diff_real"])
            phase_imag = json.loads(row["iq_phase_diff_imag"])
            self.assertEqual(len(amplitudes), 56)
            self.assertEqual(len(legacy_amplitudes), 57)
            self.assertEqual(legacy_amplitudes[28], 0.0)
            self.assertEqual(legacy_amplitudes[:28], [5.0] * 28)
            self.assertEqual(legacy_amplitudes[29:], [5.0] * 28)
            self.assertEqual(len(phase_real), 54)
            self.assertEqual(len(phase_imag), 54)
            flat = amplitudes + phase_real + phase_imag + [
                float(row["iq_phase_valid_fraction"]),
                float(row["iq_phase_coherence"]),
            ]
            self.assertEqual(len(flat), 166)
            self.assertEqual(flat, link_v7c.extract_flat_features(_iq_frame()).tolist())
            self.assertGreaterEqual(float(row["iq_phase_valid_fraction"]), 0.0)
            self.assertLessEqual(float(row["iq_phase_valid_fraction"]), 1.0)
            self.assertGreaterEqual(float(row["iq_phase_coherence"]), 0.0)
            self.assertLessEqual(float(row["iq_phase_coherence"]), 1.0)
            self.assertIn(row["csi_format"], {"c5c6", "b64"})
            self.assertEqual(int(row["csi_data_len"]), 114)
            self.assertEqual(int(row["csi_first_word"]), 0)
            self.assertEqual(int(row["csi_iq_pairs"]), 57)
            self.assertEqual(row["meta_scenario_id"], metadata["scenario_id"])
            self.assertAlmostEqual(float(row["meta_distance_m"]), 1.25)

        sidecar_path = dataset_builder.feature_contract_sidecar_path(output_csv)
        self.assertTrue(sidecar_path.is_file())
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        self.assertEqual(sidecar["manifest_schema"], "dqn_csi_feature_contract/v1")
        self.assertEqual(sidecar["artifact"]["sha256"], _sha256(output_csv))
        self.assertEqual(sidecar["artifact"]["row_count"], 2)
        self.assertEqual(sidecar["sources"]["csi_csv"]["sha256"], _sha256(csi_csv))
        self.assertEqual(sidecar["sources"]["ack_csv"]["sha256"], _sha256(ack_csv))
        self.assertEqual(
            sidecar["sources"]["metadata_json"]["sha256"],
            _sha256(metadata_json),
        )
        self.assertEqual(sidecar["scenario_metadata"], metadata)
        self.assertEqual(sidecar["qualification"]["status"], "noncausal")
        self.assertFalse(sidecar["qualification"]["causal_alignment"])
        self.assertTrue(sidecar["qualification"]["gain_compensation_exact"])
        self.assertFalse(sidecar["qualification"]["deployment_candidate"])
        self.assertTrue(sidecar["qualification"]["blocking_reasons"])
        self.assertEqual(
            sidecar["producer"]["transform"]["sha256"],
            _sha256(Path(link_v7c.__file__)),
        )
        self.assertEqual(
            sidecar["build"]["reward"],
            {"mode": "delay", "parameters": {"formula": "-service_ms"}, "derived": {}},
        )

        contract = sidecar["feature_contract"]
        self.assertEqual(contract["feature_contract_id"], link_v7c.CONTRACT_ID)
        self.assertEqual(contract["feature_count"], 166)
        self.assertEqual(len(contract["feature_names"]), 166)
        self.assertEqual(contract["feature_contract_sha256"], link_v7c.CONTRACT_SHA256)
        canonical_contract = json.dumps(
            contract["contract"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        self.assertEqual(canonical_contract, link_v7c.CONTRACT_CANONICAL_JSON)
        self.assertEqual(
            hashlib.sha256(canonical_contract.encode("utf-8")).hexdigest(),
            link_v7c.CONTRACT_SHA256,
        )
        self.assertEqual(sidecar["capture_provenance"]["data_len"], {"114": 2})
        self.assertEqual(sidecar["capture_provenance"]["first_word"], {"0": 2})
        self.assertEqual(sidecar["capture_provenance"]["iq_pairs"], {"57": 2})
        self.assertEqual(
            sidecar["capture_provenance"]["format"],
            {"b64": 1, "c5c6": 1},
        )
        self.assertEqual(
            sidecar["capture_provenance"]["b64_version"],
            {"2.0": 1, "<missing>": 1},
        )
        self.assertEqual(
            sidecar["capture_provenance"]["gain_compensation_exact"],
            {"1.0": 1, "<missing>": 1},
        )

    def test_previous_csi_carries_source_state_provenance(self) -> None:
        csi_rows = [
            _csi_row(
                10,
                mac="aa:bb:cc:dd:ee:10",
                local_timestamp=1010,
                rx_state=10,
            ),
            _csi_row(
                12,
                mac="aa:bb:cc:dd:ee:12",
                local_timestamp=1212,
                rx_state=12,
            ),
        ]
        # Make the newer state observably different from the earlier state.
        csi_rows[1]["data_json"] = json.dumps(_iq_frame(imaginary=5, real=12))
        ack_rows = [_ack_row(10), _ack_row(11), _ack_row(12), _ack_row(13)]
        _, _, output_csv = self._build(
            csi_rows,
            ack_rows,
            csi_feature_contract=link_v7c.CONTRACT_ID,
            state_alignment="previous_csi",
        )

        rows_by_seq = {int(row["seq"]): row for row in _read_csv(output_csv)}
        self.assertEqual(set(rows_by_seq), {11, 12, 13})

        # Target 12 has its own CSI, but a live decision can only use CSI 10.
        target_12 = rows_by_seq[12]
        self.assertEqual(int(target_12["state_seq"]), 10)
        self.assertEqual(target_12["csi_mac"], "aa:bb:cc:dd:ee:10")
        self.assertEqual(int(float(target_12["csi_local_timestamp"])), 1010)
        self.assertEqual(int(float(target_12["csi_rx_state"])), 10)

        # The next decision is the first one allowed to consume CSI 12.
        target_13 = rows_by_seq[13]
        self.assertEqual(int(target_13["state_seq"]), 12)
        self.assertEqual(target_13["csi_mac"], "aa:bb:cc:dd:ee:12")
        self.assertEqual(int(float(target_13["csi_local_timestamp"])), 1212)
        self.assertEqual(int(float(target_13["csi_rx_state"])), 12)

        sidecar = json.loads(
            dataset_builder.feature_contract_sidecar_path(output_csv).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(sidecar["qualification"]["status"], "candidate")
        self.assertTrue(sidecar["qualification"]["causal_alignment"])
        self.assertTrue(sidecar["qualification"]["gain_compensation_exact"])
        self.assertTrue(sidecar["qualification"]["deployment_candidate"])
        self.assertEqual(sidecar["qualification"]["blocking_reasons"], [])

    def test_training_gate_verifies_artifact_and_deployment_qualification(self) -> None:
        _, _, output_csv = self._build(
            [_csi_row(1), _csi_row(2)],
            [_ack_row(1), _ack_row(2)],
            output_name="qualified_training.csv",
            csi_feature_contract=link_v7c.CONTRACT_ID,
            state_alignment="previous_csi",
        )
        sidecar_path = dataset_builder.feature_contract_sidecar_path(output_csv)
        original_csv = output_csv.read_bytes()
        original_sidecar = sidecar_path.read_text(encoding="utf-8")

        validated = link_v7c.validate_dataset_artifact(output_csv)
        self.assertTrue(validated["qualification"]["deployment_candidate"])

        output_csv.write_bytes(original_csv + b"\n")
        with self.assertRaisesRegex(ValueError, "Dataset artifact does not match"):
            link_v7c.validate_dataset_artifact(output_csv)
        output_csv.write_bytes(original_csv)

        for field, value in (
            ("row_count", 999),
            ("columns", ["wrong_column"]),
        ):
            with self.subTest(artifact_field=field):
                sidecar = json.loads(original_sidecar)
                sidecar["artifact"][field] = value
                sidecar_path.write_text(
                    json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "Dataset artifact does not match"):
                    link_v7c.validate_dataset_artifact(output_csv)

        for field, value in (
            ("input_scalar_count", 113),
            ("feature_names", ["wrong_feature"]),
        ):
            with self.subTest(contract_field=field):
                sidecar = json.loads(original_sidecar)
                sidecar["feature_contract"][field] = value
                sidecar_path.write_text(
                    json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "embedded feature contract"):
                    link_v7c.validate_dataset_artifact(output_csv)

        sidecar = json.loads(original_sidecar)
        sidecar["feature_contract"]["contract"]["input"]["scalar_count"] = 113
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "embedded feature contract"):
            link_v7c.validate_dataset_artifact(output_csv)

        sidecar = json.loads(original_sidecar)
        sidecar["qualification"]["deployment_candidate"] = False
        sidecar["qualification"]["status"] = "blocked"
        sidecar["qualification"]["blocking_reasons"] = ["test blocker"]
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "not qualified for deployment"):
            link_v7c.validate_dataset_artifact(output_csv)

    def test_b64_gain_exactness_blocks_only_deployment_qualification(self) -> None:
        cases = (
            ("legacy_v1", 1, 0, "blocked", False),
            ("v2_without_exact_marker", 2, "", "blocked", False),
            ("exact_v2", 2, 1, "candidate", True),
        )
        for name, version, exact_marker, expected_status, expected_candidate in cases:
            with self.subTest(case=name):
                csi_rows = [
                    _csi_row(
                        seq,
                        format="b64",
                        b64_version=version,
                        gain_compensation_exact=exact_marker,
                    )
                    for seq in (1, 2)
                ]
                _, _, output_csv = self._build(
                    csi_rows,
                    [_ack_row(1), _ack_row(2)],
                    output_name=f"gain_qualification_{name}.csv",
                    csi_feature_contract=link_v7c.CONTRACT_ID,
                    state_alignment="previous_csi",
                )

                # Historical B64 remains usable training input in every case.
                self.assertEqual(len(_read_csv(output_csv)), 1)
                sidecar = json.loads(
                    dataset_builder.feature_contract_sidecar_path(output_csv).read_text(
                        encoding="utf-8"
                    )
                )
                qualification = sidecar["qualification"]
                self.assertEqual(qualification["status"], expected_status)
                self.assertEqual(
                    qualification["deployment_candidate"], expected_candidate
                )
                self.assertEqual(
                    qualification["gain_compensation_exact"], expected_candidate
                )
                self.assertEqual(
                    bool(qualification["blocking_reasons"]), not expected_candidate
                )

    def test_v7_rejects_invalid_capture_provenance_and_aggregate_dedup(self) -> None:
        invalid_cases = (
            (
                "wrong data length",
                [_csi_row(1, data_len=113)],
                {"dedup_strategy": "keep_first"},
                "data_len=113",
            ),
            (
                "invalid first word",
                [_csi_row(1, first_word=1)],
                {"dedup_strategy": "keep_first"},
                "first_word=1",
            ),
            (
                "unknown capture format",
                [_csi_row(1, format="unversioned")],
                {"dedup_strategy": "keep_first"},
                "capture format 'unversioned'",
            ),
            (
                "non C5 legacy transport",
                [_csi_row(1, format="legacy")],
                {"dedup_strategy": "keep_first"},
                "capture format 'legacy'",
            ),
            (
                "aggregate dedup",
                [_csi_row(1), _csi_row(1, host_time="2026-01-02T04:00:00+00:00")],
                {"dedup_strategy": "aggregate"},
                "does not support aggregate",
            ),
        )
        for name, csi_rows, kwargs, message in invalid_cases:
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                self._build(
                    csi_rows,
                    [_ack_row(1)],
                    output_name=f"invalid_{name.replace(' ', '_')}.csv",
                    csi_feature_contract=link_v7c.CONTRACT_ID,
                    **kwargs,
                )

    def test_v7_quarantines_isolated_invalid_capture_before_causal_alignment(self) -> None:
        csi_rows = [
            _csi_row(10),
            _csi_row(11, format="legacy"),
            _csi_row(12, data_json=json.dumps(_iq_frame(imaginary=5, real=12))),
        ]
        _, _, output_csv = self._build(
            csi_rows,
            [_ack_row(10), _ack_row(11), _ack_row(12), _ack_row(13)],
            output_name="quarantined_capture.csv",
            csi_feature_contract=link_v7c.CONTRACT_ID,
            state_alignment="previous_csi",
        )

        rows_by_seq = {int(row["seq"]): row for row in _read_csv(output_csv)}
        self.assertEqual(set(rows_by_seq), {11, 12, 13})
        self.assertEqual(int(rows_by_seq[12]["state_seq"]), 10)
        self.assertEqual(int(rows_by_seq[13]["state_seq"]), 12)

        sidecar = json.loads(
            dataset_builder.feature_contract_sidecar_path(output_csv).read_text(
                encoding="utf-8"
            )
        )
        source_filter = sidecar["capture_validation"]["source_filter"]
        self.assertEqual(source_filter["source_row_count"], 3)
        self.assertEqual(source_filter["retained_row_count"], 2)
        self.assertEqual(source_filter["rejected_row_count"], 1)
        self.assertEqual(sidecar["capture_provenance"]["format"], {"c5c6": 2})
        self.assertEqual(
            sidecar["source_capture_provenance"]["format"],
            {"c5c6": 2, "legacy": 1},
        )
        self.assertTrue(sidecar["qualification"]["deployment_candidate"])

    def test_legacy_default_does_not_require_provenance_or_sidecar(self) -> None:
        legacy_columns = (
            "seq",
            "rssi",
            "noise_floor",
            "fft_gain",
            "agc_gain",
            "channel",
            "sig_len",
            "data_json",
        )
        legacy_row = {
            "seq": 7,
            "rssi": -60,
            "noise_floor": -98,
            "fft_gain": 1,
            "agc_gain": 2,
            "channel": 6,
            "sig_len": 128,
            "data_json": json.dumps([3, 4] * 117),
        }
        _, _, output_csv = self._build(
            [legacy_row],
            [_ack_row(7)],
            output_name="legacy.csv",
            csi_columns=legacy_columns,
        )

        rows = _read_csv(output_csv)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(json.loads(rows[0]["iq_raw"])), 117)
        self.assertNotIn("iq_active_amplitudes", rows[0])
        self.assertFalse(
            dataset_builder.feature_contract_sidecar_path(output_csv).exists()
        )

    def test_merge_verifier_rejects_tampered_tracked_dataset(self) -> None:
        _, _, output_csv = self._build(
            [_csi_row(1)],
            [_ack_row(1)],
            output_name="tracked.csv",
            csi_feature_contract=link_v7c.CONTRACT_ID,
        )
        with output_csv.open("a", encoding="utf-8") as stream:
            stream.write("\n")

        with self.assertRaisesRegex(ValueError, "hash does not match sidecar"):
            dataset_merger.load_verified_contract_lineage([output_csv])

    def test_merge_cannot_promote_causal_inexact_b64_parent(self) -> None:
        parent_specs = (
            ("a", 10, {}),
            (
                "b",
                20,
                {
                    "format": "b64",
                    "b64_version": 1,
                    "gain_compensation_exact": 0,
                },
            ),
        )
        for suffix, first_seq, capture_overrides in parent_specs:
            self._build(
                [
                    _csi_row(first_seq, **capture_overrides),
                    _csi_row(first_seq + 1, **capture_overrides),
                ],
                [_ack_row(first_seq), _ack_row(first_seq + 1)],
                output_name=f"causal_gain_parent_{suffix}.csv",
                csi_feature_contract=link_v7c.CONTRACT_ID,
                state_alignment="previous_csi",
            )

        merged_csv = self.directory / "causal_gain_merged.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            dataset_merger.merge_datasets(
                input_glob=str(self.directory / "causal_gain_parent_[ab].csv"),
                output_csv=merged_csv,
                balance="none",
                shuffle=False,
                seed=42,
            )

        sidecar = json.loads(
            dataset_builder.feature_contract_sidecar_path(merged_csv).read_text(
                encoding="utf-8"
            )
        )
        qualification = sidecar["qualification"]
        self.assertTrue(qualification["causal_alignment"])
        self.assertFalse(qualification["gain_compensation_exact"])
        self.assertFalse(qualification["deployment_candidate"])
        self.assertEqual(qualification["status"], "blocked")
        self.assertTrue(qualification["blocking_reasons"])

    def test_merge_preserves_lineage_and_rejects_contract_mismatch(self) -> None:
        self.assertEqual(
            dataset_merger.scenario_name_from_path(
                Path(
                    "rl_dqn_dataset_LOS_1m_F202_run02_"
                    "link_v7c_ht20_v1_utility.csv"
                ),
                link_v7c.CONTRACT_ID,
            ),
            "LOS_1m_F202_run02",
        )
        parent_paths = []
        for suffix, seq in (("a", 1), ("b", 2)):
            _, _, output_csv = self._build(
                [_csi_row(seq)],
                [_ack_row(seq)],
                output_name=f"contract_parent_{suffix}.csv",
                csi_feature_contract=link_v7c.CONTRACT_ID,
            )
            parent_paths.append(output_csv)

        merged_csv = self.directory / "merged.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            dataset_merger.merge_datasets(
                input_glob=str(self.directory / "contract_parent_[ab].csv"),
                output_csv=merged_csv,
                balance="none",
                shuffle=False,
                seed=42,
            )
        merged_sidecar_path = dataset_builder.feature_contract_sidecar_path(merged_csv)
        merged_sidecar = json.loads(merged_sidecar_path.read_text(encoding="utf-8"))
        self.assertEqual(merged_sidecar["artifact"]["sha256"], _sha256(merged_csv))
        self.assertEqual(merged_sidecar["artifact"]["row_count"], 2)
        self.assertEqual(len(merged_sidecar["parents"]), 2)
        self.assertEqual(merged_sidecar["qualification"]["status"], "noncausal")
        self.assertFalse(merged_sidecar["qualification"]["causal_alignment"])
        self.assertTrue(merged_sidecar["qualification"]["gain_compensation_exact"])
        self.assertFalse(merged_sidecar["qualification"]["deployment_candidate"])
        self.assertEqual(
            merged_sidecar["feature_contract"]["feature_contract_id"],
            link_v7c.CONTRACT_ID,
        )

        second_sidecar_path = dataset_builder.feature_contract_sidecar_path(parent_paths[1])
        second_sidecar = json.loads(second_sidecar_path.read_text(encoding="utf-8"))
        second_contract = second_sidecar["feature_contract"]
        second_contract["feature_contract_id"] = "deliberately_incompatible_v1"
        second_contract["contract"]["schema_id"] = "deliberately_incompatible_v1"
        canonical = json.dumps(
            second_contract["contract"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        second_contract["feature_contract_sha256"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        second_sidecar_path.write_text(
            json.dumps(second_sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Feature-contract mismatch"):
            dataset_merger.load_verified_contract_lineage(parent_paths)


if __name__ == "__main__":
    unittest.main()
