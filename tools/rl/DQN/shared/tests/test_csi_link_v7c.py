"""Golden and Python/C parity tests for the link-v7c HT20 CSI contract."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SHARED_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
CSI_C_DIR = REPOSITORY_ROOT / "csi_recv" / "main"
sys.path.insert(0, str(SHARED_DIR))

import csi_link_v7c as link_v7c  # noqa: E402


# First receiver frame captured in LOS_1m_F202_run02/csi_data.csv.  Keeping the
# raw scalars here makes the parity test independent of a large scenario tree.
REAL_CAPTURED_FRAME = (
    -10, 7, -14, 7, -14, 6, -15, 4, -17, 4, -17, 2, -18, 1, -19, 1,
    -21, 1, -22, 0, -22, -2, -22, -2, -23, -3, -24, -3, -24, -5, -26, -5,
    -27, -6, -28, -6, -29, -5, -29, -7, -30, -7, -31, -9, -31, -9, -32, -11,
    -33, -10, -32, -10, -35, -10, -34, -9, 0, 0, -35, -10, -33, -9, -34, -11,
    -35, -10, -33, -11, -32, -10, -33, -10, -32, -9, -32, -8, -30, -9,
    -30, -9, -30, -10, -28, -9, -28, -10, -29, -8, -28, -8, -28, -9,
    -27, -9, -26, -10, -26, -12, -25, -12, -24, -14, -23, -15, -22, -16,
    -21, -19, -18, -19, -16, -20, -12, -22,
)


def _constant_frame(imaginary: int = 4, real: int = 3) -> tuple[int, ...]:
    values: list[int] = []
    for _ in range(link_v7c.COMPLEX_BIN_COUNT):
        values.extend((imaginary, real))
    # DC must not contribute, even when it differs from every active bin.
    values[2 * link_v7c.DC_RAW_BIN_INDEX : 2 * link_v7c.DC_RAW_BIN_INDEX + 2] = (
        0,
        0,
    )
    return tuple(values)


def _synthetic_frame() -> tuple[int, ...]:
    """Create a deterministic, strong-signal frame with nonuniform phase."""

    values: list[int] = []
    for raw_bin in range(link_v7c.COMPLEX_BIN_COUNT):
        amplitude = 20 + raw_bin % 7
        angle = 0.31 * raw_bin + 0.07 * ((raw_bin * raw_bin) % 5)
        real = round(amplitude * math.cos(angle))
        imaginary = round(amplitude * math.sin(angle))
        values.extend((imaginary, real))
    values[2 * link_v7c.DC_RAW_BIN_INDEX : 2 * link_v7c.DC_RAW_BIN_INDEX + 2] = (
        0,
        0,
    )
    return tuple(values)


def _deep_fade_frame() -> tuple[int, ...]:
    values = list(_constant_frame())
    # Both bins are below the absolute phase threshold.  Each invalidates the
    # two adjacent pairs on its own side of DC.
    values[2 * 10 : 2 * 10 + 2] = (0, 1)
    values[2 * 30 : 2 * 30 + 2] = (0, 0)
    return tuple(values)


def _adaptive_threshold_boundary_frame() -> tuple[int, ...]:
    """Pin equality at the adaptive floor and rejection immediately below it."""

    values: list[int] = []
    for _ in range(link_v7c.COMPLEX_BIN_COUNT):
        values.extend((0, 100))
    values[2 * link_v7c.DC_RAW_BIN_INDEX : 2 * link_v7c.DC_RAW_BIN_INDEX + 2] = (
        0,
        0,
    )
    # The active median remains 100, hence the adaptive threshold is exactly
    # 10. Bin 10 must remain valid at equality; bin 11 must be rejected at 9.
    values[2 * 10 : 2 * 10 + 2] = (0, 10)
    values[2 * 11 : 2 * 11 + 2] = (0, 9)
    return tuple(values)


class TestLinkV7cManifest(unittest.TestCase):
    def test_contract_id_digest_and_canonical_json_are_stable(self) -> None:
        self.assertEqual(link_v7c.CONTRACT_ID, "link_v7c_ht20_v1")
        self.assertEqual(link_v7c.SCHEMA_ID, "link_v7c_ht20_v1")
        self.assertEqual(
            link_v7c.CONTRACT_SHA256,
            "df4f262b3fdf57f2f693b40b8584c08d5193ba092290d7f771e1a52575c8603a",
        )
        self.assertEqual(
            hashlib.sha256(link_v7c.CONTRACT_CANONICAL_JSON.encode("utf-8")).hexdigest(),
            link_v7c.CONTRACT_SHA256,
        )
        manifest = link_v7c.contract_manifest_dict()
        self.assertEqual(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            link_v7c.CONTRACT_CANONICAL_JSON,
        )

    def test_manifest_is_immutable_and_copies_do_not_leak_mutation(self) -> None:
        with self.assertRaises(TypeError):
            link_v7c.CONTRACT_MANIFEST["schema_id"] = "changed"
        with self.assertRaises(TypeError):
            link_v7c.CONTRACT_MANIFEST["input"]["scalar_count"] = 0

        mutable_copy = link_v7c.contract_manifest_dict()
        mutable_copy["input"]["scalar_count"] = 0
        self.assertEqual(
            link_v7c.CONTRACT_MANIFEST["input"]["scalar_count"],
            link_v7c.INPUT_SCALAR_COUNT,
        )

    def test_feature_names_pin_shape_and_flattened_order(self) -> None:
        names = link_v7c.FEATURE_NAMES
        self.assertEqual(link_v7c.FEATURE_COUNT, 166)
        self.assertEqual(len(names), 166)
        self.assertEqual(len(set(names)), 166)
        self.assertEqual(names[0], "iq_amp_raw_bin_00")
        self.assertEqual(names[27], "iq_amp_raw_bin_27")
        self.assertEqual(names[28], "iq_amp_raw_bin_29")
        self.assertEqual(names[55], "iq_amp_raw_bin_56")
        self.assertEqual(names[56], "iq_phase_diff_real_raw_bins_00_01")
        self.assertEqual(names[82], "iq_phase_diff_real_raw_bins_26_27")
        self.assertEqual(names[83], "iq_phase_diff_real_raw_bins_29_30")
        self.assertEqual(names[109], "iq_phase_diff_real_raw_bins_55_56")
        self.assertEqual(names[110], "iq_phase_diff_imag_raw_bins_00_01")
        self.assertEqual(names[163], "iq_phase_diff_imag_raw_bins_55_56")
        self.assertEqual(names[164:], link_v7c.QUALITY_FEATURE_NAMES)
        self.assertEqual(link_v7c.DIFFERENTIAL_PHASE_FEATURE_START, 56)
        self.assertEqual(link_v7c.DIFFERENTIAL_PHASE_FEATURE_STOP, 164)
        self.assertEqual(link_v7c.PHASE_DERIVED_FEATURE_START, 56)
        self.assertEqual(link_v7c.PHASE_DERIVED_FEATURE_STOP, 166)


class TestLinkV7cPython(unittest.TestCase):
    def test_analytic_constant_frame_pins_exact_shape_and_sections(self) -> None:
        features = link_v7c.extract_features(_constant_frame())
        flat = features.flatten()

        self.assertEqual(features.amplitudes.shape, (56,))
        self.assertEqual(features.phase_real.shape, (54,))
        self.assertEqual(features.phase_imag.shape, (54,))
        self.assertEqual(flat.shape, (166,))
        self.assertEqual(flat.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(flat[:56], np.full(56, 5.0, dtype=np.float32))
        np.testing.assert_array_equal(flat[56:110], np.ones(54, dtype=np.float32))
        np.testing.assert_array_equal(flat[110:164], np.zeros(54, dtype=np.float32))
        np.testing.assert_array_equal(flat[164:], np.ones(2, dtype=np.float32))

    def test_all_zero_frame_is_a_valid_all_zero_feature_vector(self) -> None:
        flat = link_v7c.extract_flat_features([0] * link_v7c.INPUT_SCALAR_COUNT)
        np.testing.assert_array_equal(flat, np.zeros(166, dtype=np.float32))

    def test_deep_fades_mask_only_adjacent_phase_pairs(self) -> None:
        features = link_v7c.extract_features(_deep_fade_frame())
        invalid_pair_indices = (9, 10, 27, 28)
        valid_pair_indices = tuple(
            index for index in range(54) if index not in invalid_pair_indices
        )

        np.testing.assert_array_equal(
            features.phase_real[list(invalid_pair_indices)],
            np.zeros(4, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            features.phase_imag[list(invalid_pair_indices)],
            np.zeros(4, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            features.phase_real[list(valid_pair_indices)],
            np.ones(50, dtype=np.float32),
        )
        self.assertAlmostEqual(float(features.valid_fraction), 50.0 / 54.0, places=7)
        self.assertEqual(float(features.coherence), 1.0)

    def test_adaptive_threshold_is_inclusive_at_exact_boundary(self) -> None:
        features = link_v7c.extract_features(_adaptive_threshold_boundary_frame())

        # Pair 9 (bins 9-10) includes the endpoint at exactly 10 and remains
        # valid. Pairs 10 and 11 touch bin 11 at 9 and are invalid.
        self.assertEqual(float(features.phase_real[9]), 1.0)
        np.testing.assert_array_equal(
            features.phase_real[[10, 11]], np.zeros(2, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            features.phase_imag[[10, 11]], np.zeros(2, dtype=np.float32)
        )
        self.assertAlmostEqual(float(features.valid_fraction), 52.0 / 54.0, places=7)
        self.assertEqual(float(features.coherence), 1.0)

    def test_dc_is_excluded_from_every_feature(self) -> None:
        baseline = list(_synthetic_frame())
        changed = baseline.copy()
        changed[2 * link_v7c.DC_RAW_BIN_INDEX] = 32767
        changed[2 * link_v7c.DC_RAW_BIN_INDEX + 1] = -32768
        np.testing.assert_array_equal(
            link_v7c.extract_flat_features(baseline),
            link_v7c.extract_flat_features(changed),
        )

    def test_common_phase_rotations_preserve_every_feature(self) -> None:
        baseline = link_v7c.extract_flat_features(REAL_CAPTURED_FRAME)
        sign_flipped = link_v7c.extract_flat_features([-value for value in REAL_CAPTURED_FRAME])
        np.testing.assert_array_equal(sign_flipped, baseline)

        # H' = jH is exact in the integer [imaginary, real] representation:
        # [imag', real'] = [real, -imag].
        quarter_turn: list[int] = []
        for imaginary, real in zip(REAL_CAPTURED_FRAME[0::2], REAL_CAPTURED_FRAME[1::2]):
            quarter_turn.extend((real, -imaginary))
        np.testing.assert_array_equal(
            link_v7c.extract_flat_features(quarter_turn),
            baseline,
        )

    def test_common_adjacent_phase_ramp_is_removed(self) -> None:
        baseline = link_v7c.extract_flat_features(REAL_CAPTURED_FRAME)
        quarter_turn_ramp: list[int] = []
        for raw_bin, (imaginary, real) in enumerate(
            zip(REAL_CAPTURED_FRAME[0::2], REAL_CAPTURED_FRAME[1::2])
        ):
            # Multiply H by j**raw_bin using exact integer sign/swap operations.
            rotation = raw_bin % 4
            if rotation == 0:
                transformed = (imaginary, real)
            elif rotation == 1:
                transformed = (real, -imaginary)
            elif rotation == 2:
                transformed = (-imaginary, -real)
            else:
                transformed = (-real, imaginary)
            quarter_turn_ramp.extend(transformed)

        ramped = link_v7c.extract_flat_features(quarter_turn_ramp)
        np.testing.assert_array_equal(ramped[:56], baseline[:56])
        np.testing.assert_allclose(ramped[56:], baseline[56:], rtol=0.0, atol=2e-7)

    def test_scaling_changes_only_amplitude_section(self) -> None:
        raw = _synthetic_frame()
        baseline = link_v7c.extract_flat_features(raw)
        doubled = link_v7c.extract_flat_features([2 * value for value in raw])
        np.testing.assert_allclose(doubled[:56], 2.0 * baseline[:56], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(doubled[56:], baseline[56:], rtol=0.0, atol=2e-7)

    def test_validation_rejects_wrong_shapes_types_and_ranges(self) -> None:
        valid = list(_constant_frame())
        for bad_length in (valid[:-1], valid + [0]):
            with self.subTest(length=len(bad_length)), self.assertRaises(ValueError):
                link_v7c.validate_compensated_iq(bad_length)

        with self.assertRaises(ValueError):
            link_v7c.validate_compensated_iq(np.zeros((57, 2), dtype=np.int16))
        with self.assertRaises(TypeError):
            link_v7c.validate_compensated_iq([1.0] * 114)
        with self.assertRaises(TypeError):
            link_v7c.validate_compensated_iq([True] * 114)
        for outside in (-32769, 32768):
            bad = valid.copy()
            bad[0] = outside
            with self.subTest(outside=outside), self.assertRaises(ValueError):
                link_v7c.validate_compensated_iq(bad)

        boundaries = valid.copy()
        boundaries[0] = -32768
        boundaries[1] = 32767
        validated = link_v7c.validate_compensated_iq(boundaries)
        self.assertEqual(validated.dtype, np.dtype(np.int16))
        self.assertFalse(validated.flags.writeable)

    def test_json_parser_is_strict(self) -> None:
        expected = link_v7c.validate_compensated_iq(REAL_CAPTURED_FRAME)
        parsed = link_v7c.parse_compensated_iq_json(json.dumps(REAL_CAPTURED_FRAME))
        np.testing.assert_array_equal(parsed, expected)

        invalid_json_values: Iterable[object] = (
            "[",
            "{}",
            json.dumps([[0, 0]] * 57),
            json.dumps([0.0] * 114),
            json.dumps([32768] + [0] * 113),
            b"[]",
        )
        for value in invalid_json_values:
            with self.subTest(value_type=type(value).__name__), self.assertRaises(
                (TypeError, ValueError)
            ):
                link_v7c.parse_compensated_iq_json(value)


class TestLinkV7cPortableC(unittest.TestCase):
    _temporary_directory: tempfile.TemporaryDirectory[str]
    library: ctypes.CDLL

    @classmethod
    def setUpClass(cls) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            raise unittest.SkipTest("a C99 host compiler named 'cc' is required")

        cls._temporary_directory = tempfile.TemporaryDirectory(prefix="link_v7c_test_")
        library_path = Path(cls._temporary_directory.name) / "libcsi_link_v7c.so"
        compile_result = subprocess.run(
            [
                compiler,
                "-std=c99",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-fPIC",
                "-shared",
                str(CSI_C_DIR / "csi_link_v7c_features.c"),
                "-I",
                str(CSI_C_DIR),
                "-lm",
                "-o",
                str(library_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(
                "failed to compile portable link-v7c C reference:\n"
                + compile_result.stdout
                + compile_result.stderr
            )

        cls.library = ctypes.CDLL(str(library_path))
        cls.library.csi_link_v7c_compute_flat.argtypes = (
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        )
        cls.library.csi_link_v7c_compute_flat.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def _compute_c(
        self,
        values: Sequence[int],
        *,
        scalar_count: int | None = None,
        flat_count: int = link_v7c.FEATURE_COUNT,
        initial_output: float = 0.0,
    ) -> tuple[int, np.ndarray]:
        input_array = (ctypes.c_int16 * len(values))(*values)
        output_array = (ctypes.c_float * link_v7c.FEATURE_COUNT)(
            *([initial_output] * link_v7c.FEATURE_COUNT)
        )
        status = self.library.csi_link_v7c_compute_flat(
            input_array,
            len(values) if scalar_count is None else scalar_count,
            output_array,
            flat_count,
        )
        return status, np.ctypeslib.as_array(output_array).copy()

    def test_exported_c_contract_id_matches_python(self) -> None:
        c_contract = (ctypes.c_char * (len(link_v7c.CONTRACT_ID) + 1)).in_dll(
            self.library, "csi_link_v7c_contract_id"
        )
        self.assertEqual(c_contract.value.decode("ascii"), link_v7c.CONTRACT_ID)

    def test_python_c_golden_vectors_match(self) -> None:
        cases = {
            "synthetic": _synthetic_frame(),
            "deep_fade": _deep_fade_frame(),
            "adaptive_threshold_boundary": _adaptive_threshold_boundary_frame(),
            "all_zero": (0,) * link_v7c.INPUT_SCALAR_COUNT,
            "real_capture": REAL_CAPTURED_FRAME,
        }
        for name, values in cases.items():
            with self.subTest(case=name):
                status, c_flat = self._compute_c(values)
                self.assertEqual(status, 0)
                py_flat = link_v7c.extract_flat_features(values)
                np.testing.assert_allclose(c_flat, py_flat, rtol=2e-6, atol=2e-6)

    def test_seeded_randomized_python_c_parity(self) -> None:
        rng = np.random.default_rng(20260719)

        # Full-range frames stress float32 amplitude/common-phasor arithmetic;
        # low-amplitude frames exercise the absolute threshold and masking.
        for case_index in range(96):
            if case_index % 2:
                values = rng.integers(
                    -32768, 32768, size=link_v7c.INPUT_SCALAR_COUNT, dtype=np.int32
                )
            else:
                values = rng.integers(
                    -12, 13, size=link_v7c.INPUT_SCALAR_COUNT, dtype=np.int32
                )
            with self.subTest(case=case_index):
                raw = values.tolist()
                status, c_flat = self._compute_c(raw)
                self.assertEqual(status, 0)
                np.testing.assert_allclose(
                    c_flat,
                    link_v7c.extract_flat_features(raw),
                    rtol=2e-6,
                    atol=2e-6,
                )

    def test_c_analytic_flattened_order(self) -> None:
        status, flat = self._compute_c(_constant_frame())
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(flat[:56], np.full(56, 5.0, dtype=np.float32))
        np.testing.assert_array_equal(flat[56:110], np.ones(54, dtype=np.float32))
        np.testing.assert_array_equal(flat[110:164], np.zeros(54, dtype=np.float32))
        np.testing.assert_array_equal(flat[164:], np.ones(2, dtype=np.float32))

    def test_c_rejects_wrong_sizes_and_null_pointers(self) -> None:
        values = _constant_frame()
        status, output = self._compute_c(values, scalar_count=113, initial_output=9.0)
        self.assertEqual(status, -2)
        np.testing.assert_array_equal(output, np.zeros(166, dtype=np.float32))

        status, _ = self._compute_c(values, scalar_count=115)
        self.assertEqual(status, -2)
        status, _ = self._compute_c(values, flat_count=165)
        self.assertEqual(status, -3)

        output_array = (ctypes.c_float * link_v7c.FEATURE_COUNT)()
        status = self.library.csi_link_v7c_compute_flat(
            None,
            link_v7c.INPUT_SCALAR_COUNT,
            output_array,
            link_v7c.FEATURE_COUNT,
        )
        self.assertEqual(status, -1)

        input_array = (ctypes.c_int16 * link_v7c.INPUT_SCALAR_COUNT)(*values)
        status = self.library.csi_link_v7c_compute_flat(
            input_array,
            link_v7c.INPUT_SCALAR_COUNT,
            None,
            link_v7c.FEATURE_COUNT,
        )
        self.assertEqual(status, -1)


if __name__ == "__main__":
    unittest.main()
