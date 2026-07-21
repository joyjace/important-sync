"""Deployment-contract tests for exact CSI_B64 gain compensation and link-v7c.

The tests deliberately cross the host parser, a freshly compiled copy of the
portable receiver C, and the Python link-v7c feature reference.  They require
only NumPy plus a host C compiler; serial is stubbed when pyserial is absent.
"""

from __future__ import annotations

import ast
import base64
import ctypes
import importlib.util
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np


SHARED_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
CSI_C_DIR = REPOSITORY_ROOT / "csi_recv" / "main"
COLLECTOR_PATH = REPOSITORY_ROOT / "tools" / "csi_data_read_parse_SSH.py"
TRAIN_DQN_PATH = (
    REPOSITORY_ROOT / "tools" / "rl" / "DQN" / "dqn_model" / "train_dqn.py"
)
sys.path.insert(0, str(SHARED_DIR))

import csi_link_v7c as link_v7c  # noqa: E402


def _source_string_constant(path: Path, name: str) -> str:
    """Read a top-level string constant without importing training dependencies."""

    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            value = ast.literal_eval(statement.value)
            if not isinstance(value, str):
                break
            return value
    raise AssertionError(f"missing top-level string constant {name} in {path}")


try:
    import serial as _serial  # noqa: F401,E402
except ModuleNotFoundError:
    # Parsing one line does not use the serial API.  Keep this protocol test
    # runnable in a minimal model-build environment without pyserial.
    sys.modules["serial"] = types.ModuleType("serial")

_COLLECTOR_SPEC = importlib.util.spec_from_file_location(
    "_csi_data_read_parse_SSH_protocol_test", COLLECTOR_PATH
)
if _COLLECTOR_SPEC is None or _COLLECTOR_SPEC.loader is None:
    raise RuntimeError(f"cannot load CSI collector from {COLLECTOR_PATH}")
collector = importlib.util.module_from_spec(_COLLECTOR_SPEC)
_COLLECTOR_SPEC.loader.exec_module(collector)


def _reference_crc16_ccitt(data: bytes) -> int:
    """Small bitwise CRC reference independent of the parser's lookup table."""

    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def _i8_bytes(values: Sequence[int]) -> bytes:
    if any(value < -128 or value > 127 for value in values):
        raise ValueError("raw CSI values must fit signed int8")
    return bytes(value & 0xFF for value in values)


def _make_b64_line(
    *,
    version: str = "2",
    gain: str = "40B1A347",
    raw_values: Sequence[int] = (127, -127),
    data_len: object | None = None,
    encoding: str = "i8",
    payload: str | None = None,
    prefix: str = "",
) -> str:
    raw_bytes = _i8_bytes(raw_values)
    if payload is None:
        payload = base64.b64encode(raw_bytes).decode("ascii")
    if data_len is None:
        data_len = len(raw_bytes)
    body = ",".join(
        [
            "CSI_B64",
            version,
            "42",
            "AA:BB:CC:DD:EE:FF",
            "-57",
            "16",
            "-97",
            "3",
            "49",
            "1",
            "123456",
            "171",
            "0",
            str(data_len),
            "0",
            gain,
            encoding,
            payload,
        ]
    )
    crc = _reference_crc16_ccitt(body.encode("utf-8"))
    return f"{prefix}{body},{crc:04X}"


def _f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", float(value)))[0]


def _gain_from_bits(bits: int) -> float:
    return struct.unpack(">f", bits.to_bytes(4, "big"))[0]


def _python_v2_compensate(gain: float, raw_value: int) -> tuple[int, int]:
    """Mirror the v2 binary32 multiply, validation, and C-style cast."""

    if not math.isfinite(gain) or gain <= 0.0:
        return -2, 0
    try:
        scaled = _f32(gain * raw_value)
    except OverflowError:
        return -3, 0
    if not math.isfinite(scaled) or scaled < -32768.0 or scaled > 32767.0:
        return -3, 0
    return 0, int(scaled)


def _active_amplitude_summary(amplitudes: Sequence[float]) -> np.ndarray:
    """Independent float32 implementation of the five v7c state summaries."""

    values = np.asarray(amplitudes, dtype=np.float32)
    if values.shape != (56,):
        raise ValueError("expected 56 active amplitudes")

    total = np.float32(0.0)
    total_sq = np.float32(0.0)
    for value in values:
        total = np.float32(total + value)
        total_sq = np.float32(total_sq + np.float32(value * value))
    mean = np.float32(total / np.float32(56.0))
    variance = np.float32(
        np.float32(total_sq / np.float32(56.0)) - np.float32(mean * mean)
    )
    variance = np.float32(max(variance, np.float32(0.0)))
    ordered = np.sort(values)

    def quantile(q: float) -> np.float32:
        position = np.float32(np.float32(q) * np.float32(55.0))
        lower = int(position)
        upper = min(lower + 1, 55)
        fraction = np.float32(position - np.float32(lower))
        return np.float32(
            np.float32(ordered[lower] * np.float32(np.float32(1.0) - fraction))
            + np.float32(ordered[upper] * fraction)
        )

    return np.asarray(
        [
            mean,
            np.float32(np.sqrt(variance)),
            quantile(0.10),
            quantile(0.50),
            quantile(0.90),
        ],
        dtype=np.float32,
    )


class TestCsiB64Protocol(unittest.TestCase):
    def test_single_port_auto_detection_is_content_based_at_921600(self) -> None:
        self.assertEqual(collector.infer_single_port_mode(921600), "auto")

    def test_crc_reference_and_standard_check_value(self) -> None:
        self.assertEqual(_reference_crc16_ccitt(b"123456789"), 0x29B1)
        self.assertEqual(collector.crc16_ccitt(b"123456789"), 0x29B1)

    def test_exact_v2_smoking_vector(self) -> None:
        line = _make_b64_line()
        self.assertEqual(
            line,
            "CSI_B64,2,42,AA:BB:CC:DD:EE:FF,-57,16,-97,3,49,1,"
            "123456,171,0,2,0,40B1A347,i8,f4E=,678B",
        )

        frame, error = collector.parse_csi_b64_line(line)

        self.assertIsNone(error)
        self.assertIsNotNone(frame)
        self.assertEqual(frame["raw_data"], [705, -705])
        self.assertEqual(frame["b64_version"], 2)
        self.assertEqual(frame["compensate_gain_f32_hex"], "40B1A347")
        self.assertEqual(frame["compensate_gain"], _gain_from_bits(0x40B1A347))
        self.assertIs(frame["gain_compensation_exact"], True)

    def test_v1_decimal_capture_compatibility_is_preserved(self) -> None:
        line = _make_b64_line(
            version="1",
            gain="5.551181",
            prefix="boot noise before frame: ",
        )

        frame, error = collector.parse_csi_b64_line(line)

        self.assertIsNone(error)
        self.assertEqual(frame["raw_data"], [704, -704])
        self.assertEqual(frame["b64_version"], 1)
        self.assertEqual(frame["compensate_gain"], 5.551181)
        self.assertIsNone(frame["compensate_gain_f32_hex"])
        self.assertIs(frame["gain_compensation_exact"], False)
        self.assertTrue(frame["line"].startswith("CSI_B64,1,"))

    def test_crc_and_record_shape_errors_are_rejected(self) -> None:
        valid = _make_b64_line()
        body, crc_text = valid.rsplit(",", 1)
        wrong_crc = f"{body},{int(crc_text, 16) ^ 1:04X}"
        cases = {
            "not CSI": "unrelated log output",
            "wrong field count": valid + ",extra",
            "invalid CRC text": f"{body},NOPE",
            "CRC mismatch": wrong_crc,
        }
        expected_prefixes = {
            "not CSI": "not_csi",
            "wrong field count": "b64_wrong_field_count",
            "invalid CRC text": "b64_invalid_crc_field",
            "CRC mismatch": "b64_crc_mismatch",
        }
        for name, line in cases.items():
            with self.subTest(case=name):
                frame, error = collector.parse_csi_b64_line(line)
                self.assertIsNone(frame)
                self.assertTrue(error.startswith(expected_prefixes[name]), error)

    def test_v2_gain_field_is_exact_and_strict(self) -> None:
        cases = {
            "too short": ("40B1A34", "b64_invalid_gain_f32_hex"),
            "nonhex": ("40B1A34Z", "b64_invalid_gain_f32_hex"),
            "whitespace": (" 40B1A347", "b64_invalid_gain_f32_hex"),
            "zero": ("00000000", "b64_invalid_gain_value"),
            "negative zero": ("80000000", "b64_invalid_gain_value"),
            "negative": ("BF800000", "b64_invalid_gain_value"),
            "infinity": ("7F800000", "b64_invalid_gain_value"),
            "nan": ("7FC00000", "b64_invalid_gain_value"),
        }
        for name, (gain, expected_error) in cases.items():
            with self.subTest(case=name):
                frame, error = collector.parse_csi_b64_line(
                    _make_b64_line(gain=gain)
                )
                self.assertIsNone(frame)
                self.assertEqual(error, expected_error)

    def test_v1_invalid_gains_are_rejected_without_breaking_v1_syntax(self) -> None:
        cases = {
            "not numeric": ("invalid", "b64_invalid_len_or_gain"),
            "zero": ("0", "b64_invalid_gain_value"),
            "negative": ("-1.25", "b64_invalid_gain_value"),
            "infinity": ("inf", "b64_invalid_gain_value"),
            "nan": ("nan", "b64_invalid_gain_value"),
        }
        for name, (gain, expected_error) in cases.items():
            with self.subTest(case=name):
                frame, error = collector.parse_csi_b64_line(
                    _make_b64_line(version="1", gain=gain)
                )
                self.assertIsNone(frame)
                self.assertEqual(error, expected_error)

    def test_encoding_payload_and_length_errors_are_rejected(self) -> None:
        cases = {
            "unknown version": (
                _make_b64_line(version="3"),
                "b64_unknown_version: 3",
            ),
            "unknown encoding": (
                _make_b64_line(encoding="i16"),
                "b64_unknown_encoding: i16",
            ),
            "invalid length": (
                _make_b64_line(data_len="two"),
                "b64_invalid_len_or_gain",
            ),
            "invalid base64": (
                _make_b64_line(payload="***="),
                "b64_decode_error",
            ),
            "length mismatch": (
                _make_b64_line(data_len=4),
                "b64_len_mismatch: header=4, actual=2",
            ),
            "odd decoded count": (
                _make_b64_line(raw_values=(1, 2, 3)),
                "b64_odd_value_count: 3",
            ),
        }
        for name, (line, expected_error) in cases.items():
            with self.subTest(case=name):
                frame, error = collector.parse_csi_b64_line(line)
                self.assertIsNone(frame)
                self.assertEqual(error, expected_error)

    def test_compensated_int16_overflow_is_rejected(self) -> None:
        gain_300_bits = struct.pack(">f", 300.0).hex().upper()
        frame, error = collector.parse_csi_b64_line(
            _make_b64_line(gain=gain_300_bits)
        )
        self.assertIsNone(frame)
        self.assertEqual(error, "b64_compensated_value_overflow")


class _CStateMetadata(ctypes.Structure):
    _fields_ = (
        ("rssi", ctypes.c_float),
        ("snr", ctypes.c_float),
        ("fft_gain", ctypes.c_float),
        ("agc_gain", ctypes.c_float),
        ("channel", ctypes.c_float),
        ("sig_len", ctypes.c_float),
        ("state_age_packets", ctypes.c_uint16),
        ("state_packet_gap", ctypes.c_uint16),
        ("state_missing_packets", ctypes.c_uint16),
        ("state_is_stale", ctypes.c_uint8),
    )


class TestPortableDeploymentPipeline(unittest.TestCase):
    _temporary_directory: tempfile.TemporaryDirectory[str]
    library: ctypes.CDLL

    @classmethod
    def setUpClass(cls) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            raise unittest.SkipTest("a C99 host compiler named 'cc' is required")

        cls._temporary_directory = tempfile.TemporaryDirectory(
            prefix="csi_b64_deployment_test_"
        )
        library_path = Path(cls._temporary_directory.name) / "libcsi_deployment.so"
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
                str(CSI_C_DIR / "csi_gain_compensation.c"),
                str(CSI_C_DIR / "csi_link_v7c_features.c"),
                str(CSI_C_DIR / "csi_link_v7c_state.c"),
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
                "failed to compile portable deployment reference:\n"
                + compile_result.stdout
                + compile_result.stderr
            )

        cls.library = ctypes.CDLL(str(library_path))
        cls.library.csi_gain_f32_bits.argtypes = (ctypes.c_float,)
        cls.library.csi_gain_f32_bits.restype = ctypes.c_uint32
        cls.library.csi_gain_f32_from_bits.argtypes = (ctypes.c_uint32,)
        cls.library.csi_gain_f32_from_bits.restype = ctypes.c_float
        cls.library.csi_gain_compensate_i8.argtypes = (
            ctypes.c_float,
            ctypes.c_int8,
            ctypes.POINTER(ctypes.c_int16),
        )
        cls.library.csi_gain_compensate_i8.restype = ctypes.c_int
        cls.library.csi_gain_compensate_frame_i8.argtypes = (
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int16),
        )
        cls.library.csi_gain_compensate_frame_i8.restype = ctypes.c_int
        cls.library.csi_link_v7c_build_receiver_state.argtypes = (
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_size_t,
            ctypes.POINTER(_CStateMetadata),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        )
        cls.library.csi_link_v7c_build_receiver_state.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def _compensate_frame_c(
        self, gain: float, raw_values: Sequence[int]
    ) -> tuple[int, np.ndarray]:
        raw = (ctypes.c_int8 * len(raw_values))(*raw_values)
        output = (ctypes.c_int16 * len(raw_values))(
            *([12345] * len(raw_values))
        )
        status = self.library.csi_gain_compensate_frame_i8(
            gain, raw, len(raw_values), output
        )
        return status, np.ctypeslib.as_array(output).copy()

    def test_float_bit_roundtrip_and_c_smoking_vector(self) -> None:
        bits = 0x40B1A347
        gain = self.library.csi_gain_f32_from_bits(bits)
        self.assertEqual(self.library.csi_gain_f32_bits(gain), bits)

        positive = ctypes.c_int16()
        negative = ctypes.c_int16()
        self.assertEqual(
            self.library.csi_gain_compensate_i8(gain, 127, ctypes.byref(positive)),
            0,
        )
        self.assertEqual(
            self.library.csi_gain_compensate_i8(gain, -127, ctypes.byref(negative)),
            0,
        )
        self.assertEqual((positive.value, negative.value), (705, -705))

    def test_training_and_portable_c_state_contract_identity_matches(self) -> None:
        expected_id = _source_string_constant(
            TRAIN_DQN_PATH, "LINK_V7C_STATE_CONTRACT_ID"
        )
        expected_sha256 = _source_string_constant(
            TRAIN_DQN_PATH, "LINK_V7C_STATE_CONTRACT_SHA256"
        )
        c_id = (ctypes.c_char * (len(expected_id) + 1)).in_dll(
            self.library, "csi_link_v7c_state_contract_id"
        )
        c_sha256 = (ctypes.c_char * (len(expected_sha256) + 1)).in_dll(
            self.library, "csi_link_v7c_state_contract_sha256"
        )

        self.assertEqual(c_id.value.decode("ascii"), expected_id)
        self.assertEqual(c_sha256.value.decode("ascii"), expected_sha256)

    def test_all_i8_values_match_python_across_exact_binary32_gains(self) -> None:
        # Every possible raw int8 value is checked for each representative
        # exact gain, including subnormal, rounding-boundary, overflow, and
        # maximum-finite binary32 cases.
        gain_bits = (
            0x00000001,
            0x00800000,
            0x3EAAAAAB,
            0x3F000000,
            0x3F7FFFFF,
            0x3F800000,
            0x3F800001,
            0x40B1A347,
            0x41200000,
            0x437FFFFF,
            0x43800000,
            0x43960000,
            0x7F7FFFFF,
        )
        for bits in gain_bits:
            gain = self.library.csi_gain_f32_from_bits(bits)
            self.assertEqual(self.library.csi_gain_f32_bits(gain), bits)
            for raw_value in range(-128, 128):
                output = ctypes.c_int16(12345)
                c_status = self.library.csi_gain_compensate_i8(
                    gain, raw_value, ctypes.byref(output)
                )
                expected_status, expected_value = _python_v2_compensate(
                    gain, raw_value
                )
                message = f"gain_bits={bits:08X}, raw={raw_value}"
                self.assertEqual(c_status, expected_status, message)
                self.assertEqual(output.value, expected_value, message)

        gain = self.library.csi_gain_f32_from_bits(0x40B1A347)
        raw_values = tuple(range(-128, 128))
        status, output = self._compensate_frame_c(gain, raw_values)
        self.assertEqual(status, 0)
        expected = np.asarray(
            [_python_v2_compensate(gain, value)[1] for value in raw_values],
            dtype=np.int16,
        )
        np.testing.assert_array_equal(output, expected)

    def test_c_rejects_bad_gains_and_zeroes_failed_frames(self) -> None:
        for name, gain in {
            "positive zero": 0.0,
            "negative zero": -0.0,
            "negative": -1.0,
            "infinity": math.inf,
            "nan": math.nan,
        }.items():
            with self.subTest(case=name):
                output = ctypes.c_int16(12345)
                status = self.library.csi_gain_compensate_i8(
                    gain, 1, ctypes.byref(output)
                )
                self.assertEqual(status, -2)
                self.assertEqual(output.value, 0)

        self.assertEqual(
            self.library.csi_gain_compensate_i8(1.0, 1, None),
            -1,
        )
        status, output = self._compensate_frame_c(300.0, (1, 127, 2))
        self.assertEqual(status, -3)
        np.testing.assert_array_equal(output, np.zeros(3, dtype=np.int16))

        output_array = (ctypes.c_int16 * 2)()
        self.assertEqual(
            self.library.csi_gain_compensate_frame_i8(
                1.0, None, 2, output_array
            ),
            -1,
        )

    def test_b64_raw_gain_to_v7c_receiver_state_python_c_parity(self) -> None:
        raw_values: list[int] = []
        for raw_bin in range(link_v7c.COMPLEX_BIN_COUNT):
            imaginary = ((raw_bin * 17 + 11) % 101) - 50
            real = ((raw_bin * 29 + 7) % 99) - 49
            raw_values.extend((imaginary, real))
        self.assertEqual(len(raw_values), link_v7c.INPUT_SCALAR_COUNT)

        line = _make_b64_line(raw_values=raw_values)
        frame, error = collector.parse_csi_b64_line(line)
        self.assertIsNone(error)
        gain = frame["compensate_gain"]

        gain_status, c_compensated = self._compensate_frame_c(gain, raw_values)
        self.assertEqual(gain_status, 0)
        np.testing.assert_array_equal(
            c_compensated,
            np.asarray(frame["raw_data"], dtype=np.int16),
        )

        metadata = _CStateMetadata(
            rssi=frame["rssi"],
            snr=frame["rssi"] - frame["noise_floor"],
            fft_gain=frame["fft_gain"],
            agc_gain=frame["agc_gain"],
            channel=frame["channel"],
            sig_len=frame["sig_len"],
            state_age_packets=3,
            state_packet_gap=4,
            state_missing_packets=2,
            state_is_stale=1,
        )
        c_input = (ctypes.c_int16 * len(c_compensated))(*c_compensated)
        c_state_buffer = (ctypes.c_float * 181)()
        state_status = self.library.csi_link_v7c_build_receiver_state(
            c_input,
            len(c_compensated),
            ctypes.byref(metadata),
            c_state_buffer,
            181,
        )
        self.assertEqual(state_status, 0)
        c_state = np.ctypeslib.as_array(c_state_buffer).copy()

        python_features = link_v7c.extract_flat_features(frame["raw_data"])
        python_tail = np.concatenate(
            [
                np.asarray(
                    [
                        metadata.rssi,
                        metadata.snr,
                        metadata.fft_gain,
                        metadata.agc_gain,
                        metadata.channel,
                        metadata.sig_len,
                        np.log1p(np.float32(metadata.state_age_packets)),
                        np.log1p(np.float32(metadata.state_packet_gap)),
                        np.log1p(np.float32(metadata.state_missing_packets)),
                        np.float32(1.0),
                    ],
                    dtype=np.float32,
                ),
                _active_amplitude_summary(python_features[:56]),
            ]
        ).astype(np.float32)
        python_state = np.concatenate([python_features, python_tail]).astype(
            np.float32
        )

        self.assertEqual(c_state.shape, (181,))
        self.assertTrue(np.all(np.isfinite(c_state)))
        np.testing.assert_allclose(c_state, python_state, rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
