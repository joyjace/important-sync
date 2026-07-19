#!/usr/bin/env python3
"""Reference implementation of the ``link_v7c_ht20_v1`` CSI contract.

The input is one already gain-compensated ESP32-C5 HT20 CSI frame: exactly
57 complex bins (114 signed int16 scalars) interleaved as ``[imaginary, real]``.
Raw bin 28 is DC and is deliberately excluded from both amplitude features and
phase-difference pairs.

The flattened feature order is stable and intentionally simple to reproduce in
firmware::

    56 active-bin amplitudes
    54 corrected adjacent-phase real components
    54 corrected adjacent-phase imaginary components
    phase valid fraction
    phase coherence

All feature arithmetic is performed with NumPy float32 scalars in raw-bin
order.  The explicit scalar operation ordering mirrors the portable C
reference and avoids accidentally promoting the reference calculation to
float64.  ``CONTRACT_SHA256`` fingerprints the canonical contract document;
dataset builders and exported models should persist both the schema ID and
this digest.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


SCHEMA_ID = "link_v7c_ht20_v1"
# ``CONTRACT_ID`` is the terminology used by the portable C implementation;
# retain ``SCHEMA_ID`` for dataset/model metadata without duplicating a value.
CONTRACT_ID = SCHEMA_ID
CONTRACT_VERSION = 1

COMPLEX_BIN_COUNT = 57
DC_RAW_BIN_INDEX = 28
INPUT_SCALAR_COUNT = COMPLEX_BIN_COUNT * 2
ACTIVE_AMPLITUDE_COUNT = 56
PHASE_PAIR_COUNT = 54
FEATURE_COUNT = ACTIVE_AMPLITUDE_COUNT + 2 * PHASE_PAIR_COUNT + 2
DIFFERENTIAL_PHASE_FEATURE_START = ACTIVE_AMPLITUDE_COUNT
DIFFERENTIAL_PHASE_FEATURE_STOP = ACTIVE_AMPLITUDE_COUNT + 2 * PHASE_PAIR_COUNT
PHASE_DERIVED_FEATURE_START = DIFFERENTIAL_PHASE_FEATURE_START
PHASE_DERIVED_FEATURE_STOP = FEATURE_COUNT

ABSOLUTE_MIN_AMPLITUDE = np.float32(2.0)
RELATIVE_MIN_AMPLITUDE = np.float32(0.10)

ACTIVE_RAW_BIN_INDICES: Tuple[int, ...] = tuple(
    raw_bin for raw_bin in range(COMPLEX_BIN_COUNT) if raw_bin != DC_RAW_BIN_INDEX
)
PHASE_PAIR_RAW_BIN_INDICES: Tuple[Tuple[int, int], ...] = tuple(
    (raw_bin, raw_bin + 1) for raw_bin in range(0, DC_RAW_BIN_INDEX - 1)
) + tuple(
    (raw_bin, raw_bin + 1)
    for raw_bin in range(DC_RAW_BIN_INDEX + 1, COMPLEX_BIN_COUNT - 1)
)

AMPLITUDE_FEATURE_NAMES: Tuple[str, ...] = tuple(
    f"iq_amp_raw_bin_{raw_bin:02d}" for raw_bin in ACTIVE_RAW_BIN_INDICES
)
PHASE_REAL_FEATURE_NAMES: Tuple[str, ...] = tuple(
    f"iq_phase_diff_real_raw_bins_{current:02d}_{following:02d}"
    for current, following in PHASE_PAIR_RAW_BIN_INDICES
)
PHASE_IMAG_FEATURE_NAMES: Tuple[str, ...] = tuple(
    f"iq_phase_diff_imag_raw_bins_{current:02d}_{following:02d}"
    for current, following in PHASE_PAIR_RAW_BIN_INDICES
)
QUALITY_FEATURE_NAMES: Tuple[str, str] = (
    "iq_phase_valid_fraction",
    "iq_phase_coherence",
)
FEATURE_NAMES: Tuple[str, ...] = (
    AMPLITUDE_FEATURE_NAMES
    + PHASE_REAL_FEATURE_NAMES
    + PHASE_IMAG_FEATURE_NAMES
    + QUALITY_FEATURE_NAMES
)

if len(ACTIVE_RAW_BIN_INDICES) != ACTIVE_AMPLITUDE_COUNT:
    raise RuntimeError("link_v7c active-bin map does not contain 56 bins")
if len(PHASE_PAIR_RAW_BIN_INDICES) != PHASE_PAIR_COUNT:
    raise RuntimeError("link_v7c phase-pair map does not contain 54 pairs")
if len(FEATURE_NAMES) != FEATURE_COUNT or len(set(FEATURE_NAMES)) != FEATURE_COUNT:
    raise RuntimeError("link_v7c feature names must be 166 unique values")


_CONTRACT_DOCUMENT: Dict[str, Any] = {
    "schema_id": SCHEMA_ID,
    "contract_version": CONTRACT_VERSION,
    "input": {
        "description": "gain-compensated ESP32-C5 HT20 CSI",
        "scalar_type": "signed_int16",
        "scalar_count": INPUT_SCALAR_COUNT,
        "complex_bin_count": COMPLEX_BIN_COUNT,
        "scalar_layout": "flat_interleaved",
        "scalar_order_per_bin": ["imaginary", "real"],
        "canonical_complex_value": "H = real + j*imaginary",
    },
    "carrier_map": {
        "raw_bin_indices": list(range(COMPLEX_BIN_COUNT)),
        "dc_raw_bin_index": DC_RAW_BIN_INDEX,
        "active_raw_bin_indices": list(ACTIVE_RAW_BIN_INDICES),
        "phase_pair_raw_bin_indices": [
            list(pair) for pair in PHASE_PAIR_RAW_BIN_INDICES
        ],
        "pairs_do_not_cross_dc": True,
    },
    "amplitude": {
        "formula": "sqrt(real*real + imaginary*imaginary)",
        "output_raw_bin_indices": list(ACTIVE_RAW_BIN_INDICES),
        "dc_is_excluded": True,
    },
    "phase": {
        "raw_differential": "D = H_next*conj(H_current)/(amp_next*amp_current)",
        "validity_threshold": {
            "formula": "max(absolute_min, relative_min*median(active_amplitudes))",
            "absolute_min": 2.0,
            "relative_min": 0.10,
            "comparison": "both endpoint amplitudes >= threshold",
            "median": "mean of sorted active amplitudes at zero-based indices 27 and 28",
        },
        "valid_pair_weight": "min(amp_current, amp_next)",
        "common_differential": "unit(sum(weight*D for valid pairs))",
        "common_zero_fallback": "1+j0 when the weighted-sum magnitude is zero",
        "correction": "D_corrected = D*conj(common_differential)",
        "invalid_pair_output": [0.0, 0.0],
        "epsilon": 0.0,
    },
    "quality": {
        "valid_fraction": "valid_pair_count/54",
        "coherence": (
            "clamp(abs(sum(weight*D))/sum(weight), 0, 1); "
            "zero when no common phasor exists"
        ),
    },
    "numeric": {
        "feature_dtype": "float32",
        "operation_order": "raw-bin order, then phase-pair order, with float32 scalar operations",
        "finite_input_guarantee": "signed int16 input makes every result finite",
    },
    "output": {
        "feature_count": FEATURE_COUNT,
        "flattened_sections": [
            {"name": "amplitude", "offset": 0, "count": ACTIVE_AMPLITUDE_COUNT},
            {
                "name": "phase_real",
                "offset": ACTIVE_AMPLITUDE_COUNT,
                "count": PHASE_PAIR_COUNT,
            },
            {
                "name": "phase_imag",
                "offset": ACTIVE_AMPLITUDE_COUNT + PHASE_PAIR_COUNT,
                "count": PHASE_PAIR_COUNT,
            },
            {"name": "phase_valid_fraction", "offset": FEATURE_COUNT - 2, "count": 1},
            {"name": "phase_coherence", "offset": FEATURE_COUNT - 1, "count": 1},
        ],
        "feature_names": list(FEATURE_NAMES),
    },
}

# Canonical means sorted-key, whitespace-free, UTF-8 JSON.  The exact text is
# public so non-Python consumers can reproduce and verify the fingerprint.
CONTRACT_CANONICAL_JSON = json.dumps(
    _CONTRACT_DOCUMENT,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
)

# Deliberately a literal rather than a digest recomputed and accepted silently.
# Import fails if a contract edit is not accompanied by an intentional version
# and digest update.
CONTRACT_SHA256 = "df4f262b3fdf57f2f693b40b8584c08d5193ba092290d7f771e1a52575c8603a"
_computed_contract_sha256 = hashlib.sha256(
    CONTRACT_CANONICAL_JSON.encode("utf-8")
).hexdigest()
if _computed_contract_sha256 != CONTRACT_SHA256:
    raise RuntimeError(
        "link_v7c contract digest mismatch: update CONTRACT_SHA256 only after "
        f"reviewing the contract change (computed {_computed_contract_sha256})"
    )


def _deep_freeze(value: Any) -> Any:
    """Return an immutable representation of a JSON-compatible value."""

    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


CONTRACT_MANIFEST: Mapping[str, Any] = _deep_freeze(_CONTRACT_DOCUMENT)
del _CONTRACT_DOCUMENT
del _computed_contract_sha256


def _readonly_float32_vector(values: Any, expected_count: int, name: str) -> np.ndarray:
    """Validate and copy one finite float32 feature section."""

    array = np.asarray(values, dtype=np.float32)
    if array.shape != (expected_count,):
        raise ValueError(f"{name} must have shape ({expected_count},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array = np.array(array, dtype=np.float32, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class LinkV7cFeatures:
    """Structured, immutable result of the link-v7c feature transform."""

    amplitudes: np.ndarray
    phase_real: np.ndarray
    phase_imag: np.ndarray
    valid_fraction: np.float32
    coherence: np.float32

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "amplitudes",
            _readonly_float32_vector(
                self.amplitudes, ACTIVE_AMPLITUDE_COUNT, "amplitudes"
            ),
        )
        object.__setattr__(
            self,
            "phase_real",
            _readonly_float32_vector(self.phase_real, PHASE_PAIR_COUNT, "phase_real"),
        )
        object.__setattr__(
            self,
            "phase_imag",
            _readonly_float32_vector(self.phase_imag, PHASE_PAIR_COUNT, "phase_imag"),
        )

        valid_fraction = np.float32(self.valid_fraction)
        coherence = np.float32(self.coherence)
        if (
            not np.isfinite(valid_fraction)
            or not np.float32(0.0) <= valid_fraction <= np.float32(1.0)
        ):
            raise ValueError("valid_fraction must be finite and in [0, 1]")
        if not np.isfinite(coherence) or not np.float32(0.0) <= coherence <= np.float32(1.0):
            raise ValueError("coherence must be finite and in [0, 1]")
        object.__setattr__(self, "valid_fraction", valid_fraction)
        object.__setattr__(self, "coherence", coherence)

    def flatten(self) -> np.ndarray:
        """Return a new float32 vector in the canonical 166-feature order."""

        return flatten_features(self)


def validate_compensated_iq(values: Sequence[int]) -> np.ndarray:
    """Validate and return the canonical read-only ``int16[114]`` input.

    Values must be a one-dimensional sequence of exactly 114 integer scalars
    in the signed-int16 range.  Floating-point values, booleans, nested
    ``(57, 2)`` arrays, and strings are rejected even if they could be coerced.
    Accepting wider integer containers is intentional because ``json.loads``
    represents JSON integers as Python ``int``.
    """

    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(
            "compensated IQ must be an integer sequence; use "
            "parse_compensated_iq_json() for JSON text"
        )

    try:
        array = np.asarray(values)
    except Exception as exc:  # NumPy raises several types for exotic inputs.
        raise TypeError("compensated IQ must be a one-dimensional integer sequence") from exc

    if array.shape != (INPUT_SCALAR_COUNT,):
        raise ValueError(
            f"compensated IQ must have shape ({INPUT_SCALAR_COUNT},), got {array.shape}"
        )
    if array.dtype.kind == "b":
        raise TypeError("compensated IQ booleans are not signed-int16 scalars")

    if array.dtype.kind in "iu":
        if np.any(array < -32768) or np.any(array > 32767):
            raise ValueError("compensated IQ contains a value outside signed-int16 range")
    elif array.dtype.kind == "O":
        for index, value in enumerate(array):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(
                    f"compensated IQ scalar {index} is not an integer: {value!r}"
                )
            if value < -32768 or value > 32767:
                raise ValueError(
                    f"compensated IQ scalar {index} is outside signed-int16 range: {value}"
                )
    else:
        raise TypeError(
            f"compensated IQ requires integer scalars, got dtype {array.dtype}"
        )

    result = np.array(array, dtype=np.int16, copy=True)
    result.flags.writeable = False
    return result


def parse_compensated_iq_json(text: str) -> np.ndarray:
    """Parse a JSON array and apply strict link-v7c input validation."""

    if not isinstance(text, str):
        raise TypeError("compensated IQ JSON must be text")
    try:
        values = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid compensated IQ JSON: {exc.msg}") from exc
    if not isinstance(values, list):
        raise ValueError("compensated IQ JSON must contain one flat array")
    return validate_compensated_iq(values)


def _f32_sqrt(value: np.float32) -> np.float32:
    """Square root without promoting the reference calculation to float64."""

    return np.float32(np.sqrt(np.float32(value)))


def extract_features(compensated_iq: Sequence[int]) -> LinkV7cFeatures:
    """Extract the structured link-v7c features from one compensated frame."""

    iq = validate_compensated_iq(compensated_iq)
    bin_real = np.empty(COMPLEX_BIN_COUNT, dtype=np.float32)
    bin_imag = np.empty(COMPLEX_BIN_COUNT, dtype=np.float32)
    bin_amplitude = np.empty(COMPLEX_BIN_COUNT, dtype=np.float32)

    for raw_bin in range(COMPLEX_BIN_COUNT):
        imaginary = np.float32(iq[2 * raw_bin])
        real = np.float32(iq[2 * raw_bin + 1])
        real_squared = np.float32(real * real)
        imaginary_squared = np.float32(imaginary * imaginary)
        amplitude = _f32_sqrt(np.float32(real_squared + imaginary_squared))
        bin_real[raw_bin] = real
        bin_imag[raw_bin] = imaginary
        bin_amplitude[raw_bin] = amplitude

    amplitudes = np.asarray(
        [bin_amplitude[raw_bin] for raw_bin in ACTIVE_RAW_BIN_INDICES],
        dtype=np.float32,
    )
    sorted_amplitudes = np.sort(amplitudes)
    median_amplitude = np.float32(
        np.float32(sorted_amplitudes[27] + sorted_amplitudes[28])
        * np.float32(0.5)
    )
    relative_threshold = np.float32(
        RELATIVE_MIN_AMPLITUDE * median_amplitude
    )
    threshold = (
        relative_threshold
        if relative_threshold >= ABSOLUTE_MIN_AMPLITUDE
        else ABSOLUTE_MIN_AMPLITUDE
    )

    raw_phase_real = np.zeros(PHASE_PAIR_COUNT, dtype=np.float32)
    raw_phase_imag = np.zeros(PHASE_PAIR_COUNT, dtype=np.float32)
    pair_weights = np.zeros(PHASE_PAIR_COUNT, dtype=np.float32)
    common_real = np.float32(0.0)
    common_imag = np.float32(0.0)
    weight_sum = np.float32(0.0)
    valid_count = 0

    for pair_index, (current, following) in enumerate(PHASE_PAIR_RAW_BIN_INDICES):
        current_amplitude = bin_amplitude[current]
        following_amplitude = bin_amplitude[following]
        if current_amplitude < threshold or following_amplitude < threshold:
            continue

        denominator = np.float32(current_amplitude * following_amplitude)
        numerator_real = np.float32(
            np.float32(bin_real[following] * bin_real[current])
            + np.float32(bin_imag[following] * bin_imag[current])
        )
        numerator_imag = np.float32(
            np.float32(bin_imag[following] * bin_real[current])
            - np.float32(bin_real[following] * bin_imag[current])
        )
        differential_real = np.float32(numerator_real / denominator)
        differential_imag = np.float32(numerator_imag / denominator)
        weight = (
            current_amplitude
            if current_amplitude < following_amplitude
            else following_amplitude
        )

        raw_phase_real[pair_index] = differential_real
        raw_phase_imag[pair_index] = differential_imag
        pair_weights[pair_index] = weight
        common_real = np.float32(
            common_real + np.float32(weight * differential_real)
        )
        common_imag = np.float32(
            common_imag + np.float32(weight * differential_imag)
        )
        weight_sum = np.float32(weight_sum + weight)
        valid_count += 1

    common_magnitude = _f32_sqrt(
        np.float32(
            np.float32(common_real * common_real)
            + np.float32(common_imag * common_imag)
        )
    )
    common_unit_real = np.float32(1.0)
    common_unit_imag = np.float32(0.0)
    coherence = np.float32(0.0)
    if weight_sum > np.float32(0.0) and common_magnitude > np.float32(0.0):
        common_unit_real = np.float32(common_real / common_magnitude)
        common_unit_imag = np.float32(common_imag / common_magnitude)
        coherence = np.float32(common_magnitude / weight_sum)
        if coherence > np.float32(1.0):
            coherence = np.float32(1.0)
        elif coherence < np.float32(0.0):
            coherence = np.float32(0.0)

    phase_real = np.zeros(PHASE_PAIR_COUNT, dtype=np.float32)
    phase_imag = np.zeros(PHASE_PAIR_COUNT, dtype=np.float32)
    for pair_index in range(PHASE_PAIR_COUNT):
        if pair_weights[pair_index] <= np.float32(0.0):
            continue
        phase_real[pair_index] = np.float32(
            np.float32(raw_phase_real[pair_index] * common_unit_real)
            + np.float32(raw_phase_imag[pair_index] * common_unit_imag)
        )
        phase_imag[pair_index] = np.float32(
            np.float32(raw_phase_imag[pair_index] * common_unit_real)
            - np.float32(raw_phase_real[pair_index] * common_unit_imag)
        )

    valid_fraction = np.float32(
        np.float32(valid_count) / np.float32(PHASE_PAIR_COUNT)
    )
    return LinkV7cFeatures(
        amplitudes=amplitudes,
        phase_real=phase_real,
        phase_imag=phase_imag,
        valid_fraction=valid_fraction,
        coherence=coherence,
    )


def flatten_features(features: LinkV7cFeatures) -> np.ndarray:
    """Flatten structured features in the canonical 166-value order."""

    if not isinstance(features, LinkV7cFeatures):
        raise TypeError("features must be a LinkV7cFeatures instance")
    flat = np.concatenate(
        (
            features.amplitudes,
            features.phase_real,
            features.phase_imag,
            np.asarray(
                [features.valid_fraction, features.coherence], dtype=np.float32
            ),
        )
    ).astype(np.float32, copy=False)
    if flat.shape != (FEATURE_COUNT,):
        raise RuntimeError(f"internal link_v7c flatten shape is {flat.shape}, not (166,)")
    return flat


def extract_flat_features(compensated_iq: Sequence[int]) -> np.ndarray:
    """Validate, extract, and flatten one frame in a single call."""

    return flatten_features(extract_features(compensated_iq))


def contract_manifest_dict() -> Dict[str, Any]:
    """Return a fresh JSON-safe copy of the canonical contract document."""

    # Round-tripping the already canonical JSON cannot inherit mutable state
    # from CONTRACT_MANIFEST and preserves exactly the fingerprinted values.
    return json.loads(CONTRACT_CANONICAL_JSON)


def contract_sidecar(include_feature_names: bool = True) -> Dict[str, Any]:
    """Return compact provenance metadata for a dataset or model sidecar."""

    sidecar: Dict[str, Any] = {
        "feature_contract_id": SCHEMA_ID,
        "feature_contract_sha256": CONTRACT_SHA256,
        "input_scalar_count": INPUT_SCALAR_COUNT,
        "feature_count": FEATURE_COUNT,
    }
    if include_feature_names:
        sidecar["feature_names"] = list(FEATURE_NAMES)
    return sidecar


def _inspect_csv_artifact(dataset_path: Path) -> tuple[str, int, list[str]]:
    """Return the byte digest, physical data-row count, and CSV header."""

    digest = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    with dataset_path.open("rb") as handle:
        first_line = handle.readline()
        if not first_line:
            raise ValueError(f"Dataset artifact is empty: {dataset_path}")
        digest.update(first_line)
        newline_count += first_line.count(b"\n")
        last_byte = first_line[-1:]
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]

    try:
        header_text = first_line.decode("utf-8-sig").rstrip("\r\n")
        columns = next(csv.reader([header_text]))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"Cannot parse dataset CSV header {dataset_path}: {exc}") from exc

    # Versioned dataset writers guarantee one CSV record per physical line.
    physical_records = newline_count + (0 if last_byte in (b"\n", b"\r") else 1)
    return digest.hexdigest(), max(0, physical_records - 1), columns


def validate_dataset_artifact(dataset_csv: str | Path) -> Dict[str, Any]:
    """Validate a v7c dataset's contract, bytes, and deployment qualification."""

    dataset_path = Path(dataset_csv)
    sidecar_path = Path(f"{dataset_path}.feature_contract.json")
    if not sidecar_path.is_file():
        raise ValueError(f"link_v7c requires feature-contract sidecar {sidecar_path}")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read feature-contract sidecar {sidecar_path}: {exc}") from exc

    record = sidecar.get("feature_contract")
    if not isinstance(record, dict):
        raise ValueError(f"Feature-contract sidecar {sidecar_path} has no feature_contract object")
    expected_contract = {
        "feature_contract_id": CONTRACT_ID,
        "feature_contract_sha256": CONTRACT_SHA256,
        "feature_count": FEATURE_COUNT,
    }
    contract_mismatches = [
        f"{key}={record.get(key)!r} (expected {value!r})"
        for key, value in expected_contract.items()
        if record.get(key) != value
    ]
    if contract_mismatches:
        raise ValueError(
            f"Incompatible feature-contract sidecar {sidecar_path}: "
            + "; ".join(contract_mismatches)
        )
    embedded_contract = record.get("contract")
    if not isinstance(embedded_contract, dict):
        raise ValueError(
            f"Feature-contract sidecar {sidecar_path} has no canonical contract document"
        )
    embedded_canonical = json.dumps(
        embedded_contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    embedded_sha256 = hashlib.sha256(embedded_canonical.encode("utf-8")).hexdigest()
    embedded_mismatches = []
    if embedded_sha256 != CONTRACT_SHA256:
        embedded_mismatches.append(
            f"embedded contract digest {embedded_sha256!r} != {CONTRACT_SHA256!r}"
        )
    if embedded_canonical != CONTRACT_CANONICAL_JSON:
        embedded_mismatches.append("embedded canonical contract document differs")
    if record.get("input_scalar_count") != INPUT_SCALAR_COUNT:
        embedded_mismatches.append(
            f"input_scalar_count={record.get('input_scalar_count')!r} "
            f"(expected {INPUT_SCALAR_COUNT})"
        )
    if record.get("feature_names") != list(FEATURE_NAMES):
        embedded_mismatches.append("feature_names differ from the canonical order")
    if embedded_mismatches:
        raise ValueError(
            f"Incompatible embedded feature contract in {sidecar_path}: "
            + "; ".join(embedded_mismatches)
        )

    if sidecar.get("manifest_schema") not in {
        "dqn_csi_feature_contract/v1",
        "dqn_csi_feature_contract_merge/v1",
    }:
        raise ValueError(
            f"Unsupported feature-contract manifest schema in {sidecar_path}: "
            f"{sidecar.get('manifest_schema')!r}"
        )

    artifact = sidecar.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError(f"Feature-contract sidecar {sidecar_path} has no artifact object")
    actual_sha256, actual_row_count, actual_columns = _inspect_csv_artifact(dataset_path)
    artifact_mismatches = []
    if artifact.get("sha256") != actual_sha256:
        artifact_mismatches.append(
            f"sha256={artifact.get('sha256')!r} (actual {actual_sha256!r})"
        )
    if artifact.get("row_count") != actual_row_count:
        artifact_mismatches.append(
            f"row_count={artifact.get('row_count')!r} (actual {actual_row_count!r})"
        )
    if artifact.get("columns") != actual_columns:
        artifact_mismatches.append("columns do not match the CSV header")
    if artifact_mismatches:
        raise ValueError(
            f"Dataset artifact does not match {sidecar_path}: "
            + "; ".join(artifact_mismatches)
        )

    qualification = sidecar.get("qualification")
    if not isinstance(qualification, dict):
        raise ValueError(f"Feature-contract sidecar {sidecar_path} has no qualification object")
    expected_qualification = {
        "status": "candidate",
        "causal_alignment": True,
        "gain_compensation_exact": True,
        "deployment_candidate": True,
        "blocking_reasons": [],
    }
    qualification_mismatches = [
        f"{key}={qualification.get(key)!r} (expected {value!r})"
        for key, value in expected_qualification.items()
        if qualification.get(key) != value
    ]
    if qualification_mismatches:
        raise ValueError(
            f"Dataset is not qualified for deployment in {sidecar_path}: "
            + "; ".join(qualification_mismatches)
        )
    return sidecar


__all__ = [
    "ABSOLUTE_MIN_AMPLITUDE",
    "ACTIVE_AMPLITUDE_COUNT",
    "ACTIVE_RAW_BIN_INDICES",
    "AMPLITUDE_FEATURE_NAMES",
    "COMPLEX_BIN_COUNT",
    "CONTRACT_CANONICAL_JSON",
    "CONTRACT_ID",
    "CONTRACT_MANIFEST",
    "CONTRACT_SHA256",
    "CONTRACT_VERSION",
    "DC_RAW_BIN_INDEX",
    "DIFFERENTIAL_PHASE_FEATURE_START",
    "DIFFERENTIAL_PHASE_FEATURE_STOP",
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "INPUT_SCALAR_COUNT",
    "LinkV7cFeatures",
    "PHASE_IMAG_FEATURE_NAMES",
    "PHASE_DERIVED_FEATURE_START",
    "PHASE_DERIVED_FEATURE_STOP",
    "PHASE_PAIR_COUNT",
    "PHASE_PAIR_RAW_BIN_INDICES",
    "PHASE_REAL_FEATURE_NAMES",
    "QUALITY_FEATURE_NAMES",
    "RELATIVE_MIN_AMPLITUDE",
    "SCHEMA_ID",
    "contract_manifest_dict",
    "contract_sidecar",
    "extract_features",
    "extract_flat_features",
    "flatten_features",
    "parse_compensated_iq_json",
    "validate_compensated_iq",
    "validate_dataset_artifact",
]
