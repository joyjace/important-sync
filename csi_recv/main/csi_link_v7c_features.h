/*
 * Portable reference implementation of the link_v7c CSI feature contract.
 *
 * This file deliberately has no ESP-IDF dependencies.  It can be compiled by
 * both the firmware and host-side golden-vector tests.
 */

#ifndef CSI_LINK_V7C_FEATURES_H
#define CSI_LINK_V7C_FEATURES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The input is 57 compensated complex bins encoded as [imaginary, real]. */
#define CSI_LINK_V7C_CONTRACT_ID                    "link_v7c_ht20_v1"
#define CSI_LINK_V7C_CONTRACT_SHA256                "df4f262b3fdf57f2f693b40b8584c08d5193ba092290d7f771e1a52575c8603a"
#define CSI_LINK_V7C_BIN_COUNT                      57u
#define CSI_LINK_V7C_DC_BIN_INDEX                   28u
#define CSI_LINK_V7C_INPUT_SCALAR_COUNT             114u
#define CSI_LINK_V7C_ACTIVE_AMPLITUDE_COUNT         56u
#define CSI_LINK_V7C_ADJACENT_PAIR_COUNT            54u

/* Flattened feature-vector layout. */
#define CSI_LINK_V7C_AMPLITUDE_OFFSET               0u
#define CSI_LINK_V7C_PHASE_REAL_OFFSET              56u
#define CSI_LINK_V7C_PHASE_IMAG_OFFSET              110u
#define CSI_LINK_V7C_VALID_FRACTION_INDEX           164u
#define CSI_LINK_V7C_COHERENCE_INDEX                165u
#define CSI_LINK_V7C_FEATURE_COUNT                  166u

/* Phase is trusted only when both bins in a pair meet this adaptive floor. */
#define CSI_LINK_V7C_MIN_AMPLITUDE_THRESHOLD        2.0f
#define CSI_LINK_V7C_MEDIAN_THRESHOLD_SCALE         0.10f

typedef enum {
    CSI_LINK_V7C_STATUS_OK = 0,
    CSI_LINK_V7C_STATUS_NULL_ARGUMENT = -1,
    CSI_LINK_V7C_STATUS_BAD_INPUT_SIZE = -2,
    CSI_LINK_V7C_STATUS_BAD_OUTPUT_SIZE = -3,
    CSI_LINK_V7C_STATUS_NONFINITE_RESULT = -4
} csi_link_v7c_status_t;

typedef struct {
    /* Amplitudes for bins 0..27 followed by bins 29..56. */
    float amps[CSI_LINK_V7C_ACTIVE_AMPLITUDE_COUNT];

    /*
     * Corrected adjacent phase differences for pairs 0-1 .. 26-27,
     * followed by 29-30 .. 55-56.  Invalid pairs are represented by
     * (0, 0).
     */
    float phase_real[CSI_LINK_V7C_ADJACENT_PAIR_COUNT];
    float phase_imag[CSI_LINK_V7C_ADJACENT_PAIR_COUNT];

    /* Fraction of the 54 pairs passing the amplitude threshold. */
    float valid_fraction;

    /* Magnitude of the weighted mean differential phasor, in [0, 1]. */
    float coherence;
} csi_link_v7c_features_t;

/* Link-visible copy of CSI_LINK_V7C_CONTRACT_ID for manifests/tests. */
extern const char csi_link_v7c_contract_id[];

/*
 * Compute link_v7c features from compensated [imaginary, real] int16 data.
 *
 * An all-zero or otherwise phase-invalid frame is valid input: amplitudes and
 * phase features are zero and both quality features are zero.  When `out` is
 * non-NULL it is cleared before input validation, so an error cannot expose a
 * partially populated feature structure.
 */
csi_link_v7c_status_t csi_link_v7c_compute(
    const int16_t *compensated_iq,
    size_t scalar_count,
    csi_link_v7c_features_t *out);

/* Flatten in contract order: amps, phase_real, phase_imag, quality pair. */
csi_link_v7c_status_t csi_link_v7c_flatten(
    const csi_link_v7c_features_t *features,
    float *flat,
    size_t flat_count);

/* Convenience entry point that computes and flattens in one call. */
csi_link_v7c_status_t csi_link_v7c_compute_flat(
    const int16_t *compensated_iq,
    size_t scalar_count,
    float *flat,
    size_t flat_count);

#ifdef __cplusplus
}
#endif

#endif /* CSI_LINK_V7C_FEATURES_H */
