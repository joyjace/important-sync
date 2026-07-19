/*
 * Portable C99 reference transform for the link_v7c CSI feature contract.
 */

#include "csi_link_v7c_features.h"

#include <math.h>
#include <string.h>

const char csi_link_v7c_contract_id[] = CSI_LINK_V7C_CONTRACT_ID;

static void sort_floats(float *values, size_t count)
{
    size_t i;

    for (i = 1u; i < count; ++i) {
        const float key = values[i];
        size_t j = i;

        while (j > 0u && values[j - 1u] > key) {
            values[j] = values[j - 1u];
            --j;
        }
        values[j] = key;
    }
}

static float median_active_amplitude(const float *active_amps)
{
    float sorted[CSI_LINK_V7C_ACTIVE_AMPLITUDE_COUNT];
    const size_t upper = CSI_LINK_V7C_ACTIVE_AMPLITUDE_COUNT / 2u;

    memcpy(sorted, active_amps, sizeof(sorted));
    sort_floats(sorted, CSI_LINK_V7C_ACTIVE_AMPLITUDE_COUNT);

    /* There are exactly 56 values, so the median is the middle-pair mean. */
    return 0.5f * (sorted[upper - 1u] + sorted[upper]);
}

static int all_features_finite(const csi_link_v7c_features_t *features)
{
    size_t i;

    for (i = 0u; i < CSI_LINK_V7C_ACTIVE_AMPLITUDE_COUNT; ++i) {
        if (!isfinite(features->amps[i])) {
            return 0;
        }
    }
    for (i = 0u; i < CSI_LINK_V7C_ADJACENT_PAIR_COUNT; ++i) {
        if (!isfinite(features->phase_real[i]) ||
            !isfinite(features->phase_imag[i])) {
            return 0;
        }
    }

    return isfinite(features->valid_fraction) &&
           isfinite(features->coherence);
}

csi_link_v7c_status_t csi_link_v7c_compute(
    const int16_t *compensated_iq,
    size_t scalar_count,
    csi_link_v7c_features_t *out)
{
    float bin_real[CSI_LINK_V7C_BIN_COUNT];
    float bin_imag[CSI_LINK_V7C_BIN_COUNT];
    float bin_amp[CSI_LINK_V7C_BIN_COUNT];
    float raw_real[CSI_LINK_V7C_ADJACENT_PAIR_COUNT] = {0.0f};
    float raw_imag[CSI_LINK_V7C_ADJACENT_PAIR_COUNT] = {0.0f};
    float pair_weight[CSI_LINK_V7C_ADJACENT_PAIR_COUNT] = {0.0f};
    float threshold;
    float common_real = 0.0f;
    float common_imag = 0.0f;
    float weight_sum = 0.0f;
    float common_magnitude;
    float common_unit_real = 1.0f;
    float common_unit_imag = 0.0f;
    size_t active_index = 0u;
    size_t pair_index = 0u;
    size_t valid_count = 0u;
    size_t bin;
    size_t i;

    if (out == NULL) {
        return CSI_LINK_V7C_STATUS_NULL_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));

    if (compensated_iq == NULL) {
        return CSI_LINK_V7C_STATUS_NULL_ARGUMENT;
    }
    if (scalar_count != CSI_LINK_V7C_INPUT_SCALAR_COUNT) {
        return CSI_LINK_V7C_STATUS_BAD_INPUT_SIZE;
    }

    for (bin = 0u; bin < CSI_LINK_V7C_BIN_COUNT; ++bin) {
        const float imag = (float)compensated_iq[2u * bin];
        const float real = (float)compensated_iq[2u * bin + 1u];
        const float amp = sqrtf(real * real + imag * imag);

        bin_real[bin] = real;
        bin_imag[bin] = imag;
        bin_amp[bin] = amp;

        if (bin != CSI_LINK_V7C_DC_BIN_INDEX) {
            out->amps[active_index] = amp;
            ++active_index;
        }
    }

    threshold = CSI_LINK_V7C_MEDIAN_THRESHOLD_SCALE *
                median_active_amplitude(out->amps);
    if (threshold < CSI_LINK_V7C_MIN_AMPLITUDE_THRESHOLD) {
        threshold = CSI_LINK_V7C_MIN_AMPLITUDE_THRESHOLD;
    }

    /* Lower side: pairs 0-1 through 26-27. */
    for (bin = 0u; bin + 1u < CSI_LINK_V7C_DC_BIN_INDEX; ++bin) {
        const size_t next = bin + 1u;

        if (bin_amp[bin] >= threshold && bin_amp[next] >= threshold) {
            const float denominator = bin_amp[bin] * bin_amp[next];
            const float d_real =
                (bin_real[next] * bin_real[bin] +
                 bin_imag[next] * bin_imag[bin]) / denominator;
            const float d_imag =
                (bin_imag[next] * bin_real[bin] -
                 bin_real[next] * bin_imag[bin]) / denominator;
            const float weight = bin_amp[bin] < bin_amp[next]
                                     ? bin_amp[bin]
                                     : bin_amp[next];

            raw_real[pair_index] = d_real;
            raw_imag[pair_index] = d_imag;
            pair_weight[pair_index] = weight;
            common_real += weight * d_real;
            common_imag += weight * d_imag;
            weight_sum += weight;
            ++valid_count;
        }
        ++pair_index;
    }

    /* Upper side: pairs 29-30 through 55-56. */
    for (bin = CSI_LINK_V7C_DC_BIN_INDEX + 1u;
         bin + 1u < CSI_LINK_V7C_BIN_COUNT;
         ++bin) {
        const size_t next = bin + 1u;

        if (bin_amp[bin] >= threshold && bin_amp[next] >= threshold) {
            const float denominator = bin_amp[bin] * bin_amp[next];
            const float d_real =
                (bin_real[next] * bin_real[bin] +
                 bin_imag[next] * bin_imag[bin]) / denominator;
            const float d_imag =
                (bin_imag[next] * bin_real[bin] -
                 bin_real[next] * bin_imag[bin]) / denominator;
            const float weight = bin_amp[bin] < bin_amp[next]
                                     ? bin_amp[bin]
                                     : bin_amp[next];

            raw_real[pair_index] = d_real;
            raw_imag[pair_index] = d_imag;
            pair_weight[pair_index] = weight;
            common_real += weight * d_real;
            common_imag += weight * d_imag;
            weight_sum += weight;
            ++valid_count;
        }
        ++pair_index;
    }

    common_magnitude = sqrtf(common_real * common_real +
                             common_imag * common_imag);
    if (weight_sum > 0.0f && common_magnitude > 0.0f &&
        isfinite(common_magnitude)) {
        common_unit_real = common_real / common_magnitude;
        common_unit_imag = common_imag / common_magnitude;
        out->coherence = common_magnitude / weight_sum;
        if (out->coherence > 1.0f) {
            out->coherence = 1.0f;
        } else if (out->coherence < 0.0f) {
            out->coherence = 0.0f;
        }
    }

    /* Multiply each valid D by the conjugate of the common unit phasor. */
    for (i = 0u; i < CSI_LINK_V7C_ADJACENT_PAIR_COUNT; ++i) {
        if (pair_weight[i] > 0.0f) {
            out->phase_real[i] =
                raw_real[i] * common_unit_real +
                raw_imag[i] * common_unit_imag;
            out->phase_imag[i] =
                raw_imag[i] * common_unit_real -
                raw_real[i] * common_unit_imag;
        }
    }

    out->valid_fraction =
        (float)valid_count / (float)CSI_LINK_V7C_ADJACENT_PAIR_COUNT;

    if (!all_features_finite(out)) {
        memset(out, 0, sizeof(*out));
        return CSI_LINK_V7C_STATUS_NONFINITE_RESULT;
    }

    return CSI_LINK_V7C_STATUS_OK;
}

csi_link_v7c_status_t csi_link_v7c_flatten(
    const csi_link_v7c_features_t *features,
    float *flat,
    size_t flat_count)
{
    if (features == NULL || flat == NULL) {
        return CSI_LINK_V7C_STATUS_NULL_ARGUMENT;
    }
    if (flat_count != CSI_LINK_V7C_FEATURE_COUNT) {
        return CSI_LINK_V7C_STATUS_BAD_OUTPUT_SIZE;
    }
    if (!all_features_finite(features)) {
        memset(flat, 0, sizeof(*flat) * flat_count);
        return CSI_LINK_V7C_STATUS_NONFINITE_RESULT;
    }

    memcpy(flat + CSI_LINK_V7C_AMPLITUDE_OFFSET,
           features->amps,
           sizeof(features->amps));
    memcpy(flat + CSI_LINK_V7C_PHASE_REAL_OFFSET,
           features->phase_real,
           sizeof(features->phase_real));
    memcpy(flat + CSI_LINK_V7C_PHASE_IMAG_OFFSET,
           features->phase_imag,
           sizeof(features->phase_imag));
    flat[CSI_LINK_V7C_VALID_FRACTION_INDEX] = features->valid_fraction;
    flat[CSI_LINK_V7C_COHERENCE_INDEX] = features->coherence;

    return CSI_LINK_V7C_STATUS_OK;
}

csi_link_v7c_status_t csi_link_v7c_compute_flat(
    const int16_t *compensated_iq,
    size_t scalar_count,
    float *flat,
    size_t flat_count)
{
    csi_link_v7c_features_t features;
    csi_link_v7c_status_t status;

    if (flat == NULL) {
        return CSI_LINK_V7C_STATUS_NULL_ARGUMENT;
    }
    if (flat_count != CSI_LINK_V7C_FEATURE_COUNT) {
        return CSI_LINK_V7C_STATUS_BAD_OUTPUT_SIZE;
    }
    memset(flat, 0, sizeof(*flat) * flat_count);

    status = csi_link_v7c_compute(compensated_iq, scalar_count, &features);
    if (status != CSI_LINK_V7C_STATUS_OK) {
        return status;
    }

    return csi_link_v7c_flatten(&features, flat, flat_count);
}
