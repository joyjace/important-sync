/* Portable C99 link_v7c receiver-only state assembler. */

#include "csi_link_v7c_state.h"

#include <math.h>
#include <string.h>

const char csi_link_v7c_state_contract_id[] = CSI_LINK_V7C_STATE_CONTRACT_ID;
const char csi_link_v7c_state_contract_sha256[] = CSI_LINK_V7C_STATE_CONTRACT_SHA256;

static void state_sort(float *values, size_t count)
{
    for (size_t i = 1u; i < count; ++i) {
        const float key = values[i];
        size_t j = i;
        while (j > 0u && values[j - 1u] > key) {
            values[j] = values[j - 1u];
            --j;
        }
        values[j] = key;
    }
}

static float state_quantile(const float *sorted, size_t count, float quantile)
{
    const float position = quantile * (float)(count - 1u);
    const size_t lower = (size_t)position;
    size_t upper = lower + 1u;
    if (upper >= count) {
        upper = count - 1u;
    }
    const float fraction = position - (float)lower;
    return sorted[lower] * (1.0f - fraction) + sorted[upper] * fraction;
}

csi_link_v7c_state_status_t csi_link_v7c_build_receiver_state(
    const int16_t *compensated_iq,
    size_t scalar_count,
    const csi_link_v7c_state_metadata_t *metadata,
    float *state,
    size_t state_count)
{
    const size_t base = CSI_LINK_V7C_FEATURE_COUNT;
    float sorted[CSI_LINK_V7C_ACTIVE_AMPLITUDE_COUNT];
    float sum = 0.0f;
    float sum_sq = 0.0f;
    float mean;
    float variance;

    if (state == NULL) {
        return CSI_LINK_V7C_STATE_STATUS_NULL_ARGUMENT;
    }
    if (state_count != CSI_LINK_V7C_STATE_FEATURE_COUNT) {
        return CSI_LINK_V7C_STATE_STATUS_BAD_OUTPUT_SIZE;
    }
    memset(state, 0, state_count * sizeof(*state));
    if (compensated_iq == NULL || metadata == NULL) {
        return CSI_LINK_V7C_STATE_STATUS_NULL_ARGUMENT;
    }
    if (scalar_count != CSI_LINK_V7C_INPUT_SCALAR_COUNT) {
        return CSI_LINK_V7C_STATE_STATUS_BAD_INPUT_SIZE;
    }
    if (csi_link_v7c_compute_flat(
            compensated_iq,
            scalar_count,
            state,
            CSI_LINK_V7C_FEATURE_COUNT) != CSI_LINK_V7C_STATUS_OK) {
        memset(state, 0, state_count * sizeof(*state));
        return CSI_LINK_V7C_STATE_STATUS_FEATURE_ERROR;
    }

    for (size_t i = 0u; i < CSI_LINK_V7C_ACTIVE_AMPLITUDE_COUNT; ++i) {
        const float amplitude = state[i];
        sum += amplitude;
        sum_sq += amplitude * amplitude;
        sorted[i] = amplitude;
    }
    mean = sum / (float)CSI_LINK_V7C_ACTIVE_AMPLITUDE_COUNT;
    variance = sum_sq / (float)CSI_LINK_V7C_ACTIVE_AMPLITUDE_COUNT - mean * mean;
    if (variance < 0.0f) {
        variance = 0.0f;
    }
    state_sort(sorted, CSI_LINK_V7C_ACTIVE_AMPLITUDE_COUNT);

    state[base + 0u] = metadata->rssi;
    state[base + 1u] = metadata->snr;
    state[base + 2u] = metadata->fft_gain;
    state[base + 3u] = metadata->agc_gain;
    state[base + 4u] = metadata->channel;
    state[base + 5u] = metadata->sig_len;
    state[base + 6u] = log1pf((float)metadata->state_age_packets);
    state[base + 7u] = log1pf((float)metadata->state_packet_gap);
    state[base + 8u] = log1pf((float)metadata->state_missing_packets);
    state[base + 9u] = metadata->state_is_stale ? 1.0f : 0.0f;
    state[base + 10u] = mean;
    state[base + 11u] = sqrtf(variance);
    state[base + 12u] = state_quantile(
        sorted, CSI_LINK_V7C_ACTIVE_AMPLITUDE_COUNT, 0.10f);
    state[base + 13u] = state_quantile(
        sorted, CSI_LINK_V7C_ACTIVE_AMPLITUDE_COUNT, 0.50f);
    state[base + 14u] = state_quantile(
        sorted, CSI_LINK_V7C_ACTIVE_AMPLITUDE_COUNT, 0.90f);

    for (size_t i = 0u; i < state_count; ++i) {
        if (!isfinite(state[i])) {
            memset(state, 0, state_count * sizeof(*state));
            return CSI_LINK_V7C_STATE_STATUS_NONFINITE_RESULT;
        }
    }
    return CSI_LINK_V7C_STATE_STATUS_OK;
}
