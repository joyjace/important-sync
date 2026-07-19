/* Portable receiver-only state assembler for link_v7c models. */

#ifndef CSI_LINK_V7C_STATE_H
#define CSI_LINK_V7C_STATE_H

#include <stddef.h>
#include <stdint.h>

#include "csi_link_v7c_features.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CSI_LINK_V7C_STATE_CONTRACT_ID \
    "link_v7c_receiveronly_v1"
#define CSI_LINK_V7C_STATE_CONTRACT_SHA256 \
    "8c88cd04a5e2c28366f567a5928e8df6df685f05284e5683a6a4d3cfd0beb790"
#define CSI_LINK_V7C_LINK_FEATURE_COUNT 15u
#define CSI_LINK_V7C_STATE_FEATURE_COUNT \
    (CSI_LINK_V7C_FEATURE_COUNT + CSI_LINK_V7C_LINK_FEATURE_COUNT)

typedef struct {
    float rssi;
    float snr;
    float fft_gain;
    float agc_gain;
    float channel;
    float sig_len;
    uint16_t state_age_packets;
    uint16_t state_packet_gap;
    uint16_t state_missing_packets;
    uint8_t state_is_stale;
} csi_link_v7c_state_metadata_t;

typedef enum {
    CSI_LINK_V7C_STATE_STATUS_OK = 0,
    CSI_LINK_V7C_STATE_STATUS_NULL_ARGUMENT = -1,
    CSI_LINK_V7C_STATE_STATUS_BAD_INPUT_SIZE = -2,
    CSI_LINK_V7C_STATE_STATUS_BAD_OUTPUT_SIZE = -3,
    CSI_LINK_V7C_STATE_STATUS_FEATURE_ERROR = -4,
    CSI_LINK_V7C_STATE_STATUS_NONFINITE_RESULT = -5
} csi_link_v7c_state_status_t;

extern const char csi_link_v7c_state_contract_id[];
extern const char csi_link_v7c_state_contract_sha256[];

/*
 * Assemble exactly 181 receiver-only state values:
 * 166 contract CSI features followed by 15 causal receiver/link features.
 */
csi_link_v7c_state_status_t csi_link_v7c_build_receiver_state(
    const int16_t *compensated_iq,
    size_t scalar_count,
    const csi_link_v7c_state_metadata_t *metadata,
    float *state,
    size_t state_count);

#ifdef __cplusplus
}
#endif

#endif /* CSI_LINK_V7C_STATE_H */
