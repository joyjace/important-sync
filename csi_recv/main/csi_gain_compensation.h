/*
 * Deterministic CSI gain compensation shared by live firmware and host tests.
 */

#ifndef CSI_GAIN_COMPENSATION_H
#define CSI_GAIN_COMPENSATION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CSI_GAIN_STATUS_OK = 0,
    CSI_GAIN_STATUS_NULL_ARGUMENT = -1,
    CSI_GAIN_STATUS_BAD_GAIN = -2,
    CSI_GAIN_STATUS_OVERFLOW = -3
} csi_gain_status_t;

/* Return the endian-independent IEEE-754 binary32 bit pattern of gain. */
uint32_t csi_gain_f32_bits(float gain);

/* Reconstruct a float from its endian-independent IEEE-754 bit pattern. */
float csi_gain_f32_from_bits(uint32_t bits);

/*
 * Match the receiver expression `(int16_t)(gain * (float)raw_value)` while
 * rejecting non-finite/non-positive gains and signed-int16 overflow.
 */
csi_gain_status_t csi_gain_compensate_i8(
    float gain,
    int8_t raw_value,
    int16_t *compensated_value);

/* Compensate one complete raw CSI scalar array. Output is zeroed on error. */
csi_gain_status_t csi_gain_compensate_frame_i8(
    float gain,
    const int8_t *raw_values,
    size_t scalar_count,
    int16_t *compensated_values);

#ifdef __cplusplus
}
#endif

#endif /* CSI_GAIN_COMPENSATION_H */
