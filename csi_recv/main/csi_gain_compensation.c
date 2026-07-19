/* Portable C99 implementation of deterministic CSI gain compensation. */

#include "csi_gain_compensation.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

_Static_assert(sizeof(float) == sizeof(uint32_t), "CSI gain protocol requires 32-bit float");
_Static_assert(FLT_RADIX == 2, "CSI gain protocol requires a binary float representation");
_Static_assert(FLT_MANT_DIG == 24, "CSI gain protocol requires IEEE-754 binary32 precision");
_Static_assert(FLT_MAX_EXP == 128, "CSI gain protocol requires IEEE-754 binary32 range");
_Static_assert(FLT_EVAL_METHOD == 0, "CSI gain compensation requires binary32 evaluation");

uint32_t csi_gain_f32_bits(float gain)
{
    uint32_t bits = 0u;
    memcpy(&bits, &gain, sizeof(bits));
    return bits;
}

float csi_gain_f32_from_bits(uint32_t bits)
{
    float gain = 0.0f;
    memcpy(&gain, &bits, sizeof(gain));
    return gain;
}

csi_gain_status_t csi_gain_compensate_i8(
    float gain,
    int8_t raw_value,
    int16_t *compensated_value)
{
    float scaled;

    if (compensated_value == NULL) {
        return CSI_GAIN_STATUS_NULL_ARGUMENT;
    }
    *compensated_value = 0;
    if (!isfinite(gain) || gain <= 0.0f) {
        return CSI_GAIN_STATUS_BAD_GAIN;
    }

    /* Both operands and the result remain binary32 on the ESP target. */
    scaled = gain * (float)raw_value;
    /* Validate the value after C's truncation toward zero. Values such as
     * 32767.5f still convert safely to 32767, while 32768.0f does not. */
    if (!isfinite(scaled) ||
        scaled >= ((float)INT16_MAX + 1.0f) ||
        scaled <= ((float)INT16_MIN - 1.0f)) {
        return CSI_GAIN_STATUS_OVERFLOW;
    }
    *compensated_value = (int16_t)scaled; /* C truncates toward zero. */
    return CSI_GAIN_STATUS_OK;
}

csi_gain_status_t csi_gain_compensate_frame_i8(
    float gain,
    const int8_t *raw_values,
    size_t scalar_count,
    int16_t *compensated_values)
{
    size_t index;

    if (raw_values == NULL || compensated_values == NULL) {
        return CSI_GAIN_STATUS_NULL_ARGUMENT;
    }
    memset(compensated_values, 0, scalar_count * sizeof(*compensated_values));
    if (!isfinite(gain) || gain <= 0.0f) {
        return CSI_GAIN_STATUS_BAD_GAIN;
    }
    for (index = 0u; index < scalar_count; ++index) {
        csi_gain_status_t status = csi_gain_compensate_i8(
            gain, raw_values[index], &compensated_values[index]);
        if (status != CSI_GAIN_STATUS_OK) {
            memset(compensated_values, 0, scalar_count * sizeof(*compensated_values));
            return status;
        }
    }
    return CSI_GAIN_STATUS_OK;
}
