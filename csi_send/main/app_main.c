/*
 * SPDX-FileCopyrightText: 2025-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
/* Get Start Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>

#include "nvs_flash.h"

#include "esp_mac.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

/* ESP-IDF 6.0+ renamed WIFI_BW_HT20/HT40 to WIFI_BW20/BW40. */
#ifndef WIFI_BW_HT20
#define WIFI_BW_HT20 WIFI_BW20
#endif
#ifndef WIFI_BW_HT40
#define WIFI_BW_HT40 WIFI_BW40
#endif

#define CONFIG_LESS_INTERFERENCE_CHANNEL   1 

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61 || (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0))
#define CONFIG_WIFI_BAND_MODE               WIFI_BAND_MODE_2G_ONLY
#define CONFIG_WIFI_2G_BANDWIDTHS           WIFI_BW_HT20
#define CONFIG_WIFI_5G_BANDWIDTHS           WIFI_BW_HT20
#define CONFIG_WIFI_2G_PROTOCOL             WIFI_PROTOCOL_11N
#define CONFIG_WIFI_5G_PROTOCOL             WIFI_PROTOCOL_11N
#else
#define CONFIG_WIFI_BANDWIDTH               WIFI_BW_HT20
#endif

#define CONFIG_ESP_NOW_PHYMODE           WIFI_PHY_MODE_HT20
#define CONFIG_ESP_NOW_RATE             WIFI_PHY_RATE_MCS0_LGI 
/* Bitrate options (choose one for `CONFIG_ESP_NOW_RATE` - type `wifi_phy_rate_t`):
 *
 * Legacy (802.11b/g):
 *  - WIFI_PHY_RATE_1M_L, WIFI_PHY_RATE_2M_L, WIFI_PHY_RATE_5M5_L, WIFI_PHY_RATE_11M_L
 *    : Very low rates (1 / 2 / 5.5 / 11 Mbps) - most robust, lowest throughput.
 *  - WIFI_PHY_RATE_6M .. WIFI_PHY_RATE_54M
 *    : OFDM legacy rates (6 to 54 Mbps) - moderate throughput/reliability trade-offs.
 *
 * HT MCS (802.11n) - HT20 / HT40 with Long/Short Guard Interval variants:
 *  - WIFI_PHY_RATE_MCS0_LGI .. WIFI_PHY_RATE_MCS7_LGI
 *    : MCS0..MCS7 with Long GI. Example (HT20): MCS0=6.5 Mbps, MCS7=65 Mbps.
 *  - WIFI_PHY_RATE_MCS0_SGI .. WIFI_PHY_RATE_MCS7_SGI
 *    : MCS0..MCS7 with Short GI (slightly higher throughput, slightly less robust).
 *
 * HT40 (40 MHz) doubles many HT20 rates roughly (e.g. MCS0_LGI ~13.5 Mbps for HT40).
 *
 * Notes:
 *  - Higher MCS number => higher throughput but requires better signal/SNR.
 *  - LGI vs SGI: SGI gives a small throughput boost (~10%) but can be less stable.
 *  - Choose legacy rates for robustness, MCS rates for higher throughput when link is good.
 */
#define CONFIG_SEND_FREQUENCY               200
#define CONFIG_PACKET_PACING_ENABLED        1  // 1 = paced by CONFIG_SEND_FREQUENCY, 0 = max-rate (no fixed sleep)
#define CONFIG_ACK_TIMING_MODE              2  // 1 = async pipeline, 2 = stop-and-wait (one packet in flight)
#define CONFIG_RATE_SWITCH_MODE             0  // 0 = TIME_BASED, 1 = PACKET_BASED, 2 = STATIC (fixed rate, no switching)
#define CONFIG_RATE_SWITCH_INTERVAL_SEC     10 // Used when TIME_BASED
#define CONFIG_RATE_SWITCH_PACKET_COUNT     1 // Used when PACKET_BASED
#ifndef CONFIG_CSI_LIVE_MCS_SELECTION_ENABLED
#define CONFIG_CSI_LIVE_MCS_SELECTION_ENABLED 1
#endif
#ifndef CONFIG_CSI_LIVE_MCS_ALGO
#define CONFIG_CSI_LIVE_MCS_ALGO 1
#endif
#ifndef CONFIG_CSI_LIVE_MCS_MIN_INDEX
#define CONFIG_CSI_LIVE_MCS_MIN_INDEX 0
#endif
#ifndef CONFIG_CSI_LIVE_MCS_MAX_INDEX
#define CONFIG_CSI_LIVE_MCS_MAX_INDEX 7
#endif
#ifndef CONFIG_CSI_MINSTREL_UPDATE_EVERY_PKTS
#define CONFIG_CSI_MINSTREL_UPDATE_EVERY_PKTS 20
#endif
#ifndef CONFIG_CSI_MINSTREL_PROBE_EVERY_PKTS
#define CONFIG_CSI_MINSTREL_PROBE_EVERY_PKTS 10
#endif
#ifndef CONFIG_CSI_MINSTREL_EWMA_ALPHA_NUM
#define CONFIG_CSI_MINSTREL_EWMA_ALPHA_NUM 1
#endif
#ifndef CONFIG_CSI_MINSTREL_EWMA_ALPHA_DEN
#define CONFIG_CSI_MINSTREL_EWMA_ALPHA_DEN 4
#endif
#ifndef CONFIG_CSI_CUSTOM_POLICY_DEFAULT_MCS
#define CONFIG_CSI_CUSTOM_POLICY_DEFAULT_MCS 3
#endif
#ifndef CONFIG_CSI_LIVE_MCS_DECISION_LOG_ENABLED
#define CONFIG_CSI_LIVE_MCS_DECISION_LOG_ENABLED 1
#endif
#ifndef CONFIG_CSI_REMOTE_MCS_RECOMMENDATION_ENABLED
#define CONFIG_CSI_REMOTE_MCS_RECOMMENDATION_ENABLED 1
#endif
#ifndef CONFIG_CSI_REMOTE_MCS_MIN_CONFIDENCE
#define CONFIG_CSI_REMOTE_MCS_MIN_CONFIDENCE 0
#endif
#ifndef CONFIG_CSI_REMOTE_MCS_MAX_AGE_MS
#define CONFIG_CSI_REMOTE_MCS_MAX_AGE_MS 300
#endif
#ifndef CONFIG_CSI_DQN_REMOTE_RECOMMENDATION_ENABLED
#define CONFIG_CSI_DQN_REMOTE_RECOMMENDATION_ENABLED 0
#endif
#ifndef CONFIG_CSI_DQN_DEFAULT_MCS
#define CONFIG_CSI_DQN_DEFAULT_MCS 4
#endif
#ifndef CONFIG_CSI_DQN_REMOTE_MIN_CONFIDENCE
#define CONFIG_CSI_DQN_REMOTE_MIN_CONFIDENCE 0
#endif
#ifndef CONFIG_CSI_DQN_REMOTE_MAX_AGE_MS
#define CONFIG_CSI_DQN_REMOTE_MAX_AGE_MS 50
#endif
#ifndef CONFIG_CSI_DQN_REMOTE_MAX_SEQ_GAP
#define CONFIG_CSI_DQN_REMOTE_MAX_SEQ_GAP 0
#endif
#ifndef CONFIG_CSI_DQN_FAILURE_STEPDOWN_ENABLED
#define CONFIG_CSI_DQN_FAILURE_STEPDOWN_ENABLED 0
#endif
#ifndef CONFIG_CSI_DQN_FAILURE_STEPDOWN_COUNT
#define CONFIG_CSI_DQN_FAILURE_STEPDOWN_COUNT 3
#endif
#ifndef CONFIG_CSI_DQN_LOG_ENABLED
#define CONFIG_CSI_DQN_LOG_ENABLED 1
#endif
//test
#define CONFIG_LIVE_MCS_SELECTION_ENABLED   CONFIG_CSI_LIVE_MCS_SELECTION_ENABLED
#define CONFIG_LIVE_MCS_ALGO                CONFIG_CSI_LIVE_MCS_ALGO
#define CONFIG_LIVE_MCS_MIN_INDEX           CONFIG_CSI_LIVE_MCS_MIN_INDEX
#define CONFIG_LIVE_MCS_MAX_INDEX           CONFIG_CSI_LIVE_MCS_MAX_INDEX
#define CONFIG_MINSTREL_UPDATE_EVERY_PKTS   CONFIG_CSI_MINSTREL_UPDATE_EVERY_PKTS
#define CONFIG_MINSTREL_PROBE_EVERY_PKTS    CONFIG_CSI_MINSTREL_PROBE_EVERY_PKTS
#define CONFIG_MINSTREL_EWMA_ALPHA_NUM      CONFIG_CSI_MINSTREL_EWMA_ALPHA_NUM
#define CONFIG_MINSTREL_EWMA_ALPHA_DEN      CONFIG_CSI_MINSTREL_EWMA_ALPHA_DEN
#define CONFIG_CUSTOM_POLICY_DEFAULT_MCS    CONFIG_CSI_CUSTOM_POLICY_DEFAULT_MCS
#define CONFIG_LIVE_MCS_DECISION_LOG_ENABLED CONFIG_CSI_LIVE_MCS_DECISION_LOG_ENABLED
#define CONFIG_REMOTE_MCS_RECOMMENDATION_ENABLED CONFIG_CSI_REMOTE_MCS_RECOMMENDATION_ENABLED
#define CONFIG_REMOTE_MCS_MIN_CONFIDENCE CONFIG_CSI_REMOTE_MCS_MIN_CONFIDENCE
#define CONFIG_REMOTE_MCS_MAX_AGE_MS CONFIG_CSI_REMOTE_MCS_MAX_AGE_MS
#define CONFIG_DQN_REMOTE_RECOMMENDATION_ENABLED CONFIG_CSI_DQN_REMOTE_RECOMMENDATION_ENABLED
#define CONFIG_DQN_DEFAULT_MCS CONFIG_CSI_DQN_DEFAULT_MCS
#define CONFIG_DQN_REMOTE_MIN_CONFIDENCE CONFIG_CSI_DQN_REMOTE_MIN_CONFIDENCE
#define CONFIG_DQN_REMOTE_MAX_AGE_MS CONFIG_CSI_DQN_REMOTE_MAX_AGE_MS
#define CONFIG_DQN_REMOTE_MAX_SEQ_GAP CONFIG_CSI_DQN_REMOTE_MAX_SEQ_GAP
#define CONFIG_DQN_FAILURE_STEPDOWN_ENABLED CONFIG_CSI_DQN_FAILURE_STEPDOWN_ENABLED
#define CONFIG_DQN_FAILURE_STEPDOWN_COUNT CONFIG_CSI_DQN_FAILURE_STEPDOWN_COUNT
#define CONFIG_DQN_LOG_ENABLED CONFIG_CSI_DQN_LOG_ENABLED
#define CONFIG_ESP_NOW_PAYLOAD_LEN          128 // Bytes per ESP-NOW data frame (>= 4 to keep sequence ID) (16, 64, 128)
#define CONFIG_IDENTICAL_TX_PAYLOAD         0  // 1 = send the same ESP-NOW body every packet; seq stays only in local ACK logs
// TX power in units of 0.25 dBm. Range [8, 84] => [2 dBm, 20 dBm].
// Mapping: {set value range, actual value} = {{[8,19],8},{[20,27],20},{[28,33],28},{[34,43],34},{[44,51],44},{[52,55],52},{[56,59],56},{[60,65],60},{[66,71],66},{[72,79],72},{[80,84],80}}
#define CONFIG_WIFI_TX_POWER                8

#if CONFIG_LESS_INTERFERENCE_CHANNEL < 1 || CONFIG_LESS_INTERFERENCE_CHANNEL > 13
#error "CONFIG_LESS_INTERFERENCE_CHANNEL must be in [1, 13] for 2.4 GHz"
#endif

#if !CONFIG_IDENTICAL_TX_PAYLOAD && CONFIG_ESP_NOW_PAYLOAD_LEN < 4
#error "CONFIG_ESP_NOW_PAYLOAD_LEN must be at least 4 bytes"
#endif

#if CONFIG_ESP_NOW_PAYLOAD_LEN > ESP_NOW_MAX_DATA_LEN
#error "CONFIG_ESP_NOW_PAYLOAD_LEN exceeds ESP_NOW_MAX_DATA_LEN"
#endif

#if CONFIG_PACKET_PACING_ENABLED
#if CONFIG_SEND_FREQUENCY <= 0
#error "CONFIG_SEND_FREQUENCY must be > 0 when CONFIG_PACKET_PACING_ENABLED is 1"
#endif
#endif

#if CONFIG_LIVE_MCS_MIN_INDEX < 0 || CONFIG_LIVE_MCS_MIN_INDEX > 7
#error "CONFIG_LIVE_MCS_MIN_INDEX must be in [0, 7]"
#endif

#if CONFIG_LIVE_MCS_MAX_INDEX < 0 || CONFIG_LIVE_MCS_MAX_INDEX > 7
#error "CONFIG_LIVE_MCS_MAX_INDEX must be in [0, 7]"
#endif

#if CONFIG_LIVE_MCS_MIN_INDEX > CONFIG_LIVE_MCS_MAX_INDEX
#error "CONFIG_LIVE_MCS_MIN_INDEX must be <= CONFIG_LIVE_MCS_MAX_INDEX"
#endif

#if CONFIG_LIVE_MCS_ALGO < 0 || CONFIG_LIVE_MCS_ALGO > 2
#error "CONFIG_LIVE_MCS_ALGO must be 0 (Minstrel), 1 (Custom), or 2 (DQN)"
#endif

#if CONFIG_MINSTREL_EWMA_ALPHA_NUM <= 0 || CONFIG_MINSTREL_EWMA_ALPHA_NUM > CONFIG_MINSTREL_EWMA_ALPHA_DEN
#error "CONFIG_MINSTREL_EWMA_ALPHA_NUM must be in (0, CONFIG_MINSTREL_EWMA_ALPHA_DEN]"
#endif

#if CONFIG_MINSTREL_EWMA_ALPHA_DEN <= 0
#error "CONFIG_MINSTREL_EWMA_ALPHA_DEN must be > 0"
#endif

#if CONFIG_REMOTE_MCS_MIN_CONFIDENCE < 0 || CONFIG_REMOTE_MCS_MIN_CONFIDENCE > 100
#error "CONFIG_REMOTE_MCS_MIN_CONFIDENCE must be in [0, 100]"
#endif

#if CONFIG_REMOTE_MCS_MAX_AGE_MS <= 0
#error "CONFIG_REMOTE_MCS_MAX_AGE_MS must be > 0"
#endif

#if CONFIG_DQN_DEFAULT_MCS < 0 || CONFIG_DQN_DEFAULT_MCS > 7
#error "CONFIG_DQN_DEFAULT_MCS must be in [0, 7]"
#endif

#if CONFIG_DQN_REMOTE_MIN_CONFIDENCE < 0 || CONFIG_DQN_REMOTE_MIN_CONFIDENCE > 100
#error "CONFIG_DQN_REMOTE_MIN_CONFIDENCE must be in [0, 100]"
#endif

#if CONFIG_DQN_REMOTE_MAX_AGE_MS <= 0 || CONFIG_DQN_REMOTE_MAX_SEQ_GAP < 0
#error "DQN recommendation age must be positive and sequence gap must be non-negative"
#endif

#if CONFIG_DQN_FAILURE_STEPDOWN_ENABLED && CONFIG_DQN_FAILURE_STEPDOWN_COUNT <= 0
#error "CONFIG_DQN_FAILURE_STEPDOWN_COUNT must be positive"
#endif

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define ESP_IF_WIFI_STA ESP_MAC_WIFI_STA
#endif

static const uint8_t CONFIG_CSI_SEND_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x10};
static const uint8_t CONFIG_CSI_RECV_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x11};
static const char *TAG = "csi_send";

#define MCS_RECO_MAGIC 0xA5
#define MCS_RECO_VERSION 1
#define DQN_RECO_MAGIC 0xD7
#define DQN_RECO_VERSION 1

typedef struct __attribute__((packed)) {
    uint8_t magic;
    uint8_t version;
    uint8_t recommended_mcs;
    uint8_t confidence;
    uint32_t seq;
    int8_t rssi;
    int8_t snr;
    uint8_t reserved[2];
} mcs_reco_msg_t;

typedef struct __attribute__((packed)) {
    uint8_t magic;
    uint8_t version;
    uint8_t recommended_mcs;
    uint8_t confidence;
    uint32_t seq;
    int8_t rssi;
    int8_t snr;
    uint16_t margin_milli;
} dqn_reco_msg_t;

_Static_assert(sizeof(dqn_reco_msg_t) == 12, "DQN feedback protocol size changed");

typedef struct {
    uint8_t valid;
    uint8_t recommended_mcs;
    uint8_t confidence;
    uint32_t seq;
    int8_t rssi;
    int8_t snr;
    int64_t rx_ts_us;
} remote_mcs_reco_state_t;

static remote_mcs_reco_state_t s_remote_mcs_reco = {0};
static portMUX_TYPE s_remote_mcs_reco_mux = portMUX_INITIALIZER_UNLOCKED;

typedef struct {
    uint8_t valid;
    uint8_t recommended_mcs;
    uint8_t confidence;
    uint16_t margin_milli;
    uint32_t seq;
    int8_t rssi;
    int8_t snr;
    int64_t rx_ts_us;
} dqn_remote_reco_state_t;

#if CONFIG_DQN_REMOTE_RECOMMENDATION_ENABLED
static dqn_remote_reco_state_t s_dqn_remote_reco = {0};
static portMUX_TYPE s_dqn_remote_reco_mux = portMUX_INITIALIZER_UNLOCKED;
#endif
static uint32_t s_dqn_consecutive_failures = 0;
static volatile uint8_t s_dqn_failure_stepdown_pending = 0;

static wifi_second_chan_t get_secondary_channel_for_ht40(int primary_channel)
{
    if (primary_channel <= 4) {
        return WIFI_SECOND_CHAN_ABOVE;
    }

    return WIFI_SECOND_CHAN_BELOW;
}

static inline void tx_pacing_delay_us(void)
{
#if CONFIG_PACKET_PACING_ENABLED
    usleep(1000 * 1000 / CONFIG_SEND_FREQUENCY);
#endif
}

static uint32_t s_ack_success_count  = 0;
static uint32_t s_ack_fail_count     = 0;
static uint8_t s_tx_payload[CONFIG_ESP_NOW_PAYLOAD_LEN] = {0};
static wifi_phy_rate_t s_current_configured_rate = CONFIG_ESP_NOW_RATE;

#if CONFIG_ACK_TIMING_MODE == 2
static TaskHandle_t s_sender_task_handle = NULL;
#endif

/*
 * Keep all printf()/fflush() work out of the ESP-NOW send callback. The
 * callback only captures timing/metadata, enqueues a compact record, wakes the
 * sender in stop-and-wait mode, and returns.
 */
#define ACK_LOG_QUEUE_SIZE          256
#define ACK_LOG_TASK_STACK_SIZE     4096
#define ACK_LOG_TASK_PRIORITY       (tskIDLE_PRIORITY + 1)

#define ACK_SEQ_QUEUE_SIZE          2048

#if CONFIG_RATE_SWITCH_MODE == 0
#define ACK_SERVICE_MEDIAN_MAX_SAMPLES  (CONFIG_SEND_FREQUENCY * CONFIG_RATE_SWITCH_INTERVAL_SEC * 2)
#else
#define ACK_SERVICE_MEDIAN_MAX_SAMPLES  CONFIG_RATE_SWITCH_PACKET_COUNT
#endif

typedef struct {
    uint32_t seq;
    int64_t send_ts_us;
    wifi_phy_rate_t configured_rate;
} ack_seq_item_t;

typedef enum {
    ACK_LOG_EVENT_STATUS = 0,
    ACK_LOG_EVENT_PDR_FINAL,
    ACK_LOG_EVENT_RESET,
    ACK_LOG_EVENT_CALLBACK_ERROR,
    ACK_LOG_EVENT_TRACKING_ERROR,
} ack_log_event_type_t;

typedef struct {
    ack_log_event_type_t type;
    uint32_t seq;
    uint32_t total;
    uint32_t success_count;
    uint32_t new_mcs_index;
    int delivered;
    int64_t event_ts_us;
    int64_t send_ts_us;
    int64_t service_us;
    int configured_rate;
    int actual_rate;
    int tx_status;
    int data_len;
    int used_mcs;
    int next_mcs;
    int best_mcs;
    int is_probe;
    float used_ewma_pdr;
    float used_ewma_service_us;
    float selected_reward;
} ack_log_event_t;

static ack_seq_item_t s_ack_seq_queue[ACK_SEQ_QUEUE_SIZE];
static uint16_t s_ack_seq_head = 0;
static uint16_t s_ack_seq_tail = 0;
static portMUX_TYPE s_ack_seq_mux = portMUX_INITIALIZER_UNLOCKED;
static portMUX_TYPE s_ack_stats_mux = portMUX_INITIALIZER_UNLOCKED;

static QueueHandle_t s_ack_log_queue = NULL;
static volatile uint32_t s_ack_log_drop_count = 0;
static uint32_t s_service_median_samples[ACK_SERVICE_MEDIAN_MAX_SAMPLES];
static uint32_t s_service_median_sample_count = 0;

static const wifi_phy_rate_t s_esp_now_rate_cycle[] = {
    WIFI_PHY_RATE_MCS0_LGI,
    WIFI_PHY_RATE_MCS1_LGI,
    WIFI_PHY_RATE_MCS2_LGI,
    WIFI_PHY_RATE_MCS3_LGI,
    WIFI_PHY_RATE_MCS4_LGI,
    WIFI_PHY_RATE_MCS5_LGI,
    WIFI_PHY_RATE_MCS6_LGI,
    WIFI_PHY_RATE_MCS7_LGI,
};

#if CONFIG_LIVE_MCS_ALGO == 0
static const float s_mcs_rate_mbps[] = {
    6.5f, 13.0f, 19.5f, 26.0f, 39.0f, 52.0f, 58.5f, 65.0f,
};
#endif

typedef struct {
    uint32_t attempts_window;
    uint32_t success_window;
    float ewma_pdr;
    float ewma_service_us;
    uint32_t service_samples;
} live_rate_stat_t;

typedef struct {
    uint8_t used_mcs;
    uint8_t next_mcs;
    uint8_t best_mcs;
    uint8_t is_probe;
    float used_ewma_pdr;
    float used_ewma_service_us;
    float selected_reward;
} live_policy_snapshot_t;

static live_rate_stat_t s_live_rate_stats[sizeof(s_esp_now_rate_cycle) / sizeof(s_esp_now_rate_cycle[0])];
static uint32_t s_live_controller_total_acks = 0;
static uint8_t s_live_current_mcs_index = CONFIG_LIVE_MCS_MIN_INDEX;
static uint8_t s_live_best_mcs_index = CONFIG_LIVE_MCS_MIN_INDEX;
static uint8_t s_live_probe_active = 0;

static void esp_now_set_peer_rate(const uint8_t *peer_addr, wifi_phy_rate_t rate);

static bool rate_to_mcs_index(wifi_phy_rate_t rate, uint8_t *mcs_index)
{
    for (size_t i = 0; i < sizeof(s_esp_now_rate_cycle) / sizeof(s_esp_now_rate_cycle[0]); ++i) {
        if (s_esp_now_rate_cycle[i] == rate) {
            *mcs_index = (uint8_t)i;
            return true;
        }
    }

    return false;
}

static wifi_phy_rate_t mcs_index_to_rate(uint8_t mcs_index)
{
    if (mcs_index > 7) {
        mcs_index = 7;
    }
    return s_esp_now_rate_cycle[mcs_index];
}

static uint8_t clamp_live_mcs_index(uint8_t mcs_index)
{
#if CONFIG_LIVE_MCS_MIN_INDEX > 0
    if (mcs_index < CONFIG_LIVE_MCS_MIN_INDEX) {
        return CONFIG_LIVE_MCS_MIN_INDEX;
    }
#endif
    if (mcs_index > CONFIG_LIVE_MCS_MAX_INDEX) {
        return CONFIG_LIVE_MCS_MAX_INDEX;
    }
    return mcs_index;
}

static void live_rate_controller_reset(void)
{
    memset(s_live_rate_stats, 0, sizeof(s_live_rate_stats));
    s_live_controller_total_acks = 0;
    s_live_probe_active = 0;

    uint8_t initial_mcs = CONFIG_LIVE_MCS_MIN_INDEX;
#if CONFIG_LIVE_MCS_ALGO == 1
    initial_mcs = clamp_live_mcs_index(CONFIG_CUSTOM_POLICY_DEFAULT_MCS);
#elif CONFIG_LIVE_MCS_ALGO == 2
    initial_mcs = clamp_live_mcs_index(CONFIG_DQN_DEFAULT_MCS);
#else
    uint8_t configured_mcs = 0;
    if (rate_to_mcs_index(CONFIG_ESP_NOW_RATE, &configured_mcs)) {
        initial_mcs = clamp_live_mcs_index(configured_mcs);
    }
#endif

    s_live_current_mcs_index = initial_mcs;
    s_live_best_mcs_index = initial_mcs;
    s_dqn_consecutive_failures = 0;
    s_dqn_failure_stepdown_pending = 0;

    for (uint8_t i = CONFIG_LIVE_MCS_MIN_INDEX; i <= CONFIG_LIVE_MCS_MAX_INDEX; ++i) {
        s_live_rate_stats[i].ewma_pdr = 0.5f;
        s_live_rate_stats[i].ewma_service_us = 1000.0f;
        s_live_rate_stats[i].service_samples = 0;
    }
}

#if CONFIG_LIVE_MCS_ALGO == 0
static uint8_t minstrel_like_pick_best_mcs(void)
{
    uint8_t best = CONFIG_LIVE_MCS_MIN_INDEX;
    float best_tp = -1.0f;

    for (uint8_t i = CONFIG_LIVE_MCS_MIN_INDEX; i <= CONFIG_LIVE_MCS_MAX_INDEX; ++i) {
        float expected_tp = s_live_rate_stats[i].ewma_pdr * s_mcs_rate_mbps[i];
        if (expected_tp > best_tp) {
            best_tp = expected_tp;
            best = i;
        }
    }

    return best;
}

static uint8_t minstrel_like_pick_probe_mcs(uint8_t best_mcs)
{
    uint8_t probe = best_mcs;

    if (best_mcs < CONFIG_LIVE_MCS_MAX_INDEX) {
        probe = (uint8_t)(best_mcs + 1);
    } else if (best_mcs > CONFIG_LIVE_MCS_MIN_INDEX) {
        probe = (uint8_t)(best_mcs - 1);
    }

    return clamp_live_mcs_index(probe);
}

static uint8_t minstrel_like_next_mcs(void)
{
    bool do_update = (CONFIG_MINSTREL_UPDATE_EVERY_PKTS > 0)
                     && ((s_live_controller_total_acks % CONFIG_MINSTREL_UPDATE_EVERY_PKTS) == 0);
    bool do_probe = (CONFIG_MINSTREL_PROBE_EVERY_PKTS > 0)
                    && ((s_live_controller_total_acks % CONFIG_MINSTREL_PROBE_EVERY_PKTS) == 0);

    if (do_update) {
        s_live_best_mcs_index = minstrel_like_pick_best_mcs();
    }

    if (do_probe) {
        s_live_probe_active = 1;
        return minstrel_like_pick_probe_mcs(s_live_best_mcs_index);
    }

    s_live_probe_active = 0;
    return s_live_best_mcs_index;
}
#endif

static void live_rate_controller_on_ack(int delivered,
                                        wifi_phy_rate_t configured_rate,
                                        int64_t service_us,
                                        live_policy_snapshot_t *snapshot)
{
#if CONFIG_LIVE_MCS_SELECTION_ENABLED
    uint8_t used_mcs = 0;
    if (!rate_to_mcs_index(configured_rate, &used_mcs)) {
        return;
    }

    used_mcs = clamp_live_mcs_index(used_mcs);
    live_rate_stat_t *stat = &s_live_rate_stats[used_mcs];
    stat->attempts_window++;
    if (delivered) {
        stat->success_window++;
    }

    float sample_pdr = delivered ? 1.0f : 0.0f;
    stat->ewma_pdr = (stat->ewma_pdr * (float)(CONFIG_MINSTREL_EWMA_ALPHA_DEN - CONFIG_MINSTREL_EWMA_ALPHA_NUM)
                      + sample_pdr * (float)CONFIG_MINSTREL_EWMA_ALPHA_NUM)
                     / (float)CONFIG_MINSTREL_EWMA_ALPHA_DEN;

    if (service_us > 0) {
        float sample_service = (float)service_us;
        if (stat->service_samples == 0) {
            stat->ewma_service_us = sample_service;
        } else {
            stat->ewma_service_us = (stat->ewma_service_us * (float)(CONFIG_MINSTREL_EWMA_ALPHA_DEN - CONFIG_MINSTREL_EWMA_ALPHA_NUM)
                                    + sample_service * (float)CONFIG_MINSTREL_EWMA_ALPHA_NUM)
                                   / (float)CONFIG_MINSTREL_EWMA_ALPHA_DEN;
        }
        stat->service_samples++;
    }

    s_live_controller_total_acks++;

    float selected_reward = 0.0f;

#if CONFIG_LIVE_MCS_ALGO == 0
    s_live_current_mcs_index = minstrel_like_next_mcs();
    selected_reward = s_live_rate_stats[s_live_current_mcs_index].ewma_pdr * s_mcs_rate_mbps[s_live_current_mcs_index];
#elif CONFIG_LIVE_MCS_ALGO == 1
    /*
     * CUSTOM_POLICY is receiver-driven only. ACK telemetry is still tracked for
     * logs, but local ACK statistics do not choose the next MCS.
     */
    s_live_current_mcs_index = clamp_live_mcs_index(s_live_current_mcs_index);
    s_live_best_mcs_index = s_live_current_mcs_index;
    s_live_probe_active = 0;
#else
    /*
     * DQN owns the MCS decision. ACK telemetry is still tracked for logs; local
     * stepdown only runs when the explicit emergency fallback is enabled.
     */
    if (delivered) {
        s_dqn_consecutive_failures = 0;
    } else {
        s_dqn_consecutive_failures++;
#if CONFIG_DQN_FAILURE_STEPDOWN_ENABLED
        if (s_dqn_consecutive_failures >= CONFIG_DQN_FAILURE_STEPDOWN_COUNT) {
            if (s_live_current_mcs_index > CONFIG_LIVE_MCS_MIN_INDEX) {
                s_live_current_mcs_index--;
                __atomic_store_n(&s_dqn_failure_stepdown_pending, 1U, __ATOMIC_RELEASE);
            }
            s_dqn_consecutive_failures = 0;
        }
#endif
    }
    s_live_current_mcs_index = clamp_live_mcs_index(s_live_current_mcs_index);
    s_live_best_mcs_index = s_live_current_mcs_index;
    s_live_probe_active = 0;
#endif

    if (snapshot != NULL) {
        snapshot->used_mcs = used_mcs;
        snapshot->next_mcs = s_live_current_mcs_index;
        snapshot->best_mcs = s_live_best_mcs_index;
        snapshot->is_probe = s_live_probe_active;
        snapshot->used_ewma_pdr = stat->ewma_pdr;
        snapshot->used_ewma_service_us = stat->ewma_service_us;
        snapshot->selected_reward = selected_reward;
    }
#else
    (void)delivered;
    (void)configured_rate;
    (void)service_us;
    (void)snapshot;
#endif
}

static void live_rate_controller_apply(const uint8_t *peer_addr, uint32_t seq, const char *reason)
{
#if CONFIG_LIVE_MCS_SELECTION_ENABLED
    const wifi_phy_rate_t target = mcs_index_to_rate(clamp_live_mcs_index(s_live_current_mcs_index));
    if (target != s_current_configured_rate) {
        const wifi_phy_rate_t old_rate = s_current_configured_rate;
        uint8_t old_mcs = 0;
        uint8_t new_mcs = 0;
        const bool old_known = rate_to_mcs_index(old_rate, &old_mcs);
        const bool new_known = rate_to_mcs_index(target, &new_mcs);
        esp_now_set_peer_rate(peer_addr, target);
        printf("MCS_CHANGE,%lu,%d,%d,%d,%d,%s,%lld\n",
               (unsigned long)seq,
               old_known ? (int)old_mcs : -1,
               new_known ? (int)new_mcs : -1,
               (int)old_rate,
               (int)target,
               reason != NULL ? reason : "unknown",
               (long long)esp_timer_get_time());
        fflush(stdout);
    }
#else
    (void)peer_addr;
    (void)seq;
    (void)reason;
#endif
}

static bool ack_log_enqueue(const ack_log_event_t *event, TickType_t wait_ticks)
{
    if (s_ack_log_queue == NULL || xQueueSend(s_ack_log_queue, event, wait_ticks) != pdTRUE) {
        __atomic_fetch_add(&s_ack_log_drop_count, 1U, __ATOMIC_RELAXED);
        return false;
    }
    return true;
}

static void ack_print_service_summary(const char *tag,
                                      uint32_t total,
                                      uint32_t success_count,
                                      uint64_t service_sum_us,
                                      uint32_t service_sample_count,
                                      double median_service_us,
                                      uint64_t delivered_bytes,
                                      int64_t event_ts_us)
{
    if (service_sample_count == 0 || service_sum_us == 0) {
        return;
    }

    const double avg_service_us = (double)service_sum_us / (double)service_sample_count;
    const double goodput_mbps = ((double)delivered_bytes * 8.0) / (double)service_sum_us;

    printf("%s,%lu,%lu,%.1f,%.1f,%.3f,%lld\n",
           tag,
           (unsigned long)total,
           (unsigned long)success_count,
           avg_service_us,
           median_service_us,
           goodput_mbps,
           (long long)event_ts_us);
}

static void ack_service_median_reset(void)
{
    s_service_median_sample_count = 0;
}

static void ack_service_median_insert(uint32_t service_us)
{
    if (s_service_median_sample_count >= ACK_SERVICE_MEDIAN_MAX_SAMPLES) {
        return;
    }

    uint32_t lo = 0;
    uint32_t hi = s_service_median_sample_count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (s_service_median_samples[mid] <= service_us) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if (lo < s_service_median_sample_count) {
        memmove(&s_service_median_samples[lo + 1],
                &s_service_median_samples[lo],
                (s_service_median_sample_count - lo) * sizeof(s_service_median_samples[0]));
    }
    s_service_median_samples[lo] = service_us;
    s_service_median_sample_count++;
}

static double ack_service_median_us(void)
{
    if (s_service_median_sample_count == 0) {
        return 0.0;
    }

    uint32_t mid = s_service_median_sample_count / 2;
    if ((s_service_median_sample_count % 2) != 0) {
        return (double)s_service_median_samples[mid];
    }

    return ((double)s_service_median_samples[mid - 1] + (double)s_service_median_samples[mid]) / 2.0;
}

static void ack_logging_task(void *arg)
{
    (void)arg;
    ack_log_event_t event;
    uint64_t service_sum_us = 0;
    uint64_t delivered_bytes = 0;
    uint32_t service_sample_count = 0;

    for (;;) {
        if (xQueueReceive(s_ack_log_queue, &event, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        switch (event.type) {
        case ACK_LOG_EVENT_STATUS:
            /*
             * The first seven fields are unchanged from the original format.
             * The final four fields expose the configured rate and the scalar
             * tx_info metadata copied inside the callback.
             */
            if (event.service_us > 0) {
                service_sum_us += (uint64_t)event.service_us;
                service_sample_count++;
                if (event.service_us <= UINT32_MAX) {
                    ack_service_median_insert((uint32_t)event.service_us);
                }
                if (event.delivered && event.data_len > 0) {
                    delivered_bytes += (uint64_t)event.data_len;
                }
            }

            printf("ACK_STATUS,%lu,%d,%s,%lld,%lld,%lld,%d,%d,%d,%d\n",
                   (unsigned long)event.seq,
                   event.delivered,
                   event.delivered ? "final_ack" : "packet_drop",
                   (long long)event.event_ts_us,
                   (long long)event.send_ts_us,
                   (long long)event.service_us,
                   event.configured_rate,
                   event.actual_rate,
                   event.tx_status,
                   event.data_len);

        #if CONFIG_LIVE_MCS_SELECTION_ENABLED && CONFIG_LIVE_MCS_DECISION_LOG_ENABLED
                 printf("ACK_POLICY,%lu,%d,%d,%d,%d,%.4f,%.1f,%.4f\n",
                     (unsigned long)event.seq,
                     event.used_mcs,
                     event.next_mcs,
                     event.best_mcs,
                     event.is_probe,
                     event.used_ewma_pdr,
                     event.used_ewma_service_us,
                     event.selected_reward);
        #endif

            if (event.total > 0 && event.total % 100 == 0) {
                printf("ACK_PDR,%lu,%lu,%.1f,%lld\n",
                       (unsigned long)event.total,
                       (unsigned long)event.success_count,
                       100.0f * event.success_count / event.total,
                       (long long)event.event_ts_us);
                ack_print_service_summary("ACK_SERVICE",
                                          event.total,
                                          event.success_count,
                                          service_sum_us,
                                          service_sample_count,
                                          ack_service_median_us(),
                                          delivered_bytes,
                                          event.event_ts_us);
            }
            fflush(stdout);
            break;

        case ACK_LOG_EVENT_PDR_FINAL:
            if (event.total > 0) {
                printf("ACK_PDR_FINAL,%lu,%lu,%.1f,%lld\n",
                       (unsigned long)event.total,
                       (unsigned long)event.success_count,
                       100.0f * event.success_count / event.total,
                       (long long)event.event_ts_us);
                ack_print_service_summary("ACK_SERVICE_FINAL",
                                          event.total,
                                          event.success_count,
                                          service_sum_us,
                                          service_sample_count,
                                          ack_service_median_us(),
                                          delivered_bytes,
                                          event.event_ts_us);
                service_sum_us = 0;
                delivered_bytes = 0;
                service_sample_count = 0;
                ack_service_median_reset();
                fflush(stdout);
            }
            break;

        case ACK_LOG_EVENT_RESET:
            service_sum_us = 0;
            delivered_bytes = 0;
            service_sample_count = 0;
            ack_service_median_reset();
            printf("ACK_RESET_FOR_MCS%u\n", (unsigned int)event.new_mcs_index);
            fflush(stdout);
            break;

        case ACK_LOG_EVENT_CALLBACK_ERROR:
            printf("ACK_CALLBACK_ERROR,empty_seq_queue,%d,%lld,%d,%d,%d\n",
                   event.delivered,
                   (long long)event.event_ts_us,
                   event.actual_rate,
                   event.tx_status,
                   event.data_len);
            fflush(stdout);
            break;

        case ACK_LOG_EVENT_TRACKING_ERROR:
            printf("ACK_TRACKING_ERROR,%lu,%lld\n",
                   (unsigned long)event.seq,
                   (long long)event.event_ts_us);
            fflush(stdout);
            break;

        default:
            break;
        }

        uint32_t dropped = __atomic_exchange_n(&s_ack_log_drop_count, 0U, __ATOMIC_RELAXED);
        if (dropped > 0) {
            printf("ACK_LOG_DROPPED,%lu,%lld\n",
                   (unsigned long)dropped,
                   (long long)esp_timer_get_time());
            fflush(stdout);
        }
    }
}

static void ack_logging_init(void)
{
    s_ack_log_queue = xQueueCreate(ACK_LOG_QUEUE_SIZE, sizeof(ack_log_event_t));
    if (s_ack_log_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create ACK logging queue");
        abort();
    }

    BaseType_t task_created = xTaskCreate(ack_logging_task,
                                          "ack_logger",
                                          ACK_LOG_TASK_STACK_SIZE,
                                          NULL,
                                          ACK_LOG_TASK_PRIORITY,
                                          NULL);
    if (task_created != pdPASS) {
        ESP_LOGE(TAG, "Failed to create ACK logging task");
        vQueueDelete(s_ack_log_queue);
        s_ack_log_queue = NULL;
        abort();
    }
}

static void ack_record_status(uint32_t seq,
                              int delivered,
                              int64_t ack_ts_us,
                              int64_t send_ts_us,
                              wifi_phy_rate_t configured_rate,
                              const wifi_tx_info_t *tx_info)
{
    live_policy_snapshot_t snapshot = {0};
    int64_t service_us = (send_ts_us >= 0) ? (ack_ts_us - send_ts_us) : -1;
    live_rate_controller_on_ack(delivered, configured_rate, service_us, &snapshot);

    uint32_t total;
    uint32_t success_count;

    portENTER_CRITICAL(&s_ack_stats_mux);
    if (delivered) {
        s_ack_success_count++;
    } else {
        s_ack_fail_count++;
    }
    total = s_ack_success_count + s_ack_fail_count;
    success_count = s_ack_success_count;
    portEXIT_CRITICAL(&s_ack_stats_mux);

    ack_log_event_t event = {
        .type = ACK_LOG_EVENT_STATUS,
        .seq = seq,
        .total = total,
        .success_count = success_count,
        .delivered = delivered,
        .event_ts_us = ack_ts_us,
        .send_ts_us = send_ts_us,
        .service_us = service_us,
        .configured_rate = (int)configured_rate,
        .actual_rate = tx_info != NULL ? (int)tx_info->rate : -1,
        .tx_status = tx_info != NULL ? (int)tx_info->tx_status : -1,
        .data_len = tx_info != NULL ? (int)tx_info->data_len : -1,
        .used_mcs = (int)snapshot.used_mcs,
        .next_mcs = (int)snapshot.next_mcs,
        .best_mcs = (int)snapshot.best_mcs,
        .is_probe = (int)snapshot.is_probe,
        .used_ewma_pdr = snapshot.used_ewma_pdr,
        .used_ewma_service_us = snapshot.used_ewma_service_us,
        .selected_reward = snapshot.selected_reward,
    };

    /* Never block the high-priority Wi-Fi callback on console output. */
    (void)ack_log_enqueue(&event, 0);
}

#if !CONFIG_LIVE_MCS_SELECTION_ENABLED && CONFIG_RATE_SWITCH_MODE != 2
/* Reset ACK counters and queue final PDR output before MCS/rate switch. */
static void ack_reset_counters_for_rate_change(size_t new_mcs_index)
{
    uint32_t total;
    uint32_t success_count;

    portENTER_CRITICAL(&s_ack_stats_mux);
    total = s_ack_success_count + s_ack_fail_count;
    success_count = s_ack_success_count;
    s_ack_success_count = 0;
    s_ack_fail_count = 0;
    portEXIT_CRITICAL(&s_ack_stats_mux);

    if (total > 0) {
        ack_log_event_t final_event = {
            .type = ACK_LOG_EVENT_PDR_FINAL,
            .total = total,
            .success_count = success_count,
            .event_ts_us = esp_timer_get_time(),
        };
        (void)ack_log_enqueue(&final_event, pdMS_TO_TICKS(100));
    }

    ack_log_event_t reset_event = {
        .type = ACK_LOG_EVENT_RESET,
        .new_mcs_index = (uint32_t)new_mcs_index,
        .event_ts_us = esp_timer_get_time(),
    };
    (void)ack_log_enqueue(&reset_event, pdMS_TO_TICKS(100));
}
#endif

static bool ack_seq_enqueue(uint32_t seq, int64_t send_ts_us, wifi_phy_rate_t configured_rate)
{
    bool queued = false;

    portENTER_CRITICAL(&s_ack_seq_mux);
    uint16_t next = (uint16_t)((s_ack_seq_head + 1) % ACK_SEQ_QUEUE_SIZE);
    if (next != s_ack_seq_tail) {
        s_ack_seq_queue[s_ack_seq_head].seq = seq;
        s_ack_seq_queue[s_ack_seq_head].send_ts_us = send_ts_us;
        s_ack_seq_queue[s_ack_seq_head].configured_rate = configured_rate;
        s_ack_seq_head = next;
        queued = true;
    }
    portEXIT_CRITICAL(&s_ack_seq_mux);

    return queued;
}

static bool ack_seq_cancel_last(uint32_t seq)
{
    bool cancelled = false;

    portENTER_CRITICAL(&s_ack_seq_mux);
    if (s_ack_seq_head != s_ack_seq_tail) {
        uint16_t previous_head = (uint16_t)((s_ack_seq_head + ACK_SEQ_QUEUE_SIZE - 1) % ACK_SEQ_QUEUE_SIZE);
        if (s_ack_seq_queue[previous_head].seq == seq) {
            s_ack_seq_head = previous_head;
            cancelled = true;
        }
    }
    portEXIT_CRITICAL(&s_ack_seq_mux);

    return cancelled;
}

static bool ack_seq_dequeue(uint32_t *seq, int64_t *send_ts_us, wifi_phy_rate_t *configured_rate)
{
    bool dequeued = false;

    portENTER_CRITICAL(&s_ack_seq_mux);
    if (s_ack_seq_tail != s_ack_seq_head) {
        *seq = s_ack_seq_queue[s_ack_seq_tail].seq;
        *send_ts_us = s_ack_seq_queue[s_ack_seq_tail].send_ts_us;
        *configured_rate = s_ack_seq_queue[s_ack_seq_tail].configured_rate;
        s_ack_seq_tail = (uint16_t)((s_ack_seq_tail + 1) % ACK_SEQ_QUEUE_SIZE);
        dequeued = true;
    }
    portEXIT_CRITICAL(&s_ack_seq_mux);

    return dequeued;
}

#if CONFIG_LIVE_MCS_ALGO != 2
static bool remote_mcs_try_apply_for_next_seq(uint32_t next_seq)
{
#if CONFIG_REMOTE_MCS_RECOMMENDATION_ENABLED && CONFIG_LIVE_MCS_SELECTION_ENABLED
    remote_mcs_reco_state_t reco;

    portENTER_CRITICAL(&s_remote_mcs_reco_mux);
    reco = s_remote_mcs_reco;
    portEXIT_CRITICAL(&s_remote_mcs_reco_mux);

    if (!reco.valid) {
        return false;
    }

#if CONFIG_REMOTE_MCS_MIN_CONFIDENCE > 0
    if (reco.confidence < CONFIG_REMOTE_MCS_MIN_CONFIDENCE) {
        return false;
    }
#endif

    if ((esp_timer_get_time() - reco.rx_ts_us) > ((int64_t)CONFIG_REMOTE_MCS_MAX_AGE_MS * 1000LL)) {
        return false;
    }

    if (next_seq <= reco.seq) {
        return false;
    }

    s_live_current_mcs_index = clamp_live_mcs_index(reco.recommended_mcs);
    return true;
#else
    (void)next_seq;
    return false;
#endif
}
#endif

#if CONFIG_LIVE_MCS_ALGO == 2
static bool dqn_remote_mcs_try_apply_for_next_seq(uint32_t next_seq)
{
#if CONFIG_DQN_REMOTE_RECOMMENDATION_ENABLED && CONFIG_LIVE_MCS_SELECTION_ENABLED && CONFIG_LIVE_MCS_ALGO == 2
    dqn_remote_reco_state_t recommendation;

    portENTER_CRITICAL(&s_dqn_remote_reco_mux);
    recommendation = s_dqn_remote_reco;
    portEXIT_CRITICAL(&s_dqn_remote_reco_mux);

    if (!recommendation.valid) {
        return false;
    }
#if CONFIG_DQN_REMOTE_MIN_CONFIDENCE > 0
    if (recommendation.confidence < CONFIG_DQN_REMOTE_MIN_CONFIDENCE) {
        return false;
    }
#endif
    int64_t age_us = esp_timer_get_time() - recommendation.rx_ts_us;
    if (age_us > ((int64_t)CONFIG_DQN_REMOTE_MAX_AGE_MS * 1000LL)) {
        return false;
    }
    if (next_seq <= recommendation.seq) {
        return false;
    }
#if CONFIG_DQN_REMOTE_MAX_SEQ_GAP > 0
    uint32_t sequence_gap = next_seq - recommendation.seq;
    if (sequence_gap > CONFIG_DQN_REMOTE_MAX_SEQ_GAP) {
        return false;
    }
#endif

    /* Consume a recommendation once; an old high-rate choice must not undo a
     * later local failure step-down. */
    portENTER_CRITICAL(&s_dqn_remote_reco_mux);
    if (!s_dqn_remote_reco.valid || s_dqn_remote_reco.seq != recommendation.seq) {
        portEXIT_CRITICAL(&s_dqn_remote_reco_mux);
        return false;
    }
    s_dqn_remote_reco.valid = 0;
    portEXIT_CRITICAL(&s_dqn_remote_reco_mux);

    s_live_current_mcs_index = clamp_live_mcs_index(recommendation.recommended_mcs);
    s_dqn_consecutive_failures = 0;
#if CONFIG_DQN_LOG_ENABLED
    printf("DQN_FEEDBACK,%lu,%lu,%u,%u,%u,%d,%d,%lld\n",
           (unsigned long)recommendation.seq,
           (unsigned long)next_seq,
           (unsigned int)recommendation.recommended_mcs,
           (unsigned int)recommendation.confidence,
           (unsigned int)recommendation.margin_milli,
           (int)recommendation.rssi,
           (int)recommendation.snr,
           (long long)age_us);
    fflush(stdout);
#endif
    return true;
#else
    (void)next_seq;
    return false;
#endif
}
#endif

static void esp_now_recv_cb(const esp_now_recv_info_t *recv_info, const uint8_t *data, int data_len)
{
#if CONFIG_REMOTE_MCS_RECOMMENDATION_ENABLED || CONFIG_DQN_REMOTE_RECOMMENDATION_ENABLED
    if (recv_info == NULL || recv_info->src_addr == NULL || data == NULL) {
        return;
    }

    if (memcmp(recv_info->src_addr, CONFIG_CSI_RECV_MAC, ESP_NOW_ETH_ALEN) != 0) {
        return;
    }

#if CONFIG_REMOTE_MCS_RECOMMENDATION_ENABLED
    if (data_len == (int)sizeof(mcs_reco_msg_t)) {
        const mcs_reco_msg_t *msg = (const mcs_reco_msg_t *)data;
        if (msg->magic == MCS_RECO_MAGIC
                && msg->version == MCS_RECO_VERSION
                && msg->recommended_mcs <= 7) {
            portENTER_CRITICAL(&s_remote_mcs_reco_mux);
            s_remote_mcs_reco.valid = 1;
            s_remote_mcs_reco.recommended_mcs = msg->recommended_mcs;
            s_remote_mcs_reco.confidence = msg->confidence;
            s_remote_mcs_reco.seq = msg->seq;
            s_remote_mcs_reco.rssi = msg->rssi;
            s_remote_mcs_reco.snr = msg->snr;
            s_remote_mcs_reco.rx_ts_us = esp_timer_get_time();
            portEXIT_CRITICAL(&s_remote_mcs_reco_mux);
            return;
        }
    }
#endif

#if CONFIG_DQN_REMOTE_RECOMMENDATION_ENABLED
    if (data_len == (int)sizeof(dqn_reco_msg_t)) {
        const dqn_reco_msg_t *msg = (const dqn_reco_msg_t *)data;
        if (msg->magic == DQN_RECO_MAGIC
                && msg->version == DQN_RECO_VERSION
                && msg->recommended_mcs <= 7) {
            portENTER_CRITICAL(&s_dqn_remote_reco_mux);
            s_dqn_remote_reco.valid = 1;
            s_dqn_remote_reco.recommended_mcs = msg->recommended_mcs;
            s_dqn_remote_reco.confidence = msg->confidence;
            s_dqn_remote_reco.margin_milli = msg->margin_milli;
            s_dqn_remote_reco.seq = msg->seq;
            s_dqn_remote_reco.rssi = msg->rssi;
            s_dqn_remote_reco.snr = msg->snr;
            s_dqn_remote_reco.rx_ts_us = esp_timer_get_time();
            portEXIT_CRITICAL(&s_dqn_remote_reco_mux);
        }
    }
#endif
#else
    (void)recv_info;
    (void)data;
    (void)data_len;
#endif
}

static void esp_now_send_cb(const wifi_tx_info_t *tx_info, esp_now_send_status_t status)
{
    const int64_t ack_ts_us = esp_timer_get_time();
    const int delivered = (status == ESP_NOW_SEND_SUCCESS) ? 1 : 0;
    uint32_t seq = 0;
    int64_t send_ts_us = -1;
    wifi_phy_rate_t configured_rate = s_current_configured_rate;

    if (ack_seq_dequeue(&seq, &send_ts_us, &configured_rate)) {
        ack_record_status(seq, delivered, ack_ts_us, send_ts_us, configured_rate, tx_info);
    } else {
        ack_log_event_t event = {
            .type = ACK_LOG_EVENT_CALLBACK_ERROR,
            .delivered = delivered,
            .event_ts_us = ack_ts_us,
            .actual_rate = tx_info != NULL ? (int)tx_info->rate : -1,
            .tx_status = tx_info != NULL ? (int)tx_info->tx_status : -1,
            .data_len = tx_info != NULL ? (int)tx_info->data_len : -1,
        };
        (void)ack_log_enqueue(&event, 0);
    }

#if CONFIG_ACK_TIMING_MODE == 2
    /* Wake the sender immediately; logging happens independently. */
    if (s_sender_task_handle != NULL) {
        xTaskNotifyGive(s_sender_task_handle);
    }
#endif
}

static void esp_now_set_peer_rate(const uint8_t *peer_addr, wifi_phy_rate_t rate)
{
    esp_now_rate_config_t rate_config = {
        .phymode = CONFIG_ESP_NOW_PHYMODE,
        .rate = rate,
        .ersu = false,
        .dcm = false
    };
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer_addr, &rate_config));
    s_current_configured_rate = rate;
}

static esp_err_t esp_now_send_with_seq(const uint8_t *peer_addr, uint32_t seq)
{
#if CONFIG_IDENTICAL_TX_PAYLOAD
    (void)seq;
#else
    memcpy(s_tx_payload, &seq, sizeof(seq));
#endif
    return esp_now_send(peer_addr, s_tx_payload, sizeof(s_tx_payload));
}

static esp_err_t esp_now_send_tracked(const uint8_t *peer_addr, uint32_t seq)
{
    const int64_t send_ts_us = esp_timer_get_time();
    const wifi_phy_rate_t configured_rate = s_current_configured_rate;

    /*
     * Register the packet before submitting it, so an unusually fast callback
     * cannot arrive before the sequence/timestamp mapping exists.
     */
    if (!ack_seq_enqueue(seq, send_ts_us, configured_rate)) {
        ack_record_status(seq, 0, esp_timer_get_time(), send_ts_us, configured_rate, NULL);
        return ESP_ERR_NO_MEM;
    }

    esp_err_t ret = esp_now_send_with_seq(peer_addr, seq);
    if (ret != ESP_OK) {
        /* No send callback is expected for an immediate API failure. */
        if (!ack_seq_cancel_last(seq)) {
            ack_log_event_t event = {
                .type = ACK_LOG_EVENT_TRACKING_ERROR,
                .seq = seq,
                .event_ts_us = esp_timer_get_time(),
            };
            (void)ack_log_enqueue(&event, pdMS_TO_TICKS(10));
        }
        ack_record_status(seq, 0, esp_timer_get_time(), send_ts_us, configured_rate, NULL);
    }

    return ret;
}

static void wifi_init()
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    ESP_ERROR_CHECK(esp_netif_init());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));

#if CONFIG_IDF_TARGET_ESP32C5
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE));
    wifi_protocols_t protocols = {
        .ghz_2g = CONFIG_WIFI_2G_PROTOCOL,
        .ghz_5g = CONFIG_WIFI_5G_PROTOCOL
    };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(ESP_IF_WIFI_STA, &protocols));
    wifi_bandwidths_t bandwidth = {
        .ghz_2g = CONFIG_WIFI_2G_BANDWIDTHS,
        .ghz_5g = CONFIG_WIFI_5G_BANDWIDTHS
    };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(ESP_IF_WIFI_STA, &bandwidth));
#elif (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)) || CONFIG_IDF_TARGET_ESP32C61
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE));
    wifi_protocols_t protocols = {
        .ghz_2g = CONFIG_WIFI_2G_PROTOCOL,
    };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(ESP_IF_WIFI_STA, &protocols));
    wifi_bandwidths_t bandwidth = {
        .ghz_2g = CONFIG_WIFI_2G_BANDWIDTHS,
    };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(ESP_IF_WIFI_STA, &bandwidth));
#else
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(ESP_IF_WIFI_STA, CONFIG_WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_start());

#endif

    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
   
   
   
    // TX power units are 0.25 dBm; 84 => 21 dBm (max on supported targets).
    /*
    Attention

        3. Mapping Table {Power, max_tx_power} = {{8, 2}, {20, 5}, {28, 7}, {34, 8}, {44, 11}, {52, 13}, {56, 14}, {60, 15}, {66, 16}, {72, 18}, {80, 20}}.
    Attention

        4. Param power unit is 0.25dBm, range is [8, 84] corresponding to 2dBm - 20dBm.
    Attention

        5. Relationship between set value and actual value. As follows:
         {set value range, actual value} = {{[8, 19],8}, {[20, 27],20}, {[28, 33],28}, {[34, 43],34}, {[44, 51],44}, {[52, 55],52}, {[56, 59],56}, {[60, 65],60}, {[66, 71],66}, {[72, 79],72}, {[80, 84],80}}.

    */
    ESP_ERROR_CHECK(esp_wifi_set_max_tx_power(CONFIG_WIFI_TX_POWER));



    #if CONFIG_IDF_TARGET_ESP32C5
    if ((CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_2G_ONLY && CONFIG_WIFI_2G_BANDWIDTHS == WIFI_BW_HT20)
            || (CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_5G_ONLY && CONFIG_WIFI_5G_BANDWIDTHS == WIFI_BW_HT20)) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
            ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL,
                                                 get_secondary_channel_for_ht40(CONFIG_LESS_INTERFERENCE_CHANNEL)));
    }
#elif (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)) || CONFIG_IDF_TARGET_ESP32C61
    if (CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_2G_ONLY && CONFIG_WIFI_2G_BANDWIDTHS == WIFI_BW_HT20) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
            ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL,
                                                 get_secondary_channel_for_ht40(CONFIG_LESS_INTERFERENCE_CHANNEL)));
    }
#else
    if (CONFIG_WIFI_BANDWIDTH == WIFI_BW_HT20) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
            ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL,
                                                 get_secondary_channel_for_ht40(CONFIG_LESS_INTERFERENCE_CHANNEL)));
    }
#endif
    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, CONFIG_CSI_SEND_MAC));
}

static void wifi_esp_now_init(esp_now_peer_info_t peer)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    ESP_ERROR_CHECK(esp_now_register_send_cb(esp_now_send_cb));
    ESP_ERROR_CHECK(esp_now_register_recv_cb(esp_now_recv_cb));
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    esp_now_set_peer_rate(peer.peer_addr, CONFIG_ESP_NOW_RATE);
}

void app_main()
{
    /**
     * @brief Initialize NVS
     */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /**
     * @brief Initialize Wi-Fi
     */
    wifi_init();

    /**
     * @brief Initialize ESP-NOW
     *        ESP-NOW protocol see: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html
     */
    esp_now_peer_info_t peer = {
        .channel   = CONFIG_LESS_INTERFERENCE_CHANNEL,
        .ifidx     = WIFI_IF_STA,
        .encrypt   = false,
    };
    memcpy(peer.peer_addr, CONFIG_CSI_RECV_MAC, sizeof(peer.peer_addr));
    ack_logging_init();
    wifi_esp_now_init(peer);

#if CONFIG_ACK_TIMING_MODE == 2
    s_sender_task_handle = xTaskGetCurrentTaskHandle();
#endif

    ESP_LOGI(TAG, "================ CSI SEND ================");
    ESP_LOGI(TAG, "wifi_channel: %d, send_frequency: %d, payload_len: %d, sender_mac: " MACSTR ", receiver_mac: " MACSTR,
             CONFIG_LESS_INTERFERENCE_CHANNEL, CONFIG_SEND_FREQUENCY, CONFIG_ESP_NOW_PAYLOAD_LEN,
             MAC2STR(CONFIG_CSI_SEND_MAC), MAC2STR(CONFIG_CSI_RECV_MAC));
#if CONFIG_REMOTE_MCS_RECOMMENDATION_ENABLED
    ESP_LOGI(TAG, "Remote recommendation enabled: min_conf=%u, max_age_ms=%u",
             (unsigned int)CONFIG_REMOTE_MCS_MIN_CONFIDENCE,
             (unsigned int)CONFIG_REMOTE_MCS_MAX_AGE_MS);
#endif
#if CONFIG_DQN_REMOTE_RECOMMENDATION_ENABLED
    ESP_LOGI(
        TAG,
        "DQN recommendation enabled: min_conf=%u, max_age_ms=%u, max_seq_gap=%u, emergency_stepdown=%u after=%u failures",
        (unsigned int)CONFIG_DQN_REMOTE_MIN_CONFIDENCE,
        (unsigned int)CONFIG_DQN_REMOTE_MAX_AGE_MS,
        (unsigned int)CONFIG_DQN_REMOTE_MAX_SEQ_GAP,
        (unsigned int)CONFIG_DQN_FAILURE_STEPDOWN_ENABLED,
        (unsigned int)CONFIG_DQN_FAILURE_STEPDOWN_COUNT
    );
#endif
    printf("ACK_STATUS_HEADER,seq,delivered,termination_event,ack_ts_us,send_ts_us,service_us,configured_rate,actual_rate,tx_status,data_len\n");
    printf("ACK_SERVICE_HEADER,total,delivered,avg_service_us,median_service_us,goodput_mbps,ts_us\n");
#if CONFIG_LIVE_MCS_SELECTION_ENABLED && CONFIG_LIVE_MCS_DECISION_LOG_ENABLED
    printf("ACK_POLICY_HEADER,seq,used_mcs,next_mcs,best_mcs,is_probe,used_ewma_pdr,used_ewma_service_us,selected_reward\n");
#endif
    printf("MCS_CHANGE_HEADER,seq,old_mcs,new_mcs,old_rate,new_rate,reason,ts_us\n");
    fflush(stdout);

#if CONFIG_LIVE_MCS_SELECTION_ENABLED
    live_rate_controller_reset();
    live_rate_controller_apply(peer.peer_addr, 0, "initial");
    ESP_LOGI(TAG, "Live MCS controller enabled (%s), range=[MCS%u..MCS%u], initial=MCS%u",
#if CONFIG_LIVE_MCS_ALGO == 0
             "MINSTREL_LIKE",
#elif CONFIG_LIVE_MCS_ALGO == 1
             "CUSTOM_POLICY",
#else
             "DQN_POLICY",
#endif
             (unsigned int)CONFIG_LIVE_MCS_MIN_INDEX,
             (unsigned int)CONFIG_LIVE_MCS_MAX_INDEX,
             (unsigned int)s_live_current_mcs_index);

    for (uint32_t count = 0; ; ++count) {
#if CONFIG_LIVE_MCS_ALGO == 2
        const bool failure_stepdown = __atomic_exchange_n(
            &s_dqn_failure_stepdown_pending,
            0U,
            __ATOMIC_ACQ_REL
        ) != 0;
        const bool remote_applied = failure_stepdown
            ? false
            : dqn_remote_mcs_try_apply_for_next_seq(count);
        const char *mcs_change_reason = failure_stepdown
            ? "dqn_failure_stepdown"
            : remote_applied
                ? "dqn_recommendation"
                : "dqn_hold";
#else
        const bool remote_applied = remote_mcs_try_apply_for_next_seq(count);
        const char *mcs_change_reason = remote_applied ? "receiver_recommendation" :
#if CONFIG_LIVE_MCS_ALGO == 0
            s_live_probe_active ? "minstrel_probe" : "minstrel_best";
#else
            "custom_hold";
#endif
#endif
        live_rate_controller_apply(peer.peer_addr, count, mcs_change_reason);

        esp_err_t ret = esp_now_send_tracked(peer.peer_addr, count);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "free_heap: %ld <%s> ESP-NOW send error", esp_get_free_heap_size(), esp_err_to_name(ret));
        }

#if CONFIG_ACK_TIMING_MODE == 2
        if (ret == ESP_OK) {
            ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        }
#endif

        tx_pacing_delay_us();
    }
#else

#if CONFIG_RATE_SWITCH_MODE != 2
    size_t rate_index = 0;
#endif

#if CONFIG_RATE_SWITCH_MODE == 0
    /* TIME_BASED rate switching */
    const int64_t rate_switch_interval_us = (int64_t)CONFIG_RATE_SWITCH_INTERVAL_SEC * 1000000LL;
    int64_t next_rate_switch_us = esp_timer_get_time() + rate_switch_interval_us;
    ESP_LOGI(TAG, "ESP-NOW rate set: WIFI_PHY_RATE_MCS0_LGI (switching every %d seconds)", CONFIG_RATE_SWITCH_INTERVAL_SEC);

    for (uint32_t count = 0; ; ++count) {
        int64_t now_us = esp_timer_get_time();
        if (now_us >= next_rate_switch_us) {
            rate_index = (rate_index + 1) % (sizeof(s_esp_now_rate_cycle) / sizeof(s_esp_now_rate_cycle[0]));
            
            /* Reset counters and output final PDR for previous MCS before switching */
            ack_reset_counters_for_rate_change(rate_index);
            
            esp_now_set_peer_rate(peer.peer_addr, s_esp_now_rate_cycle[rate_index]);
            ESP_LOGI(TAG, "ESP-NOW rate set: WIFI_PHY_RATE_MCS%u_LGI", (unsigned int)rate_index);

            do {
                next_rate_switch_us += rate_switch_interval_us;
            } while (next_rate_switch_us <= now_us);
        }

        esp_err_t ret = esp_now_send_tracked(peer.peer_addr, count);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "free_heap: %ld <%s> ESP-NOW send error", esp_get_free_heap_size(), esp_err_to_name(ret));
        }

#if CONFIG_ACK_TIMING_MODE == 2
        if (ret == ESP_OK) {
            ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        }
#endif

        tx_pacing_delay_us();
    }
#elif CONFIG_RATE_SWITCH_MODE == 2
    /* STATIC rate — no switching, CONFIG_ESP_NOW_RATE is used for the entire run */
    ESP_LOGI(TAG, "ESP-NOW rate set: fixed WIFI_PHY_RATE_MCS%u_LGI (no switching)",
             (unsigned int)(CONFIG_ESP_NOW_RATE - WIFI_PHY_RATE_MCS0_LGI));

    for (uint32_t count = 0; ; ++count) {
        esp_err_t ret = esp_now_send_tracked(peer.peer_addr, count);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "free_heap: %ld <%s> ESP-NOW send error", esp_get_free_heap_size(), esp_err_to_name(ret));
        }
#if CONFIG_ACK_TIMING_MODE == 2
        if (ret == ESP_OK) {
            ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        }
#endif
        tx_pacing_delay_us();
    }
#else
    /* PACKET_BASED rate switching */
    uint32_t packets_attempted_in_current_rate = 0;
    ESP_LOGI(TAG, "ESP-NOW rate set: WIFI_PHY_RATE_MCS0_LGI (switching every %d attempted packets)", CONFIG_RATE_SWITCH_PACKET_COUNT);

    for (uint32_t count = 0; ; ++count) {
        if (packets_attempted_in_current_rate >= CONFIG_RATE_SWITCH_PACKET_COUNT) {
            rate_index = (rate_index + 1) % (sizeof(s_esp_now_rate_cycle) / sizeof(s_esp_now_rate_cycle[0]));
            
            /* Reset counters and output final PDR for previous MCS before switching */
            ack_reset_counters_for_rate_change(rate_index);
            
            esp_now_set_peer_rate(peer.peer_addr, s_esp_now_rate_cycle[rate_index]);
            ESP_LOGI(TAG, "ESP-NOW rate set: WIFI_PHY_RATE_MCS%u_LGI", (unsigned int)rate_index);
            packets_attempted_in_current_rate = 0;
        }

        packets_attempted_in_current_rate++;
        esp_err_t ret = esp_now_send_tracked(peer.peer_addr, count);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "free_heap: %ld <%s> ESP-NOW send error", esp_get_free_heap_size(), esp_err_to_name(ret));
        }

#if CONFIG_ACK_TIMING_MODE == 2
        if (ret == ESP_OK) {
            ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        }
#endif

        tx_pacing_delay_us();
    }
#endif
#endif
}
