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
#include <math.h>

#include "nvs_flash.h"

#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_csi_gain_ctrl.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "generated_reward_model_v2.h"

/* ESP-IDF 6.0+ renamed WIFI_BW_HT20/HT40 to WIFI_BW20/BW40. */
#ifndef WIFI_BW_HT20
#define WIFI_BW_HT20 WIFI_BW20
#endif
#ifndef WIFI_BW_HT40
#define WIFI_BW_HT40 WIFI_BW40
#endif
//test
#define CONFIG_LESS_INTERFERENCE_CHANNEL   1
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61 || (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0))
#define CONFIG_WIFI_BAND_MODE               WIFI_BAND_MODE_2G_ONLY
#define CONFIG_WIFI_2G_BANDWIDTHS           WIFI_BW_HT20
#define CONFIG_WIFI_5G_BANDWIDTHS           WIFI_BW_HT20
#define CONFIG_WIFI_2G_PROTOCOL             WIFI_PROTOCOL_11N
#define CONFIG_WIFI_5G_PROTOCOL             WIFI_PROTOCOL_11N
#else
#define CONFIG_WIFI_BANDWIDTH           WIFI_BW_HT20
#endif

#define CONFIG_ESP_NOW_PHYMODE           WIFI_PHY_MODE_HT20
#define CONFIG_ESP_NOW_RATE             WIFI_PHY_RATE_MCS0_LGI
#define CONFIG_MINIMIZE_CONSOLE_OUTPUT   1  // 1 = reduce non-CSI logs on UART, 0 = default logging
#define CONFIG_MCS_RECOMMENDATION_ENABLED 0 // 1 = enable MCS recommendation feature, 0 = disable
#define CONFIG_MCS_RECOMMENDATION_EVERY_N_PACKETS 20  // 1=every packet, N=every N data packets
#define CONFIG_MCS_RECOMMENDATION_USE_MODEL 1  // 1=use ML model, 0=use RSSI-based heuristic
#define CONFIG_IDENTICAL_TX_PAYLOAD         1  // Match csi_send: 1 = payload carries no per-packet sequence
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
#define CONFIG_FORCE_GAIN                   1

#if CONFIG_LESS_INTERFERENCE_CHANNEL < 1 || CONFIG_LESS_INTERFERENCE_CHANNEL > 13
#error "CONFIG_LESS_INTERFERENCE_CHANNEL must be in [1, 13] for 2.4 GHz"
#endif

#if CONFIG_MCS_RECOMMENDATION_ENABLED
#if CONFIG_MCS_RECOMMENDATION_EVERY_N_PACKETS <= 0
#error "CONFIG_MCS_RECOMMENDATION_EVERY_N_PACKETS must be > 0"
#endif
#endif

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61
#define CSI_FORCE_LLTF                      0
#endif

#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C61
#define CONFIG_GAIN_CONTROL                 1
#endif

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define ESP_IF_WIFI_STA ESP_MAC_WIFI_STA
#endif

static const uint8_t CONFIG_CSI_SEND_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x10};
static const uint8_t CONFIG_CSI_RECV_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x11};
static const char *TAG = "csi_recv";

#define MCS_RECO_MAGIC 0xA5
#define MCS_RECO_VERSION 1

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

#if CONFIG_MCS_RECOMMENDATION_ENABLED
#define MCS_RECO_QUEUE_SIZE 128
#define MCS_RECO_TASK_STACK 3072
#define MCS_RECO_TASK_PRIO (tskIDLE_PRIORITY + 1)

static QueueHandle_t s_mcs_reco_queue = NULL;
static volatile uint32_t s_mcs_reco_drop_count = 0;
static uint32_t s_mcs_data_pkt_count = 0;

static float quantile_from_sorted(const float *sorted, int n, float q)
{
    if (n <= 0) {
        return 0.0f;
    }
    if (n == 1) {
        return sorted[0];
    }

    float pos = q * (float)(n - 1);
    int lo = (int)pos;
    int hi = lo + 1;
    if (hi >= n) {
        hi = n - 1;
    }
    float t = pos - (float)lo;
    return sorted[lo] * (1.0f - t) + sorted[hi] * t;
}

static void sort_small_array(float *arr, int n)
{
    for (int i = 1; i < n; ++i) {
        float key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}

static void build_model_state(const wifi_csi_info_t *info,
                              const wifi_pkt_rx_ctrl_t *rx_ctrl,
                              int8_t fft_gain,
                              uint8_t agc_gain,
                              float state[REWARD_MODEL_STATE_DIM])
{
    memset(state, 0, sizeof(float) * REWARD_MODEL_STATE_DIM);

    float amps[117] = {0};
    int amp_count = info->len / 2;
    if (amp_count > 117) {
        amp_count = 117;
    }

    for (int i = 0; i < amp_count; ++i) {
        float i_val = (float)((int8_t)info->buf[2 * i]);
        float q_val = (float)((int8_t)info->buf[2 * i + 1]);
        float amp = sqrtf(i_val * i_val + q_val * q_val);
        amps[i] = amp;
        state[i] = amp;
    }

    float iq_mean = 0.0f;
    float iq_std = 0.0f;
    float iq_p10 = 0.0f;
    float iq_p50 = 0.0f;
    float iq_p90 = 0.0f;

    if (amp_count > 0) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int i = 0; i < amp_count; ++i) {
            sum += amps[i];
            sum_sq += amps[i] * amps[i];
        }
        iq_mean = sum / (float)amp_count;
        float variance = (sum_sq / (float)amp_count) - (iq_mean * iq_mean);
        if (variance < 0.0f) {
            variance = 0.0f;
        }
        iq_std = sqrtf(variance);

        float sorted[117] = {0};
        for (int i = 0; i < amp_count; ++i) {
            sorted[i] = amps[i];
        }
        sort_small_array(sorted, amp_count);
        iq_p10 = quantile_from_sorted(sorted, amp_count, 0.10f);
        iq_p50 = quantile_from_sorted(sorted, amp_count, 0.50f);
        iq_p90 = quantile_from_sorted(sorted, amp_count, 0.90f);
    }

    float snr = (float)((int)rx_ctrl->rssi - (int)rx_ctrl->noise_floor);
    state[117] = (float)rx_ctrl->rssi;
    state[118] = snr;
    state[119] = (float)fft_gain;
    state[120] = (float)agc_gain;
    state[121] = (float)rx_ctrl->channel;
    state[122] = (float)rx_ctrl->sig_len;
    state[123] = iq_mean;
    state[124] = iq_std;
    state[125] = iq_p10;
    state[126] = iq_p50;
    state[127] = iq_p90;
}

static float reward_model_score_action(const float state[REWARD_MODEL_STATE_DIM], uint8_t action)
{
    float x[REWARD_MODEL_INPUT_DIM] = {0};
    float h1[REWARD_MODEL_HIDDEN_DIM] = {0};
    float h2[REWARD_MODEL_HIDDEN_DIM] = {0};

    for (int i = 0; i < REWARD_MODEL_STATE_DIM; ++i) {
        x[i] = (state[i] - reward_model_state_mean[i]) / reward_model_state_std[i];
    }

    for (int i = 0; i < REWARD_MODEL_ACTION_DIM; ++i) {
        x[REWARD_MODEL_STATE_DIM + i] = (i == action) ? 1.0f : 0.0f;
    }
    x[REWARD_MODEL_STATE_DIM + REWARD_MODEL_ACTION_DIM] = (float)action / (float)(REWARD_MODEL_ACTION_DIM - 1);

    for (int o = 0; o < REWARD_MODEL_HIDDEN_DIM; ++o) {
        float sum = reward_model_b1[o];
        const int row = o * REWARD_MODEL_INPUT_DIM;
        for (int i = 0; i < REWARD_MODEL_INPUT_DIM; ++i) {
            sum += reward_model_w1[row + i] * x[i];
        }
        h1[o] = (sum > 0.0f) ? sum : 0.0f;
    }

    for (int o = 0; o < REWARD_MODEL_HIDDEN_DIM; ++o) {
        float sum = reward_model_b2[o];
        const int row = o * REWARD_MODEL_HIDDEN_DIM;
        for (int i = 0; i < REWARD_MODEL_HIDDEN_DIM; ++i) {
            sum += reward_model_w2[row + i] * h1[i];
        }
        h2[o] = (sum > 0.0f) ? sum : 0.0f;
    }

    float out = reward_model_b3[0];
    for (int i = 0; i < REWARD_MODEL_HIDDEN_DIM; ++i) {
        out += reward_model_w3[i] * h2[i];
    }

    return out;
}

static uint8_t recommend_mcs_with_model(const float state[REWARD_MODEL_STATE_DIM], uint8_t *confidence_out)
{
    uint8_t best = 0;
    float best_score = reward_model_score_action(state, 0);
    float second_score = best_score;

    for (uint8_t action = 1; action < REWARD_MODEL_ACTION_DIM; ++action) {
        float score = reward_model_score_action(state, action);
#if REWARD_MODEL_OBJECTIVE_MINIMIZE
        if (score < best_score) {
            second_score = best_score;
            best_score = score;
            best = action;
        } else if (score < second_score || best == action - 1) {
            second_score = score;
        }
#else
        if (score > best_score) {
            second_score = best_score;
            best_score = score;
            best = action;
        } else if (score > second_score || best == action - 1) {
            second_score = score;
        }
#endif
    }

    float margin = fabsf(best_score - second_score);
    int conf = (int)(margin * 30.0f);
    if (conf > 100) {
        conf = 100;
    }
    if (conf < 0) {
        conf = 0;
    }

    if (confidence_out != NULL) {
        *confidence_out = (uint8_t)conf;
    }
    return best;
}

static uint8_t recommend_mcs_from_rssi(int8_t rssi)
{
    if (rssi >= -45) return 7;
    if (rssi >= -50) return 6;
    if (rssi >= -55) return 5;
    if (rssi >= -60) return 4;
    if (rssi >= -66) return 3;
    if (rssi >= -72) return 2;
    if (rssi >= -78) return 1;
    return 0;
}

static void mcs_recommendation_task(void *arg)
{
    (void)arg;
    mcs_reco_msg_t msg;

    for (;;) {
        if (xQueueReceive(s_mcs_reco_queue, &msg, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        esp_err_t ret = esp_now_send(CONFIG_CSI_SEND_MAC, (const uint8_t *)&msg, sizeof(msg));
        if (ret != ESP_OK) {
            ESP_LOGD(TAG, "MCS recommendation send failed: %s", esp_err_to_name(ret));
        }
    }
}

static void mcs_recommendation_init(void)
{
    s_mcs_reco_queue = xQueueCreate(MCS_RECO_QUEUE_SIZE, sizeof(mcs_reco_msg_t));
    if (s_mcs_reco_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create recommendation queue");
        abort();
    }

    if (xTaskCreate(mcs_recommendation_task,
                    "mcs_reco_tx",
                    MCS_RECO_TASK_STACK,
                    NULL,
                    MCS_RECO_TASK_PRIO,
                    NULL) != pdPASS) {
        ESP_LOGE(TAG, "Failed to create recommendation task");
        vQueueDelete(s_mcs_reco_queue);
        s_mcs_reco_queue = NULL;
        abort();
    }
}
#endif

static wifi_second_chan_t get_secondary_channel_for_ht40(int primary_channel)
{
    if (primary_channel <= 4) {
        return WIFI_SECOND_CHAN_ABOVE;
    }

    return WIFI_SECOND_CHAN_BELOW;
}

static void configure_console_log_verbosity(void)
{
#if CONFIG_MINIMIZE_CONSOLE_OUTPUT
    /* Keep UART bandwidth for CSI_DATA lines emitted via ets_printf. */
    esp_log_level_set("*", ESP_LOG_ERROR);
    esp_log_level_set(TAG, ESP_LOG_NONE);
#endif
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
    esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE);
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
    esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE);
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

    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, CONFIG_CSI_RECV_MAC));
}

static void wifi_esp_now_init(esp_now_peer_info_t peer)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    esp_now_rate_config_t rate_config = {
        .phymode = CONFIG_ESP_NOW_PHYMODE,
        .rate = CONFIG_ESP_NOW_RATE,//  WIFI_PHY_RATE_MCS0_LGI,
        .ersu = false,
        .dcm = false
    };
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate_config));

}

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) {
        ESP_LOGW(TAG, "<%s> wifi_csi_cb", esp_err_to_name(ESP_ERR_INVALID_ARG));
        return;
    }

    if (memcmp(info->mac, CONFIG_CSI_SEND_MAC, 6)) {
        return;
    }

    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
    static int s_count = 0;

    /* ACK frames carry no ESP-NOW payload (sig_len < 40). Log their CSI metadata
     * and return early so the CSV parser is not fed an invalid row. */
    if (rx_ctrl->sig_len < 40) {
        ESP_LOGD(TAG, "ACK_CSI mac=" MACSTR " rssi=%d sig_len=%d timestamp=%d",
                 MAC2STR(info->mac), rx_ctrl->rssi, rx_ctrl->sig_len, rx_ctrl->timestamp);
        return;
    }
    float compensate_gain = 1.0f;
    static uint8_t agc_gain = 0;
    static int8_t fft_gain = 0;
#if CONFIG_GAIN_CONTROL
    static uint8_t agc_gain_baseline = 0;
    static int8_t fft_gain_baseline = 0;
    esp_csi_gain_ctrl_get_rx_gain(rx_ctrl, &agc_gain, &fft_gain);
    if (s_count < 100) {
        esp_csi_gain_ctrl_record_rx_gain(agc_gain, fft_gain);
    } else if (s_count == 100) {
        esp_csi_gain_ctrl_get_rx_gain_baseline(&agc_gain_baseline, &fft_gain_baseline);
#if CONFIG_FORCE_GAIN
        esp_csi_gain_ctrl_set_rx_force_gain(agc_gain_baseline, fft_gain_baseline);
        ESP_LOGD(TAG, "fft_force %d, agc_force %d", fft_gain_baseline, agc_gain_baseline);
#endif
    }
    esp_csi_gain_ctrl_get_gain_compensation(&compensate_gain, agc_gain, fft_gain);
    ESP_LOGD(TAG, "compensate_gain %f, agc_gain %d, fft_gain %d", compensate_gain, agc_gain, fft_gain);
#endif

#if CONFIG_IDENTICAL_TX_PAYLOAD
    uint32_t rx_id = (uint32_t)s_count;
#else
    uint32_t rx_id = *(uint32_t *)(info->payload + 15);
#endif

#if CONFIG_MCS_RECOMMENDATION_ENABLED
    if ((s_mcs_data_pkt_count % CONFIG_MCS_RECOMMENDATION_EVERY_N_PACKETS) == 0) {
        int8_t snr = (int8_t)(rx_ctrl->rssi - rx_ctrl->noise_floor);
        uint8_t recommended_mcs = recommend_mcs_from_rssi(rx_ctrl->rssi);
        uint8_t confidence = 100;
    #if CONFIG_MCS_RECOMMENDATION_USE_MODEL
        float state[REWARD_MODEL_STATE_DIM] = {0};
        build_model_state(info, rx_ctrl, fft_gain, agc_gain, state);
        recommended_mcs = recommend_mcs_with_model(state, &confidence);
    #endif
        mcs_reco_msg_t reco = {
            .magic = MCS_RECO_MAGIC,
            .version = MCS_RECO_VERSION,
            .recommended_mcs = recommended_mcs,
            .confidence = confidence,
            .seq = rx_id,
            .rssi = rx_ctrl->rssi,
            .snr = snr,
            .reserved = {0, 0},
        };
        if (s_mcs_reco_queue == NULL || xQueueSend(s_mcs_reco_queue, &reco, 0) != pdTRUE) {
            s_mcs_reco_drop_count++;
        }
    }
    s_mcs_data_pkt_count++;
#endif
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C61
    uint32_t rate_for_csv;
#if CONFIG_SOC_WIFI_HE_SUPPORT
    if (rx_ctrl->cur_bb_format == RX_BB_FORMAT_HT) {
        rate_for_csv = (uint32_t)(rx_ctrl->he_siga1 & 0x7F);
    } else {
        rate_for_csv = rx_ctrl->rate;
    }
#else
    rate_for_csv = (rx_ctrl->sig_mode == 1) ? rx_ctrl->mcs : rx_ctrl->rate;
#endif
    if (!s_count) {
        ESP_LOGD(TAG, "================ CSI RECV ================");
        ets_printf("type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data\n");
    }

    ets_printf("CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%u,%d,%d",
               rx_id, MAC2STR(info->mac), rx_ctrl->rssi, rate_for_csv,
               rx_ctrl->noise_floor, fft_gain, agc_gain,  rx_ctrl->channel,
               (unsigned int)rx_ctrl->timestamp, rx_ctrl->sig_len, rx_ctrl->rx_state);
#else
    if (!s_count) {
        ESP_LOGD(TAG, "================ CSI RECV ================");
        ets_printf("type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data\n");
    }

    ets_printf("CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%u,%d,%d,%d",
               rx_id, MAC2STR(info->mac), rx_ctrl->rssi, rx_ctrl->rate, rx_ctrl->sig_mode,
               rx_ctrl->mcs, rx_ctrl->cwb, rx_ctrl->smoothing, rx_ctrl->not_sounding,
               rx_ctrl->aggregation, rx_ctrl->stbc, rx_ctrl->fec_coding, rx_ctrl->sgi,
               rx_ctrl->noise_floor, rx_ctrl->ampdu_cnt, rx_ctrl->channel, rx_ctrl->secondary_channel,
               (unsigned int)rx_ctrl->timestamp, rx_ctrl->ant, rx_ctrl->sig_len, rx_ctrl->rx_state);

#endif
#if (CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61) && CSI_FORCE_LLTF
    int16_t csi = ((int16_t)(((((uint16_t)info->buf[1]) << 8) | info->buf[0]) << 4) >> 4);
    ets_printf(",%d,%d,\"[%d", (info->len - 2) / 2, info->first_word_invalid, (int16_t)(compensate_gain * csi));
    for (int i = 2; i < (info->len - 2); i += 2) {
        csi = ((int16_t)(((((uint16_t)info->buf[i + 1]) << 8) | info->buf[i]) << 4) >> 4);
        ets_printf(",%d", (int16_t)(compensate_gain * csi));
    }
#else
    ets_printf(",%d,%d,\"[%d", info->len, info->first_word_invalid, (int16_t)(compensate_gain * info->buf[0]));
    for (int i = 1; i < info->len; i++) {
        ets_printf(",%d", (int16_t)(compensate_gain * info->buf[i]));
    }
#endif
    ets_printf("]\"\n");
    s_count++;
}

static void wifi_csi_init()
{
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    /**< default config */
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61
    wifi_csi_config_t csi_config = {
        .enable                   = true,
        .acquire_csi_legacy       = false,
        .acquire_csi_force_lltf   = CSI_FORCE_LLTF,
        .acquire_csi_ht20         = true,
        .acquire_csi_ht40         = true,
        .acquire_csi_vht          = false,
        .acquire_csi_su           = false,
        .acquire_csi_mu           = false,
        .acquire_csi_dcm          = false,
        .acquire_csi_beamformed   = false,
        .acquire_csi_he_stbc_mode = 2,
        .val_scale_cfg            = 0,
        .dump_ack_en              = true,
        .reserved                 = false
    };
#elif CONFIG_IDF_TARGET_ESP32C6
    wifi_csi_config_t csi_config = {
        .enable                 = true,
        .acquire_csi_legacy     = false,
        .acquire_csi_ht20       = true,
        .acquire_csi_ht40       = true,
        .acquire_csi_su         = true,
        .acquire_csi_mu         = true,
        .acquire_csi_dcm        = true,
        .acquire_csi_beamformed = true,
        .acquire_csi_he_stbc    = 2,
        .val_scale_cfg          = false,
        .dump_ack_en            = true,
        .reserved               = false
    };
#else
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = true,
        .stbc_htltf2_en    = true,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = false,
    };
#endif
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

void app_main()
{
    configure_console_log_verbosity();

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
    memcpy(peer.peer_addr, CONFIG_CSI_SEND_MAC, sizeof(peer.peer_addr));

    wifi_esp_now_init(peer);

#if CONFIG_MCS_RECOMMENDATION_ENABLED
    mcs_recommendation_init();
#endif

    wifi_csi_init();
}
