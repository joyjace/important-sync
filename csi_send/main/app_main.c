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
#include <unistd.h>

#include "nvs_flash.h"

#include "esp_mac.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_timer.h"

/* ESP-IDF 6.0+ renamed WIFI_BW_HT20/HT40 to WIFI_BW20/BW40. */
#ifndef WIFI_BW_HT20
#define WIFI_BW_HT20 WIFI_BW20
#endif
#ifndef WIFI_BW_HT40
#define WIFI_BW_HT40 WIFI_BW40
#endif

#define CONFIG_LESS_INTERFERENCE_CHANNEL   11

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61 || (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0))
#define CONFIG_WIFI_BAND_MODE               WIFI_BAND_MODE_2G_ONLY
#define CONFIG_WIFI_2G_BANDWIDTHS           WIFI_BW_HT40
#define CONFIG_WIFI_5G_BANDWIDTHS           WIFI_BW_HT40
#define CONFIG_WIFI_2G_PROTOCOL             WIFI_PROTOCOL_11N
#define CONFIG_WIFI_5G_PROTOCOL             WIFI_PROTOCOL_11N
#else
#define CONFIG_WIFI_BANDWIDTH               WIFI_BW_HT40
#endif

#define CONFIG_ESP_NOW_PHYMODE           WIFI_PHY_MODE_HT40
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
#define CONFIG_SEND_FREQUENCY               100
#define CONFIG_PACKET_PACING_ENABLED        0  // 1 = paced by CONFIG_SEND_FREQUENCY, 0 = max-rate (no fixed sleep)
#define CONFIG_RATE_SWITCH_MODE             1  // 0 = TIME_BASED, 1 = PACKET_BASED, 2 = STATIC (fixed rate, no switching)
#define CONFIG_RATE_SWITCH_INTERVAL_SEC     10 // Used when TIME_BASED
#define CONFIG_RATE_SWITCH_PACKET_COUNT     1000 // Used when PACKET_BASED
#define CONFIG_ESP_NOW_PAYLOAD_LEN          128 // Bytes per ESP-NOW data frame (>= 4 to keep sequence ID) (16, 64, 128)
// TX power in units of 0.25 dBm. Range [8, 84] => [2 dBm, 20 dBm].
// Mapping: {set value range, actual value} = {{[8,19],8},{[20,27],20},{[28,33],28},{[34,43],34},{[44,51],44},{[52,55],52},{[56,59],56},{[60,65],60},{[66,71],66},{[72,79],72},{[80,84],80}}
#define CONFIG_WIFI_TX_POWER                80

#if CONFIG_ESP_NOW_PAYLOAD_LEN < 4
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

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define ESP_IF_WIFI_STA ESP_MAC_WIFI_STA
#endif

static const uint8_t CONFIG_CSI_SEND_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};
static const uint8_t CONFIG_CSI_RECV_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x01};
static const char *TAG = "csi_send";

static inline void tx_pacing_delay_us(void)
{
#if CONFIG_PACKET_PACING_ENABLED
    usleep(1000 * 1000 / CONFIG_SEND_FREQUENCY);
#endif
}

static uint32_t s_ack_success_count  = 0;
static uint32_t s_ack_fail_count     = 0;
static uint8_t s_tx_payload[CONFIG_ESP_NOW_PAYLOAD_LEN] = {0};

#define ACK_SEQ_QUEUE_SIZE 2048
static uint32_t s_ack_seq_queue[ACK_SEQ_QUEUE_SIZE];
static volatile uint16_t s_ack_seq_head = 0;
static volatile uint16_t s_ack_seq_tail = 0;

static void ack_emit_status(uint32_t seq, int delivered)
{
    printf("ACK_STATUS,%lu,%d\n", (unsigned long)seq, delivered);
    fflush(stdout);

    if (delivered) {
        s_ack_success_count++;
    } else {
        s_ack_fail_count++;
    }

    uint32_t total = s_ack_success_count + s_ack_fail_count;
    if (total > 0 && total % 100 == 0) {
        printf("ACK_PDR,%lu,%lu,%.1f\n", (unsigned long)total,
               (unsigned long)s_ack_success_count,
               100.0f * s_ack_success_count / total);
        fflush(stdout);
    }
}

#if CONFIG_RATE_SWITCH_MODE != 2
/* Reset ACK counters and print final PDR before MCS/rate switch */
static void ack_reset_counters_for_rate_change(size_t new_mcs_index)
{
    uint32_t total = s_ack_success_count + s_ack_fail_count;
    if (total > 0) {
        float pdr = 100.0f * s_ack_success_count / total;
        printf("ACK_PDR_FINAL,%lu,%lu,%.1f\n", (unsigned long)total,
               (unsigned long)s_ack_success_count, pdr);
        fflush(stdout);
    }
    /* Reset counters for next MCS/rate */
    s_ack_success_count = 0;
    s_ack_fail_count = 0;
    printf("ACK_RESET_FOR_MCS%u\n", (unsigned int)new_mcs_index);
    fflush(stdout);
}
#endif

static void ack_seq_enqueue(uint32_t seq)
{
    uint16_t next = (uint16_t)((s_ack_seq_head + 1) % ACK_SEQ_QUEUE_SIZE);
    if (next == s_ack_seq_tail) {
        /* Queue overflow: drop oldest to keep callback mapping moving forward. */
        s_ack_seq_tail = (uint16_t)((s_ack_seq_tail + 1) % ACK_SEQ_QUEUE_SIZE);
    }
    s_ack_seq_queue[s_ack_seq_head] = seq;
    s_ack_seq_head = next;
}

static bool ack_seq_dequeue(uint32_t *seq)
{
    if (s_ack_seq_tail == s_ack_seq_head) {
        return false;
    }
    *seq = s_ack_seq_queue[s_ack_seq_tail];
    s_ack_seq_tail = (uint16_t)((s_ack_seq_tail + 1) % ACK_SEQ_QUEUE_SIZE);
    return true;
}

static void esp_now_send_cb(const wifi_tx_info_t *tx_info, esp_now_send_status_t status)
{
    (void)tx_info;
    uint32_t seq = 0;
    int delivered = (status == ESP_NOW_SEND_SUCCESS) ? 1 : 0;
    if (!ack_seq_dequeue(&seq)) {
        ESP_LOGW(TAG, "ACK callback arrived with empty seq queue");
        return;
    }
    ack_emit_status(seq, delivered);
}

#if CONFIG_RATE_SWITCH_MODE != 2
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
#endif

static void esp_now_set_peer_rate(const uint8_t *peer_addr, wifi_phy_rate_t rate)
{
    esp_now_rate_config_t rate_config = {
        .phymode = CONFIG_ESP_NOW_PHYMODE,
        .rate = rate,
        .ersu = false,
        .dcm = false
    };
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer_addr, &rate_config));
}

static esp_err_t esp_now_send_with_seq(const uint8_t *peer_addr, uint32_t seq)
{
    memcpy(s_tx_payload, &seq, sizeof(seq));
    return esp_now_send(peer_addr, s_tx_payload, sizeof(s_tx_payload));
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
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#elif (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)) || CONFIG_IDF_TARGET_ESP32C61
    if (CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_2G_ONLY && CONFIG_WIFI_2G_BANDWIDTHS == WIFI_BW_HT20) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#else
    if (CONFIG_WIFI_BANDWIDTH == WIFI_BW_HT20) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#endif
    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, CONFIG_CSI_SEND_MAC));
}

static void wifi_esp_now_init(esp_now_peer_info_t peer)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    ESP_ERROR_CHECK(esp_now_register_send_cb(esp_now_send_cb));
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
        .peer_addr = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x01},
    };
    wifi_esp_now_init(peer);

    ESP_LOGI(TAG, "================ CSI SEND ================");
    ESP_LOGI(TAG, "wifi_channel: %d, send_frequency: %d, payload_len: %d, sender_mac: " MACSTR ", receiver_mac: " MACSTR,
             CONFIG_LESS_INTERFERENCE_CHANNEL, CONFIG_SEND_FREQUENCY, CONFIG_ESP_NOW_PAYLOAD_LEN,
             MAC2STR(CONFIG_CSI_SEND_MAC), MAC2STR(CONFIG_CSI_RECV_MAC));

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

        esp_err_t ret = esp_now_send_with_seq(peer.peer_addr, count);
        if (ret == ESP_OK) {
            ack_seq_enqueue(count);
        } else {
            /* Immediate enqueue/send failure: no callback expected, mark as lost now. */
            ack_emit_status(count, 0);
            ESP_LOGW(TAG, "free_heap: %ld <%s> ESP-NOW send error", esp_get_free_heap_size(), esp_err_to_name(ret));
        }

        tx_pacing_delay_us();
    }
#elif CONFIG_RATE_SWITCH_MODE == 2
    /* STATIC rate — no switching, CONFIG_ESP_NOW_RATE is used for the entire run */
    ESP_LOGI(TAG, "ESP-NOW rate set: fixed WIFI_PHY_RATE_MCS%u_LGI (no switching)",
             (unsigned int)(CONFIG_ESP_NOW_RATE - WIFI_PHY_RATE_MCS0_LGI));

    for (uint32_t count = 0; ; ++count) {
        esp_err_t ret = esp_now_send_with_seq(peer.peer_addr, count);
        if (ret == ESP_OK) {
            ack_seq_enqueue(count);
        } else {
            ack_emit_status(count, 0);
            ESP_LOGW(TAG, "free_heap: %ld <%s> ESP-NOW send error", esp_get_free_heap_size(), esp_err_to_name(ret));
        }
        tx_pacing_delay_us();
    }
#else
    /* PACKET_BASED rate switching */
    uint32_t packets_sent_in_current_rate = 0;
    ESP_LOGI(TAG, "ESP-NOW rate set: WIFI_PHY_RATE_MCS0_LGI (switching every %d packets)", CONFIG_RATE_SWITCH_PACKET_COUNT);

    for (uint32_t count = 0; ; ++count) {
        if (packets_sent_in_current_rate >= CONFIG_RATE_SWITCH_PACKET_COUNT) {
            rate_index = (rate_index + 1) % (sizeof(s_esp_now_rate_cycle) / sizeof(s_esp_now_rate_cycle[0]));
            
            /* Reset counters and output final PDR for previous MCS before switching */
            ack_reset_counters_for_rate_change(rate_index);
            
            esp_now_set_peer_rate(peer.peer_addr, s_esp_now_rate_cycle[rate_index]);
            ESP_LOGI(TAG, "ESP-NOW rate set: WIFI_PHY_RATE_MCS%u_LGI", (unsigned int)rate_index);
            packets_sent_in_current_rate = 0;
        }

        esp_err_t ret = esp_now_send_with_seq(peer.peer_addr, count);
        if (ret == ESP_OK) {
            ack_seq_enqueue(count);
            packets_sent_in_current_rate++;
        } else {
            /* Immediate enqueue/send failure: no callback expected, mark as lost now. */
            ack_emit_status(count, 0);
            ESP_LOGW(TAG, "free_heap: %ld <%s> ESP-NOW send error", esp_get_free_heap_size(), esp_err_to_name(ret));
        }

        tx_pacing_delay_us();
    }
#endif
}