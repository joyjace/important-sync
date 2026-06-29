#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Terminal-only CSI reader for SSH use.
# Reads CSI_DATA lines from serial, saves normalized CSV, logs malformed/non-CSI lines,
# and prints periodic summaries to stdout.

import sys
import csv
import json
import math
import time
import re
import threading
import argparse
import os
from collections import deque
from io import StringIO
from pathlib import Path

import serial


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSI_FILE = SCRIPT_DIR / 'csi_data.csv'
DEFAULT_LOG_FILE = SCRIPT_DIR / 'csi_data_log.txt'
DEFAULT_ACK_FILE = SCRIPT_DIR / 'ack_data.csv'
DEFAULT_ACK_PDR_FILE = SCRIPT_DIR / 'ack_pdr.csv'
DEFAULT_SEND_LOG_FILE = SCRIPT_DIR / 'send_serial_log.txt'


COLOR_STDOUT_ENABLED = False
COLOR_STDERR_ENABLED = False
ANSI_RESET = '\033[0m'
ANSI_COLORS = {
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
}
ANSI_BOLD = '\033[1m'


DATA_COLUMNS_NAMES_C5C6 = [
    'type', 'seq', 'mac', 'rssi', 'rate', 'noise_floor', 'fft_gain',
    'agc_gain', 'channel', 'local_timestamp', 'sig_len', 'rx_state',
    'len', 'first_word', 'data'
]

DATA_COLUMNS_NAMES_LEGACY = [
    'type', 'id', 'mac', 'rssi', 'rate', 'sig_mode', 'mcs', 'bandwidth',
    'smoothing', 'not_sounding', 'aggregation', 'stbc', 'fec_coding',
    'sgi', 'noise_floor', 'ampdu_cnt', 'channel', 'secondary_channel',
    'local_timestamp', 'ant', 'sig_len', 'rx_state', 'len', 'first_word',
    'data'
]

DATA_COLUMNS_NAMES_NEW = [
    'type', 'mac', 'len', 'first_word', 'data'
]

OUTPUT_COLUMNS = [
    'host_time',
    'format',
    'seq_or_id',
    'mac',
    'rssi',
    'rate',
    'noise_floor',
    'fft_gain',
    'agc_gain',
    'channel',
    'local_timestamp',
    'local_timestamp_fmt',
    'sig_len',
    'rx_state',
    'data_len',
    'first_word',
    'iq_pairs',
    'mean_amp',
    'min_amp',
    'max_amp',
    'sample_amps',
    'sample_phases',
    'data_json',
]

ACK_DATA_COLUMNS = [
    'host_time',
    'esp_time_us',
    'send_time_us',
    'service_us',
    'configured_rate',
    'actual_rate',
    'tx_status',
    'tx_data_len',
    'termination_event',
    'segment_id',
    'mcs_index',
    'seq',
    'delivered',
    'running_sent',
    'running_delivered',
    'running_pdr_percent',
    'rolling_window_size',
    'rolling_delivered',
    'rolling_pdr_percent',
]

ACK_PDR_COLUMNS = [
    'host_time',
    'esp_time_us',
    'source',
    'segment_id',
    'mcs_index',
    'seq',
    'sent',
    'delivered',
    'pdr_percent',
    'rolling_window_size',
    'rolling_delivered',
    'rolling_pdr_percent',
]


def safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def configure_terminal_colors(color_mode):
    global COLOR_STDOUT_ENABLED, COLOR_STDERR_ENABLED

    no_color = os.environ.get('NO_COLOR') is not None
    if color_mode == 'always':
        COLOR_STDOUT_ENABLED = True
        COLOR_STDERR_ENABLED = True
        return
    if color_mode == 'never':
        COLOR_STDOUT_ENABLED = False
        COLOR_STDERR_ENABLED = False
        return

    # auto mode
    COLOR_STDOUT_ENABLED = sys.stdout.isatty() and not no_color
    COLOR_STDERR_ENABLED = sys.stderr.isatty() and not no_color


def color_text(text, color=None, bold=False, stream='stdout'):
    enabled = COLOR_STDOUT_ENABLED if stream == 'stdout' else COLOR_STDERR_ENABLED
    if not enabled:
        return text

    parts = []
    if bold:
        parts.append(ANSI_BOLD)
    if color in ANSI_COLORS:
        parts.append(ANSI_COLORS[color])
    if not parts:
        return text
    return ''.join(parts) + text + ANSI_RESET


def pdr_color_name(pdr_percent):
    if pdr_percent >= 80.0:
        return 'green'
    if pdr_percent >= 50.0:
        return 'yellow'
    return 'red'


def color_pdr_percent(pdr_percent):
    return color_text(f'{pdr_percent:.2f}%', color=pdr_color_name(pdr_percent), bold=True)


def host_timestamp_ms():
    now_ns = time.time_ns()
    seconds = now_ns // 1_000_000_000
    millis = (now_ns // 1_000_000) % 1000
    micros = (now_ns // 1_000) % 1000
    return f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(seconds))}.{millis:03d}{micros:03d}"


def format_local_timestamp_us(local_timestamp_us):
    if local_timestamp_us is None:
        return ''

    total_us = int(local_timestamp_us)
    total_seconds, micros = divmod(total_us, 1_000_000)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{micros:06d}'


def build_ack_status_rows(seq, delivered, ack_total, ack_delivered,
                          rolling_window, segment_id, mcs_index,
                          esp_time_us=None, send_time_us=None, service_us=None,
                          configured_rate=None, actual_rate=None,
                          tx_status=None, tx_data_len=None,
                          termination_event=None):
    rolling_sent = len(rolling_window)
    rolling_delivered = sum(rolling_window)
    rolling_pdr_percent = 100.0 * rolling_delivered / rolling_sent if rolling_sent > 0 else 0.0
    running_pdr_percent = 100.0 * ack_delivered / ack_total if ack_total > 0 else 0.0
    host_time = host_timestamp_ms()
    mcs_value = '' if mcs_index is None else mcs_index

    ack_row = {
        'host_time': host_time,
        'esp_time_us': esp_time_us if esp_time_us is not None else '',
        'send_time_us': send_time_us if send_time_us is not None else '',
        'service_us': service_us if service_us is not None else '',
        'configured_rate': configured_rate if configured_rate is not None else '',
        'actual_rate': actual_rate if actual_rate is not None else '',
        'tx_status': tx_status if tx_status is not None else '',
        'tx_data_len': tx_data_len if tx_data_len is not None else '',
        'termination_event': termination_event if termination_event is not None else '',
        'segment_id': segment_id,
        'mcs_index': mcs_value,
        'seq': seq,
        'delivered': delivered,
        'running_sent': ack_total,
        'running_delivered': ack_delivered,
        'running_pdr_percent': f'{running_pdr_percent:.4f}',
        'rolling_window_size': rolling_sent,
        'rolling_delivered': rolling_delivered,
        'rolling_pdr_percent': f'{rolling_pdr_percent:.4f}',
    }

    ack_pdr_row = {
        'host_time': host_time,
        'esp_time_us': esp_time_us if esp_time_us is not None else '',
        'source': 'ACK_STATUS',
        'segment_id': segment_id,
        'mcs_index': mcs_value,
        'seq': seq,
        'sent': ack_total,
        'delivered': ack_delivered,
        'pdr_percent': f'{running_pdr_percent:.4f}',
        'rolling_window_size': rolling_sent,
        'rolling_delivered': rolling_delivered,
        'rolling_pdr_percent': f'{rolling_pdr_percent:.4f}',
    }

    return ack_row, ack_pdr_row, running_pdr_percent


ACK_STATUS_PATTERN = re.compile(
    r'^ACK_STATUS\s*,\s*(\d+)\s*,\s*([01])'
    r'(?:\s*,\s*(\d+))?'
    r'(?:\s*,\s*(\d+))?'
    r'(?:\s*,\s*(-?\d+))?\s*$'
)
ACK_PDR_PATTERN = re.compile(r'^ACK_PDR\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9]+(?:\.[0-9]+)?)(?:\s*,\s*(\d+))?\s*$')
ACK_PDR_FINAL_PATTERN = re.compile(r'^ACK_PDR_FINAL\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9]+(?:\.[0-9]+)?)(?:\s*,\s*(\d+))?\s*$')
ACK_RESET_PATTERN = re.compile(r'ACK_RESET_FOR_MCS(\d+)')


def parse_ack_line(line):
    """
    Parse one sender serial line for ACK metrics.
    Returns: (ack_record_dict, error_string)
    """
    if line.startswith('ACK_STATUS'):
        fields = [f.strip() for f in line.split(',')]
        if len(fields) < 3:
            return None, 'invalid_ack_status_fields'

        seq = safe_int(fields[1])
        delivered = safe_int(fields[2])
        termination_event = None
        esp_time_us = None
        send_time_us = None
        service_us = None
        configured_rate = None
        actual_rate = None
        tx_status = None
        tx_data_len = None

        idx = 3
        if len(fields) > idx:
            # New format inserts termination_event before timing fields.
            if fields[idx] and not re.fullmatch(r'-?\d+', fields[idx]):
                termination_event = fields[idx]
                idx += 1

        if len(fields) > idx:
            esp_time_us = safe_int(fields[idx])
        if len(fields) > idx + 1:
            send_time_us = safe_int(fields[idx + 1])
        if len(fields) > idx + 2:
            service_us = safe_int(fields[idx + 2])
        if len(fields) > idx + 3:
            configured_rate = safe_int(fields[idx + 3])
        if len(fields) > idx + 4:
            actual_rate = safe_int(fields[idx + 4])
        if len(fields) > idx + 5:
            tx_status = safe_int(fields[idx + 5])
        if len(fields) > idx + 6:
            tx_data_len = safe_int(fields[idx + 6])

        if seq is None or delivered not in (0, 1):
            return None, 'invalid_ack_status_fields'

        if not termination_event:
            termination_event = 'final_ack' if delivered == 1 else 'packet_drop'

        return {
            'type': 'ACK_STATUS',
            'seq': seq,
            'delivered': delivered,
            'esp_time_us': esp_time_us,
            'send_time_us': send_time_us,
            'service_us': service_us,
            'configured_rate': configured_rate,
            'actual_rate': actual_rate,
            'tx_status': tx_status,
            'tx_data_len': tx_data_len,
            'termination_event': termination_event,
        }, None

    pdr_match = ACK_PDR_PATTERN.search(line)
    if pdr_match:
        sent = safe_int(pdr_match.group(1))
        delivered = safe_int(pdr_match.group(2))
        pdr = safe_float(pdr_match.group(3))
        esp_time_us = safe_int(pdr_match.group(4))
        if sent is None or delivered is None or pdr is None:
            return None, 'invalid_ack_pdr_fields'
        return {
            'type': 'ACK_PDR',
            'sent': sent,
            'delivered': delivered,
            'pdr_percent': pdr,
            'esp_time_us': esp_time_us,
        }, None

    pdr_final_match = ACK_PDR_FINAL_PATTERN.search(line)
    if pdr_final_match:
        sent = safe_int(pdr_final_match.group(1))
        delivered = safe_int(pdr_final_match.group(2))
        pdr = safe_float(pdr_final_match.group(3))
        esp_time_us = safe_int(pdr_final_match.group(4))
        if sent is None or delivered is None or pdr is None:
            return None, 'invalid_ack_pdr_final_fields'
        return {
            'type': 'ACK_PDR_FINAL',
            'sent': sent,
            'delivered': delivered,
            'pdr_percent': pdr,
            'esp_time_us': esp_time_us,
        }, None

    if line.startswith('ACK_SERVICE,'):
        fields = [f.strip() for f in line.split(',')]
        if len(fields) not in (5, 6, 7):
            return None, 'invalid_ack_service_fields'

        total = safe_int(fields[1])
        delivered = safe_int(fields[2])
        avg_service_us = safe_float(fields[3])
        median_service_us = None
        if len(fields) >= 7:
            median_service_us = safe_float(fields[4])
            goodput_mbps = safe_float(fields[5])
            esp_time_us = safe_int(fields[6])
        else:
            goodput_mbps = safe_float(fields[4])
            esp_time_us = safe_int(fields[5]) if len(fields) == 6 else None
        if total is None or delivered is None or avg_service_us is None or goodput_mbps is None:
            return None, 'invalid_ack_service_fields'
        return {
            'type': 'ACK_SERVICE',
            'total': total,
            'delivered': delivered,
            'avg_service_us': avg_service_us,
            'median_service_us': median_service_us,
            'goodput_mbps': goodput_mbps,
            'esp_time_us': esp_time_us,
        }, None

    if line.startswith('ACK_SERVICE_FINAL,'):
        fields = [f.strip() for f in line.split(',')]
        if len(fields) not in (5, 6, 7):
            return None, 'invalid_ack_service_final_fields'

        total = safe_int(fields[1])
        delivered = safe_int(fields[2])
        avg_service_us = safe_float(fields[3])
        median_service_us = None
        if len(fields) >= 7:
            median_service_us = safe_float(fields[4])
            goodput_mbps = safe_float(fields[5])
            esp_time_us = safe_int(fields[6])
        else:
            goodput_mbps = safe_float(fields[4])
            esp_time_us = safe_int(fields[5]) if len(fields) == 6 else None
        if total is None or delivered is None or avg_service_us is None or goodput_mbps is None:
            return None, 'invalid_ack_service_final_fields'
        return {
            'type': 'ACK_SERVICE_FINAL',
            'total': total,
            'delivered': delivered,
            'avg_service_us': avg_service_us,
            'median_service_us': median_service_us,
            'goodput_mbps': goodput_mbps,
            'esp_time_us': esp_time_us,
        }, None

    reset_match = ACK_RESET_PATTERN.search(line)
    if reset_match:
        mcs_index = safe_int(reset_match.group(1))
        if mcs_index is None:
            return None, 'invalid_ack_reset_fields'
        return {
            'type': 'ACK_RESET',
            'mcs_index': mcs_index,
        }, None

    return None, 'not_ack'


def parse_csi_line(line):
    """
    Parse one serial line.
    Returns: (frame_dict, error_string)
    """
    index = line.find('CSI_DATA')
    if index == -1:
        return None, 'not_csi'

    line = line[index:].strip()

    try:
        fields = next(csv.reader(StringIO(line)))
    except Exception as e:
        return None, f'csv_parse_error: {e}'

    if len(fields) == len(DATA_COLUMNS_NAMES_C5C6):
        names = DATA_COLUMNS_NAMES_C5C6
        fmt = 'c5c6'
        seq_or_id = fields[1]
    elif len(fields) == len(DATA_COLUMNS_NAMES_LEGACY):
        names = DATA_COLUMNS_NAMES_LEGACY
        fmt = 'legacy'
        seq_or_id = fields[1]
    elif len(fields) == len(DATA_COLUMNS_NAMES_NEW):
        names = DATA_COLUMNS_NAMES_NEW
        fmt = 'new'
        seq_or_id = ''
    else:
        return None, f'wrong_field_count: got {len(fields)}'

    row = dict(zip(names, fields))

    data_len = safe_int(fields[-3])
    if data_len is None:
        return None, 'invalid_len_field'

    try:
        raw_data = json.loads(fields[-1])
    except json.JSONDecodeError:
        return None, 'data_incomplete_or_invalid_json'

    if not isinstance(raw_data, list):
        return None, 'data_not_list'

    if data_len != len(raw_data):
        return None, f'data_len_mismatch: header={data_len}, actual={len(raw_data)}'

    if len(raw_data) % 2 != 0:
        return None, f'odd_number_of_iq_values: {len(raw_data)}'

    frame = {
        'format': fmt,
        'seq_or_id': seq_or_id,
        'mac': row.get('mac', ''),
        'rssi': safe_int(row.get('rssi')),
        'rate': safe_int(row.get('rate')),
        'noise_floor': safe_int(row.get('noise_floor')),
        'fft_gain': safe_int(row.get('fft_gain')),
        'agc_gain': safe_int(row.get('agc_gain')),
        'channel': safe_int(row.get('channel')),
        'local_timestamp': safe_int(row.get('local_timestamp')),
        'sig_len': safe_int(row.get('sig_len')),
        'rx_state': safe_int(row.get('rx_state')),
        'data_len': data_len,
        'first_word': safe_int(row.get('first_word')),
        'raw_data': raw_data,
        'line': line,
    }

    return frame, None


def compute_stats(raw_data, sample_count):
    """
    raw_data format:
    [imag1, real1, imag2, real2, ...]
    """
    iq_pairs = len(raw_data) // 2

    if iq_pairs == 0:
        return {
            'iq_pairs': 0,
            'mean_amp': '',
            'min_amp': '',
            'max_amp': '',
            'sample_amps': '[]',
            'sample_phases': '[]',
        }

    amplitudes = []
    phases = []

    for i in range(0, len(raw_data), 2):
        imag = float(raw_data[i])
        real = float(raw_data[i + 1])
        amplitudes.append(math.hypot(real, imag))
        phases.append(math.atan2(imag, real))

    mean_amp = sum(amplitudes) / len(amplitudes)
    min_amp = min(amplitudes)
    max_amp = max(amplitudes)

    sample_amps = amplitudes[:sample_count]
    sample_phases = phases[:sample_count]

    return {
        'iq_pairs': iq_pairs,
        'mean_amp': f'{mean_amp:.6f}',
        'min_amp': f'{min_amp:.6f}',
        'max_amp': f'{max_amp:.6f}',
        'sample_amps': json.dumps([round(x, 6) for x in sample_amps], separators=(',', ':')),
        'sample_phases': json.dumps([round(x, 6) for x in sample_phases], separators=(',', ':')),
    }


def make_output_row(frame, stats):
    local_timestamp_us = frame['local_timestamp'] if frame['local_timestamp'] is not None else ''
    return {
        'host_time': host_timestamp_ms(),
        'format': frame['format'],
        'seq_or_id': frame['seq_or_id'],
        'mac': frame['mac'],
        'rssi': frame['rssi'] if frame['rssi'] is not None else '',
        'rate': frame['rate'] if frame['rate'] is not None else '',
        'noise_floor': frame['noise_floor'] if frame['noise_floor'] is not None else '',
        'fft_gain': frame['fft_gain'] if frame['fft_gain'] is not None else '',
        'agc_gain': frame['agc_gain'] if frame['agc_gain'] is not None else '',
        'channel': frame['channel'] if frame['channel'] is not None else '',
        'local_timestamp': local_timestamp_us,
        'local_timestamp_fmt': format_local_timestamp_us(frame['local_timestamp']),
        'sig_len': frame['sig_len'] if frame['sig_len'] is not None else '',
        'rx_state': frame['rx_state'] if frame['rx_state'] is not None else '',
        'data_len': frame['data_len'],
        'first_word': frame['first_word'] if frame['first_word'] is not None else '',
        'iq_pairs': stats['iq_pairs'],
        'mean_amp': stats['mean_amp'],
        'min_amp': stats['min_amp'],
        'max_amp': stats['max_amp'],
        'sample_amps': stats['sample_amps'],
        'sample_phases': stats['sample_phases'],
        'data_json': json.dumps(frame['raw_data'], separators=(',', ':')),
    }


def print_summary(frame_count, frame, stats):
    msg = (
        f"[{frame_count}] "
        f"fmt={frame['format']} "
        f"mac={frame['mac']} "
        f"rssi={frame['rssi']} "
        f"ch={frame['channel']} "
        f"pairs={stats['iq_pairs']} "
        f"mean_amp={stats['mean_amp']} "
        f"max_amp={stats['max_amp']}"
    )

    if frame['fft_gain'] is not None:
        msg += f" fft_gain={frame['fft_gain']}"
    if frame['agc_gain'] is not None:
        msg += f" agc_gain={frame['agc_gain']}"

    print(msg, flush=True)


def print_ack_summary(ack_total, ack_delivered, running_pdr_percent, seq,
                      rolling_sent, rolling_delivered, rolling_pdr_percent):
    tag = color_text('[ACK]', color='cyan', bold=True)
    print(
        f'{tag} seq={seq} delivered_total={ack_delivered}/{ack_total} '
        f'running_pdr={color_pdr_percent(running_pdr_percent)} '
        f'window={rolling_delivered}/{rolling_sent} '
        f'window_pdr={color_pdr_percent(rolling_pdr_percent)}',
        flush=True,
    )


def infer_single_port_mode(baudrate):
    # Typical setup: sender console at 115200, receiver CSI stream at 921600.
    # Allow auto mode to infer likely stream type from baudrate.
    if baudrate <= 230400:
        return 'ack'
    return 'csi'


def ack_read_parse(send_port, send_baudrate, ack_writer, ack_file_fd,
                   ack_pdr_writer, ack_pdr_file_fd, send_log_fd,
                   ack_summary_every, ack_window_size,
                   stop_event):
    if ack_summary_every < 0:
        ack_summary_every = 0
    if ack_window_size <= 0:
        ack_window_size = 100

    try:
        ser = serial.Serial(
            port=send_port,
            baudrate=send_baudrate,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=1,
        )
    except serial.SerialException as e:
        print(f'Failed to open sender serial port {send_port}: {e}', file=sys.stderr)
        return 1

    if ser.is_open:
        print(f'Opened sender serial port: {send_port} @ {send_baudrate}', flush=True)
    else:
        print(f'Failed to open sender serial port: {send_port}', file=sys.stderr)
        return 1

    ack_total = 0
    ack_delivered = 0
    ack_non_ack_count = 0
    ack_bad_count = 0
    segment_id = 0
    current_mcs_index = 0
    rolling_window = deque(maxlen=ack_window_size)

    try:
        while not stop_event.is_set():
            raw = ser.readline()
            if not raw:
                continue

            line = raw.decode('utf-8', errors='replace').strip()
            if not line:
                continue

            record, error = parse_ack_line(line)

            if error == 'not_ack':
                ack_non_ack_count += 1
                send_log_fd.write(f'[{host_timestamp_ms()}] not_ack\n')
                send_log_fd.write(line + '\n')
                send_log_fd.flush()
                continue

            if error is not None:
                ack_bad_count += 1
                send_log_fd.write(f'[{host_timestamp_ms()}] {error}\n')
                send_log_fd.write(line + '\n')
                send_log_fd.flush()
                continue

            if record is None:
                ack_bad_count += 1
                send_log_fd.write(f'[{host_timestamp_ms()}] parser_returned_none\n')
                send_log_fd.write(line + '\n')
                send_log_fd.flush()
                continue

            if record['type'] == 'ACK_STATUS':
                ack_total += 1
                ack_delivered += record['delivered']
                rolling_window.append(record['delivered'])
                ack_row, ack_pdr_row, pdr_percent = build_ack_status_rows(
                    seq=record['seq'],
                    delivered=record['delivered'],
                    ack_total=ack_total,
                    ack_delivered=ack_delivered,
                    rolling_window=rolling_window,
                    segment_id=segment_id,
                    mcs_index=current_mcs_index,
                    esp_time_us=record.get('esp_time_us'),
                    send_time_us=record.get('send_time_us'),
                    service_us=record.get('service_us'),
                    configured_rate=record.get('configured_rate'),
                    actual_rate=record.get('actual_rate'),
                    tx_status=record.get('tx_status'),
                    tx_data_len=record.get('tx_data_len'),
                    termination_event=record.get('termination_event'),
                )

                ack_writer.writerow(ack_row)
                ack_file_fd.flush()

                ack_pdr_writer.writerow(ack_pdr_row)
                ack_pdr_file_fd.flush()

                if ack_summary_every > 0 and ack_total % ack_summary_every == 0:
                    rolling_sent = len(rolling_window)
                    rolling_delivered = sum(rolling_window)
                    rolling_pdr_percent = 100.0 * rolling_delivered / rolling_sent if rolling_sent > 0 else 0.0
                    print_ack_summary(
                        ack_total=ack_total,
                        ack_delivered=ack_delivered,
                        running_pdr_percent=pdr_percent,
                        seq=record['seq'],
                        rolling_sent=rolling_sent,
                        rolling_delivered=rolling_delivered,
                        rolling_pdr_percent=rolling_pdr_percent,
                    )

                continue

            if record['type'] == 'ACK_PDR':
                ack_pdr_writer.writerow({
                    'host_time': host_timestamp_ms(),
                    'esp_time_us': record.get('esp_time_us', ''),
                    'source': 'ACK_PDR',
                    'segment_id': segment_id,
                    'mcs_index': current_mcs_index,
                    'seq': '',
                    'sent': record['sent'],
                    'delivered': record['delivered'],
                    'pdr_percent': f"{record['pdr_percent']:.4f}",
                    'rolling_window_size': '',
                    'rolling_delivered': '',
                    'rolling_pdr_percent': '',
                })
                ack_pdr_file_fd.flush()
                ack_pdr_tag = color_text('[ACK_PDR]', color='magenta', bold=True)
                print(
                    f"{ack_pdr_tag} sent={record['sent']} delivered={record['delivered']} "
                    f"pdr={color_pdr_percent(record['pdr_percent'])}",
                    flush=True,
                )

                continue

            if record['type'] == 'ACK_PDR_FINAL':
                ack_pdr_writer.writerow({
                    'host_time': host_timestamp_ms(),
                    'esp_time_us': record.get('esp_time_us', ''),
                    'source': 'ACK_PDR_FINAL',
                    'segment_id': segment_id,
                    'mcs_index': current_mcs_index,
                    'seq': '',
                    'sent': record['sent'],
                    'delivered': record['delivered'],
                    'pdr_percent': f"{record['pdr_percent']:.4f}",
                    'rolling_window_size': '',
                    'rolling_delivered': '',
                    'rolling_pdr_percent': '',
                })
                ack_pdr_file_fd.flush()
                ack_pdr_final_tag = color_text('[ACK_PDR_FINAL]', color='green', bold=True)
                print(
                    f"{ack_pdr_final_tag} sent={record['sent']} delivered={record['delivered']} "
                    f"pdr={color_pdr_percent(record['pdr_percent'])} (MCS rate switch complete)",
                    flush=True,
                )

                continue

            if record['type'] == 'ACK_SERVICE':
                ack_service_tag = color_text('[ACK_SERVICE]', color='cyan', bold=True)
                median_text = (
                    f" median_service_us={record['median_service_us']:.1f}"
                    if record.get('median_service_us') is not None else ""
                )
                print(
                    f"{ack_service_tag} total={record['total']} delivered={record['delivered']} "
                    f"avg_service_us={record['avg_service_us']:.1f}{median_text} "
                    f"goodput_mbps={record['goodput_mbps']:.3f}",
                    flush=True,
                )

                continue

            if record['type'] == 'ACK_SERVICE_FINAL':
                ack_service_final_tag = color_text('[ACK_SERVICE_FINAL]', color='blue', bold=True)
                median_text = (
                    f" median_service_us={record['median_service_us']:.1f}"
                    if record.get('median_service_us') is not None else ""
                )
                print(
                    f"{ack_service_final_tag} total={record['total']} delivered={record['delivered']} "
                    f"avg_service_us={record['avg_service_us']:.1f}{median_text} "
                    f"goodput_mbps={record['goodput_mbps']:.3f} "
                    f"(MCS rate switch complete)",
                    flush=True,
                )

                continue

            if record['type'] == 'ACK_RESET':
                ack_reset_tag = color_text('[ACK_RESET]', color='yellow', bold=True)
                print(
                    f"{ack_reset_tag} Resetting counters for MCS{record['mcs_index']}",
                    flush=True,
                )
                # Reset running counters for next MCS test
                segment_id += 1
                current_mcs_index = record['mcs_index']
                ack_total = 0
                ack_delivered = 0
                rolling_window.clear()
                continue

    finally:
        ser.close()
        print('Sender ACK reader stopped.', flush=True)
        print(f'Sender non-ACK lines logged: {ack_non_ack_count}', flush=True)
        print(f'Sender malformed ACK lines logged: {ack_bad_count}', flush=True)

    return 0


def csi_data_read_parse(port, baudrate, csv_writer, csv_file_fd, log_file_fd,
                        summary_every, sample_count, flush_every,
                        ack_writer=None, ack_file_fd=None,
                        ack_pdr_writer=None, ack_pdr_file_fd=None,
                        ack_summary_every=0,
                        ack_window_size=100,
                        single_port_mode='auto'):
    if flush_every <= 0:
        flush_every = 1
    if ack_window_size <= 0:
        ack_window_size = 100

    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=1,
        )
    except serial.SerialException as e:
        print(f'Failed to open serial port {port}: {e}', file=sys.stderr)
        return 1

    if ser.is_open:
        print(f'Opened serial port: {port} @ {baudrate}', flush=True)
    else:
        print(f'Failed to open serial port: {port}', file=sys.stderr)
        return 1

    frame_count = 0
    bad_count = 0
    ack_total = 0
    ack_delivered = 0
    segment_id = 0
    current_mcs_index = 0
    rolling_window = deque(maxlen=ack_window_size)

    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue

            line = raw.decode('utf-8', errors='replace').strip()
            if not line:
                continue

            frame = None
            error = 'not_csi'
            if single_port_mode in ('auto', 'csi'):
                frame, error = parse_csi_line(line)

            if error is None and frame is not None:
                stats = compute_stats(frame['raw_data'], sample_count)
                csv_writer.writerow(make_output_row(frame, stats))
                frame_count += 1

                if frame_count % flush_every == 0:
                    csv_file_fd.flush()
                    log_file_fd.flush()
                    if ack_file_fd is not None:
                        ack_file_fd.flush()
                    if ack_pdr_file_fd is not None:
                        ack_pdr_file_fd.flush()

                if summary_every > 0 and frame_count % summary_every == 0:
                    print_summary(frame_count, frame, stats)

                continue

            # If not CSI, try sender ACK formats on the same port.
            if single_port_mode in ('auto', 'ack') and error == 'not_csi':
                ack_record, ack_error = parse_ack_line(line)
                if ack_error is None and ack_record is not None:
                    if ack_record['type'] == 'ACK_STATUS':
                        ack_total += 1
                        ack_delivered += ack_record['delivered']
                        rolling_window.append(ack_record['delivered'])
                        ack_row, ack_pdr_row, pdr_percent = build_ack_status_rows(
                            seq=ack_record['seq'],
                            delivered=ack_record['delivered'],
                            ack_total=ack_total,
                            ack_delivered=ack_delivered,
                            rolling_window=rolling_window,
                            segment_id=segment_id,
                            mcs_index=current_mcs_index,
                            esp_time_us=ack_record.get('esp_time_us'),
                            send_time_us=ack_record.get('send_time_us'),
                            service_us=ack_record.get('service_us'),
                            configured_rate=ack_record.get('configured_rate'),
                            actual_rate=ack_record.get('actual_rate'),
                            tx_status=ack_record.get('tx_status'),
                            tx_data_len=ack_record.get('tx_data_len'),
                            termination_event=ack_record.get('termination_event'),
                        )

                        if ack_writer is not None:
                            ack_writer.writerow(ack_row)
                            if ack_file_fd is not None:
                                ack_file_fd.flush()

                        if ack_pdr_writer is not None:
                            ack_pdr_writer.writerow(ack_pdr_row)
                            if ack_pdr_file_fd is not None:
                                ack_pdr_file_fd.flush()

                        if ack_summary_every > 0 and ack_total % ack_summary_every == 0:
                            rolling_sent = len(rolling_window)
                            rolling_delivered = sum(rolling_window)
                            rolling_pdr_percent = 100.0 * rolling_delivered / rolling_sent if rolling_sent > 0 else 0.0
                            print_ack_summary(
                                ack_total=ack_total,
                                ack_delivered=ack_delivered,
                                running_pdr_percent=pdr_percent,
                                seq=ack_record['seq'],
                                rolling_sent=rolling_sent,
                                rolling_delivered=rolling_delivered,
                                rolling_pdr_percent=rolling_pdr_percent,
                            )

                        continue

                    if ack_record['type'] == 'ACK_PDR':
                        if ack_pdr_writer is not None:
                            ack_pdr_writer.writerow({
                                'host_time': host_timestamp_ms(),
                                'esp_time_us': ack_record.get('esp_time_us', ''),
                                'source': 'ACK_PDR',
                                'segment_id': segment_id,
                                'mcs_index': current_mcs_index,
                                'seq': '',
                                'sent': ack_record['sent'],
                                'delivered': ack_record['delivered'],
                                'pdr_percent': f"{ack_record['pdr_percent']:.4f}",
                                'rolling_window_size': '',
                                'rolling_delivered': '',
                                'rolling_pdr_percent': '',
                            })
                            if ack_pdr_file_fd is not None:
                                ack_pdr_file_fd.flush()
                        ack_pdr_tag = color_text('[ACK_PDR]', color='magenta', bold=True)
                        print(
                            f"{ack_pdr_tag} sent={ack_record['sent']} delivered={ack_record['delivered']} "
                            f"pdr={color_pdr_percent(ack_record['pdr_percent'])}",
                            flush=True,
                        )
                        continue

                    if ack_record['type'] == 'ACK_PDR_FINAL':
                        if ack_pdr_writer is not None:
                            ack_pdr_writer.writerow({
                                'host_time': host_timestamp_ms(),
                                'esp_time_us': ack_record.get('esp_time_us', ''),
                                'source': 'ACK_PDR_FINAL',
                                'segment_id': segment_id,
                                'mcs_index': current_mcs_index,
                                'seq': '',
                                'sent': ack_record['sent'],
                                'delivered': ack_record['delivered'],
                                'pdr_percent': f"{ack_record['pdr_percent']:.4f}",
                                'rolling_window_size': '',
                                'rolling_delivered': '',
                                'rolling_pdr_percent': '',
                            })
                            if ack_pdr_file_fd is not None:
                                ack_pdr_file_fd.flush()
                        ack_pdr_final_tag = color_text('[ACK_PDR_FINAL]', color='green', bold=True)
                        print(
                            f"{ack_pdr_final_tag} sent={ack_record['sent']} delivered={ack_record['delivered']} "
                            f"pdr={color_pdr_percent(ack_record['pdr_percent'])} (MCS rate switch complete)",
                            flush=True,
                        )
                        continue

                    if ack_record['type'] == 'ACK_SERVICE':
                        ack_service_tag = color_text('[ACK_SERVICE]', color='cyan', bold=True)
                        median_text = (
                            f" median_service_us={ack_record['median_service_us']:.1f}"
                            if ack_record.get('median_service_us') is not None else ""
                        )
                        print(
                            f"{ack_service_tag} total={ack_record['total']} delivered={ack_record['delivered']} "
                            f"avg_service_us={ack_record['avg_service_us']:.1f}{median_text} "
                            f"goodput_mbps={ack_record['goodput_mbps']:.3f}",
                            flush=True,
                        )
                        continue

                    if ack_record['type'] == 'ACK_SERVICE_FINAL':
                        ack_service_final_tag = color_text('[ACK_SERVICE_FINAL]', color='blue', bold=True)
                        median_text = (
                            f" median_service_us={ack_record['median_service_us']:.1f}"
                            if ack_record.get('median_service_us') is not None else ""
                        )
                        print(
                            f"{ack_service_final_tag} total={ack_record['total']} delivered={ack_record['delivered']} "
                            f"avg_service_us={ack_record['avg_service_us']:.1f}{median_text} "
                            f"goodput_mbps={ack_record['goodput_mbps']:.3f} "
                            f"(MCS rate switch complete)",
                            flush=True,
                        )
                        continue

                    if ack_record['type'] == 'ACK_RESET':
                        ack_reset_tag = color_text('[ACK_RESET]', color='yellow', bold=True)
                        print(
                            f"{ack_reset_tag} Resetting counters for MCS{ack_record['mcs_index']}",
                            flush=True,
                        )
                        segment_id += 1
                        current_mcs_index = ack_record['mcs_index']
                        ack_total = 0
                        ack_delivered = 0
                        rolling_window.clear()
                        continue

                # Neither CSI nor ACK: log as unknown serial line.
                bad_count += 1
                log_file_fd.write(f'[{host_timestamp_ms()}] unknown_line\n')
                log_file_fd.write(line + '\n')
                log_file_fd.flush()
                continue

            # CSI-like line but malformed.
            bad_count += 1
            log_file_fd.write(f'[{host_timestamp_ms()}] {error}\n')
            log_file_fd.write(line + '\n')
            log_file_fd.flush()
            continue

    except KeyboardInterrupt:
        print('\nStopped by user.', flush=True)
        print(f'Valid CSI frames: {frame_count}', flush=True)
        if ack_total > 0:
            print(f'Valid ACK_STATUS rows: {ack_total}', flush=True)
            print(f'ACK delivered: {ack_delivered}', flush=True)
            print(f'ACK running PDR: {100.0 * ack_delivered / ack_total:.2f}%', flush=True)
        print(f'Invalid/non-CSI lines logged: {bad_count}', flush=True)
    finally:
        ser.close()

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Read CSI data from serial port over SSH without any GUI.'
    )
    parser.add_argument(
        '-p', '--port',
        required=True,
        help='Serial port of csi_recv or csi_send device, e.g. /dev/ttyUSB0'
    )
    parser.add_argument(
        '-b', '--baud',
        type=int,
        default=921600,
        help='Serial baud rate (default: 921600)'
    )
    parser.add_argument(
        '-s', '--store',
        dest='store_file',
        default=str(DEFAULT_CSI_FILE),
        help='CSV file to save parsed CSI data'
    )
    parser.add_argument(
        '-l', '--log',
        dest='log_file',
        default=str(DEFAULT_LOG_FILE),
        help='Log file for non-CSI and malformed lines'
    )
    parser.add_argument(
        '--summary-every',
        type=int,
        default=20,
        help='Print one terminal summary every N valid CSI frames (default: 20, 0 disables)'
    )
    parser.add_argument(
        '--sample-count',
        type=int,
        default=8,
        help='How many amplitude/phase samples to keep in CSV preview fields (default: 8)'
    )
    parser.add_argument(
        '--flush-every',
        type=int,
        default=10,
        help='Flush files every N valid CSI frames (default: 10)'
    )
    parser.add_argument(
        '-sp', '--send-port',
        default=None,
        help='Optional serial port of csi_send device for ACK metrics, e.g. /dev/ttyUSB0'
    )
    parser.add_argument(
        '-sb', '--send-baud',
        type=int,
        default=921600,
        help='Sender serial baud rate for ACK metrics (default: 921600)'
    )
    parser.add_argument(
        '-a', '--ack-store',
        dest='ack_store_file',
        default=str(DEFAULT_ACK_FILE),
        help='CSV file to save ACK_STATUS running metrics from csi_send'
    )
    parser.add_argument(
        '-ap', '--ack-pdr-store',
        dest='ack_pdr_store_file',
        default=str(DEFAULT_ACK_PDR_FILE),
        help='CSV file to save sender ACK PDR rows (ACK_STATUS-derived + ACK_PDR firmware rows)'
    )
    parser.add_argument(
        '--send-log',
        dest='send_log_file',
        default=str(DEFAULT_SEND_LOG_FILE),
        help='Log file for sender non-ACK and malformed ACK lines'
    )
    parser.add_argument(
        '--ack-summary-every',
        type=int,
        default=100,
        help='Print ACK summary every N ACK_STATUS rows (default: 100, 0 disables)'
    )
    parser.add_argument(
        '--ack-window-size',
        type=int,
        default=100,
        help='Rolling ACK window size in packets for rolling PDR fields (default: 100)'
    )
    parser.add_argument(
        '--single-port-mode',
        choices=['auto', 'csi', 'ack'],
        default='auto',
        help=(
            'Parsing mode when only -p is used: auto infers from -b '
            '(<=230400 -> ack, otherwise csi).'
        )
    )
    parser.add_argument(
        '--color',
        choices=['auto', 'always', 'never'],
        default='auto',
        help='Terminal color mode for important ACK/PDR lines (default: auto)'
    )

    args = parser.parse_args()
    configure_terminal_colors(args.color)

    # Normalize all output paths early so logs clearly show exactly where files are written.
    args.store_file = str(Path(args.store_file).expanduser().resolve())
    args.log_file = str(Path(args.log_file).expanduser().resolve())
    args.ack_store_file = str(Path(args.ack_store_file).expanduser().resolve())
    args.ack_pdr_store_file = str(Path(args.ack_pdr_store_file).expanduser().resolve())
    args.send_log_file = str(Path(args.send_log_file).expanduser().resolve())

    for output_path in (
        args.store_file,
        args.log_file,
        args.ack_store_file,
        args.ack_pdr_store_file,
        args.send_log_file,
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if sys.version_info < (3, 6):
        print('Python version should be >= 3.6', file=sys.stderr)
        sys.exit(1)

    print(f'CSI output CSV: {args.store_file}', flush=True)
    print(f'CSI log file: {args.log_file}', flush=True)
    print(f'ACK CSV: {args.ack_store_file}', flush=True)
    print(f'ACK PDR CSV: {args.ack_pdr_store_file}', flush=True)

    stop_event = threading.Event()
    ack_thread = None

    with open(args.store_file, 'w', newline='') as csv_fd, \
            open(args.log_file, 'w') as log_fd, \
            open(args.ack_store_file, 'w', newline='') as ack_fd, \
            open(args.ack_pdr_store_file, 'w', newline='') as ack_pdr_fd:
        csv_writer = csv.DictWriter(csv_fd, fieldnames=OUTPUT_COLUMNS)
        csv_writer.writeheader()

        ack_writer = csv.DictWriter(ack_fd, fieldnames=ACK_DATA_COLUMNS)
        ack_writer.writeheader()

        ack_pdr_writer = csv.DictWriter(ack_pdr_fd, fieldnames=ACK_PDR_COLUMNS)
        ack_pdr_writer.writeheader()

        if args.send_port:
            print(f'Sender log file: {args.send_log_file}', flush=True)

            with open(args.send_log_file, 'w') as send_log_fd:
                ack_thread = threading.Thread(
                    target=ack_read_parse,
                    args=(
                        args.send_port,
                        args.send_baud,
                        ack_writer,
                        ack_fd,
                        ack_pdr_writer,
                        ack_pdr_fd,
                        send_log_fd,
                        args.ack_summary_every,
                        args.ack_window_size,
                        stop_event,
                    ),
                    daemon=True,
                )
                ack_thread.start()

                try:
                    rc = csi_data_read_parse(
                        port=args.port,
                        baudrate=args.baud,
                        csv_writer=csv_writer,
                        csv_file_fd=csv_fd,
                        log_file_fd=log_fd,
                        summary_every=args.summary_every,
                        sample_count=args.sample_count,
                        flush_every=args.flush_every,
                        ack_writer=ack_writer,
                        ack_file_fd=ack_fd,
                        ack_pdr_writer=ack_pdr_writer,
                        ack_pdr_file_fd=ack_pdr_fd,
                        ack_summary_every=args.ack_summary_every,
                        ack_window_size=args.ack_window_size,
                    )
                finally:
                    stop_event.set()
                    if ack_thread is not None:
                        ack_thread.join(timeout=2)
        else:
            selected_mode = args.single_port_mode
            if selected_mode == 'auto':
                selected_mode = infer_single_port_mode(args.baud)
                print(
                    f'Single-port mode auto-selected from baud {args.baud}: {selected_mode}',
                    flush=True,
                )
            else:
                print(f'Single-port mode forced: {selected_mode}', flush=True)

            rc = csi_data_read_parse(
                port=args.port,
                baudrate=args.baud,
                csv_writer=csv_writer,
                csv_file_fd=csv_fd,
                log_file_fd=log_fd,
                summary_every=args.summary_every,
                sample_count=args.sample_count,
                flush_every=args.flush_every,
                ack_writer=ack_writer,
                ack_file_fd=ack_fd,
                ack_pdr_writer=ack_pdr_writer,
                ack_pdr_file_fd=ack_pdr_fd,
                ack_summary_every=args.ack_summary_every,
                ack_window_size=args.ack_window_size,
                single_port_mode=selected_mode,
            )

    sys.exit(rc)


if __name__ == '__main__':
    main()
