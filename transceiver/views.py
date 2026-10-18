# transceiver/views.py
import base64
import io
import json
import random
import wave
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .models import TransmissionLog
from .engine.source_coding import SourceCoder
from .engine.channel_coding import ChannelCoder
from .engine.modem_phy import AcousticModemPHY


def _extract_fsk_transition_preview(waveform: np.ndarray, samples_per_symbol: int) -> list[float]:
    """
    Extract a normalized window around the first symbol transition so the
    M-FSK frequency shift is clearly rendered on the Signal Lab chart.
    """
    if len(waveform) == 0:
        return [0.0] * 200

    boundary = min(samples_per_symbol, len(waveform) // 2)
    half_win = min(220, boundary)
    start = max(0, boundary - half_win)
    end = min(len(waveform), boundary + half_win)

    segment = waveform[start:end].astype(np.float32)
    if len(segment) == 0:
        return [0.0] * 200

    peak = float(np.max(np.abs(segment)))
    if peak > 1e-5:
        segment = (segment / peak) * 0.90

    return [round(float(x), 4) for x in segment]


def dashboard_view(request):
    """Render the main SonicDrop transceiver application."""
    logs = TransmissionLog.objects.all()[:15]
    return render(request, 'transceiver/index.html', {'logs': logs})


@csrf_exempt
def transmit_and_analyze_api(request):
    """
    High-Speed Transmission Pipeline with safe Hybrid ARQ Selective Repeat support.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required.'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
        payload_type = data.get('payload_type', 'text')
        text = data.get('text', '').strip() or 'Hello UoM!'
        fax_matrix = data.get('fax_matrix', None)
        baud_rate = int(data.get('baud_rate', 65))
        levels_m = int(data.get('levels_m', 4))
        ultrasonic = bool(data.get('ultrasonic', False))
        delivery_mode = data.get('delivery_mode', 'stream')
        use_rle = bool(data.get('use_rle', False))
        use_hamming = bool(data.get('use_hamming', True))
        noise_std = float(data.get('noise_std', 0.05))
        error_mode = data.get('error_mode', 'multiple')

        retransmit_seq = data.get('retransmit_seq', None)
        force_msg_id = data.get('msg_id', None)
        simulate_arq_nack = bool(data.get('simulate_arq_nack', False))

        stream_mode_flag = (delivery_mode == 'stream')
        is_fax = (payload_type == 'fax')
        msg_id = int(force_msg_id) if force_msg_id is not None else random.randint(1, 7)

        payloads = []
        packet_items = []
        seq_numbers = []
        is_last_flags = []
        total_raw_bits = 0
        total_rle_bits = 0
        rle_saving_percent = 0.0

        if is_fax:
            if isinstance(fax_matrix, list) and len(fax_matrix) == 16:
                raw_fax_bits = SourceCoder.binary_matrix_to_bits(fax_matrix)
            else:
                raw_fax_bits = data.get('fax_bits', '0' * 256)[:256].ljust(256, '0')

            rle_stats = SourceCoder.rle_encode(raw_fax_bits)
            compressed_fax_bits = rle_stats['encoded_bits']
            total_raw_bits = len(raw_fax_bits)
            total_rle_bits = rle_stats['compressed_len']
            rle_saving_percent = rle_stats['saving_percent']

            chunk_bit_size = 32 if stream_mode_flag else 64
            all_chunks = [
                compressed_fax_bits[i:i + chunk_bit_size]
                for i in range(0, len(compressed_fax_bits), chunk_bit_size)
            ] or [compressed_fax_bits]

            # Clamp retransmit_seq safely within bounds
            if retransmit_seq is not None:
                retransmit_seq = max(0, min(int(retransmit_seq), len(all_chunks) - 1))

            for idx, rle_chunk in enumerate(all_chunks):
                if retransmit_seq is not None and idx != retransmit_seq:
                    continue
                is_last = (idx == len(all_chunks) - 1)
                header_bits = ChannelCoder.pack_packet_header(
                    msg_id=msg_id,
                    seq_num=idx,
                    is_last=is_last,
                    stream_mode=stream_mode_flag,
                    is_fax=True
                )
                payloads.append(header_bits + rle_chunk)
                packet_items.append(rle_chunk)
                seq_numbers.append(idx)
                is_last_flags.append(is_last)

            log_label = f"[16x16 Pixel Fax | RLE {total_rle_bits}b ({rle_saving_percent}% saved)]"
        else:
            raw_words = [w for w in text.split(' ') if w] or [text]
            all_chunks = raw_words if stream_mode_flag else [
                ' '.join(raw_words[i:i + 4]) for i in range(0, len(raw_words), 4)
            ]

            # Clamp retransmit_seq safely within bounds so payloads is never empty!
            if retransmit_seq is not None:
                retransmit_seq = max(0, min(int(retransmit_seq), len(all_chunks) - 1))

            for idx, chunk_text in enumerate(all_chunks):
                if retransmit_seq is not None and idx != retransmit_seq:
                    continue
                is_last = (idx == len(all_chunks) - 1)
                c_bits = SourceCoder.text_to_bits(chunk_text)
                rle_stats = SourceCoder.rle_encode(c_bits)
                encoded_bits = rle_stats['encoded_bits'] if use_rle else c_bits

                total_raw_bits += len(c_bits)
                total_rle_bits += rle_stats['compressed_len']

                header_bits = ChannelCoder.pack_packet_header(
                    msg_id=msg_id,
                    seq_num=idx,
                    is_last=is_last,
                    stream_mode=stream_mode_flag,
                    is_fax=False
                )
                payloads.append(header_bits + encoded_bits)
                packet_items.append(chunk_text)
                seq_numbers.append(idx)
                is_last_flags.append(is_last)

            if total_raw_bits > 0:
                rle_saving_percent = round((1.0 - (total_rle_bits / total_raw_bits)) * 100, 1)
            log_label = f"[ARQ Retry #{retransmit_seq}] {packet_items[0]}" if retransmit_seq is not None else text

        corrupt_target_idx = None
        if simulate_arq_nack and retransmit_seq is None and len(payloads) > 0:
            corrupt_target_idx = min(1, len(payloads) - 1)

        bodies = []
        cumulative_bits = []
        current_len = len(ChannelCoder.PREAMBLE) + len(ChannelCoder.HDLC_FLAG)
        first_crc = "00000000"

        for idx, p_bits in enumerate(payloads):
            b_info = ChannelCoder.encode_frame_body(p_bits, use_hamming=use_hamming)
            if idx == 0:
                first_crc = b_info["crc_bits"]
            stuffed = b_info["stuffed_body"]
            bodies.append(stuffed)
            current_len += len(stuffed) + len(ChannelCoder.HDLC_FLAG)
            cumulative_bits.append(current_len)

        stream_bits = ChannelCoder.PREAMBLE + ChannelCoder.HDLC_FLAG + ChannelCoder.HDLC_FLAG.join(bodies) + ChannelCoder.HDLC_FLAG
        first_tx_bits = f"{ChannelCoder.PREAMBLE}{ChannelCoder.HDLC_FLAG}{bodies[0]}{ChannelCoder.HDLC_FLAG}"

        modem = AcousticModemPHY(
            sample_rate=44100,
            baud_rate=baud_rate,
            ultrasonic=ultrasonic,
            levels_m=levels_m
        )
        clean_waveform = modem.modulate(stream_bits)
        wav_bytes = modem.waveform_to_wav_bytes(clean_waveform)
        wav_base64 = base64.b64encode(wav_bytes).decode('ascii')

        effective_bps = modem.bit_rate_bps
        packet_schedule = []
        for idx, item_val in enumerate(packet_items):
            emit_sec = round(cumulative_bits[idx] / effective_bps, 3)
            is_corrupted_pkt = (corrupt_target_idx is not None and idx == corrupt_target_idx)
            packet_schedule.append({
                'msg_id': msg_id,
                'seq_num': seq_numbers[idx],
                'is_last': is_last_flags[idx],
                'stream_mode': stream_mode_flag,
                'is_fax': is_fax,
                'crc_corrupted': is_corrupted_pkt,
                'word': item_val if not is_fax else '',
                'rle_chunk': item_val if is_fax else '',
                'emit_time_sec': emit_sec
            })

        first_wave = modem.modulate(first_tx_bits)
        if noise_std > 0:
            rx_first_wave = first_wave + np.random.normal(0.0, noise_std, size=first_wave.shape).astype(np.float32)
        else:
            rx_first_wave = first_wave.copy()

        demod_res = modem.demodulate(rx_first_wave)
        rx_bits = demod_res['bits'][:len(first_tx_bits)]

        flipped_indices = []
        header_offset = len(ChannelCoder.PREAMBLE) + len(ChannelCoder.HDLC_FLAG)
        effective_error_mode = 'burst' if corrupt_target_idx == 0 else error_mode
        if effective_error_mode in ('single', 'multiple', 'burst') and len(rx_bits) > header_offset + 8:
            body = rx_bits[header_offset:-len(ChannelCoder.HDLC_FLAG)]
            noisy_body, local_flips = ChannelCoder.inject_channel_error(body, mode=effective_error_mode, burst_len=5)
            flipped_indices = [idx + header_offset for idx in local_flips]
            rx_bits = rx_bits[:header_offset] + noisy_body + rx_bits[-len(ChannelCoder.HDLC_FLAG):]

        bit_mismatches = sum(1 for a, b in zip(first_tx_bits, rx_bits) if a != b)
        ber = round(bit_mismatches / max(len(first_tx_bits), 1), 6)

        parsed = ChannelCoder.parse_frame(rx_bits, use_hamming=use_hamming)
        crc_passed = (corrupt_target_idx is None) and (
            parsed.get('crc_passed', False) if parsed.get('valid_frame') else False
        )
        corrected_count = parsed.get('corrected_errors_count', 0) if parsed.get('valid_frame') else 0

        metrics = demod_res['metrics']
        total_stream_bits = len(stream_bits)

        log_entry = TransmissionLog.objects.create(
            input_text=log_label,
            carrier_mode='ultrasonic' if ultrasonic else 'audible',
            baud_rate=effective_bps,
            use_rle=True if is_fax else use_rle,
            use_hamming=use_hamming,
            raw_bits_len=total_raw_bits,
            total_frame_bits=total_stream_bits,
            injected_errors=bit_mismatches,
            corrected_errors=corrected_count,
            ber=ber,
            snr_db=metrics['snr_db'],
            nyquist_bps=metrics['nyquist_max_bps'],
            shannon_bps=metrics['shannon_capacity_bps'],
            crc_passed=crc_passed,
            recovered_text=log_label
        )

        waveform_preview = _extract_fsk_transition_preview(rx_first_wave, modem.samples_per_symbol)

        return JsonResponse({
            'status': 'ok',
            'log_id': log_entry.id,
            'msg_id': msg_id,
            'is_fax': is_fax,
            'is_retransmission': retransmit_seq is not None,
            'retransmitted_seq': retransmit_seq,
            'corrupted_seq': seq_numbers[corrupt_target_idx] if corrupt_target_idx is not None else None,
            'packet_count': len(packet_items),
            'packet_schedule': packet_schedule,
            'delivery_mode': delivery_mode,
            'audio_wav_base64': wav_base64,
            'duration_sec': round(len(clean_waveform) / modem.sample_rate, 2),
            'framing': {
                'raw_bits_len': total_raw_bits,
                'rle_compressed_len': total_rle_bits,
                'rle_saving_percent': rle_saving_percent,
                'crc_bits': first_crc,
                'total_frame_len': total_stream_bits,
                'tx_frame_bits': first_tx_bits[:160],
                'rx_frame_bits': rx_bits[:160],
                'flipped_indices': flipped_indices,
            },
            'fec_and_crc': {
                'valid_frame': True,
                'crc_passed': crc_passed,
                'physical_bit_errors': bit_mismatches,
                'ber': ber,
                'corrected_count': corrected_count,
                'recovered_text': log_label,
            },
            'channel_metrics': metrics,
            'charts': {
                'waveform_preview': waveform_preview,
            }
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@csrf_exempt
def decode_wav_api(request):
    """
    Demodulate a live microphone snapshot or uploaded WAV file and reconstruct
    CRC-verified UTF-8 Text or 16x16 Pixel Fax packets.
    """
    if request.method != 'POST' or 'audio_file' not in request.FILES:
        return JsonResponse({'error': 'WAV audio_file required.'}, status=400)

    try:
        audio_file = request.FILES['audio_file']
        baud_rate = int(request.POST.get('baud_rate', 65))
        levels_m = int(request.POST.get('levels_m', 4))
        ultrasonic = request.POST.get('ultrasonic', 'false').lower() == 'true'
        use_hamming = request.POST.get('use_hamming', 'true').lower() == 'true'
        use_rle = request.POST.get('use_rle', 'false').lower() == 'true'

        wav_bytes = audio_file.read()
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_pcm = wf.readframes(n_frames)
            waveform = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

        modem = AcousticModemPHY(
            sample_rate=sample_rate,
            baud_rate=baud_rate,
            ultrasonic=ultrasonic,
            levels_m=levels_m
        )
        stream_res = modem.demodulate_stream(waveform, use_hamming=use_hamming)
        valid_frames = stream_res['valid_frames']
        metrics = stream_res['metrics']

        if not valid_frames:
            return JsonResponse({
                'status': 'error',
                'error': 'Waiting for valid HDLC acoustic frames...',
                'metrics': metrics
            })

        packets = []
        total_repaired = 0
        last_matched_bits = ''
        last_crc = ''
        last_positions = []

        for f in valid_frames:
            total_repaired += f['corrected_errors_count']
            last_matched_bits = f.get('matched_frame_bits', '')[:160]
            last_crc = f['extracted_crc']
            last_positions = f.get('corrected_positions', [])

            header_info = ChannelCoder.unpack_packet_header(f['payload_bits'])
            if not header_info.get('valid'):
                continue

            c_bits = header_info['content_bits']
            if header_info['is_fax']:
                valid_rle_len = (len(c_bits) // 8) * 8
                rle_chunk = c_bits[:valid_rle_len]
                if not rle_chunk:
                    continue
                packets.append({
                    'msg_id': header_info['msg_id'],
                    'seq_num': header_info['seq_num'],
                    'is_last': header_info['is_last'],
                    'stream_mode': header_info['stream_mode'],
                    'is_fax': True,
                    'crc_corrupted': False,
                    'word': '',
                    'rle_chunk': rle_chunk
                })
            else:
                raw_word_bits = SourceCoder.rle_decode(c_bits) if use_rle else c_bits
                word_text = SourceCoder.bits_to_text(raw_word_bits)
                if not SourceCoder.is_clean_text(word_text):
                    continue
                packets.append({
                    'msg_id': header_info['msg_id'],
                    'seq_num': header_info['seq_num'],
                    'is_last': header_info['is_last'],
                    'stream_mode': header_info['stream_mode'],
                    'is_fax': False,
                    'crc_corrupted': False,
                    'word': word_text,
                    'rle_chunk': ''
                })

        if not packets:
            return JsonResponse({'status': 'error', 'error': 'No valid packet headers found.'})

        latest_msg_id = packets[-1]['msg_id']
        session_packets = [p for p in packets if p['msg_id'] == latest_msg_id]
        is_fax_session = session_packets[-1]['is_fax']
        stream_mode = session_packets[-1]['stream_mode']
        is_complete = any(p['is_last'] for p in session_packets)

        if is_fax_session:
            ordered_rle = {p['seq_num']: p['rle_chunk'] for p in session_packets}
            combined_rle = ''.join(ordered_rle[k] for k in sorted(ordered_rle.keys()))
            decoded_fax_bits = SourceCoder.rle_decode(combined_rle)
            assembled_text = f"[16x16 Fax Image: {len(decoded_fax_bits)}/256 pixels decoded]"
        else:
            ordered_words = {p['seq_num']: p['word'] for p in session_packets}
            assembled_text = ' '.join(ordered_words[k] for k in sorted(ordered_words.keys()))
            decoded_fax_bits = ''

        last_burst = stream_res['last_burst_wave']
        if len(last_burst) < modem.samples_per_symbol * 2:
            last_burst = modem.modulate(last_matched_bits[:16] or '10101010')
        waveform_preview = _extract_fsk_transition_preview(last_burst, modem.samples_per_symbol)

        return JsonResponse({
            'status': 'ok',
            'msg_id': latest_msg_id,
            'is_fax': is_fax_session,
            'fax_bits': decoded_fax_bits[:256],
            'stream_mode': stream_mode,
            'is_complete': is_complete,
            'arq_feedback': 'ACK',
            'feedback_freq_hz': AcousticModemPHY.ACK_FREQ_HZ,
            'nack_seqs': [],
            'packets': session_packets,
            'recovered_text': assembled_text,
            'crc_passed': True,
            'corrected_errors_count': total_repaired,
            'metrics': metrics,
            'framing': {
                'raw_bits_len': 256 if is_fax_session else len(assembled_text.encode('utf-8')) * 8,
                'crc_bits': last_crc,
                'total_frame_len': len(last_matched_bits),
                'tx_frame_bits': last_matched_bits,
                'rx_frame_bits': last_matched_bits,
                'flipped_indices': [p['global_bit_index'] + 16 for p in last_positions]
            },
            'fec_and_crc': {
                'crc_passed': True,
                'ber': round(total_repaired / max(len(last_matched_bits), 1), 5),
                'corrected_count': total_repaired,
                'recovered_text': assembled_text,
            },
            'channel_metrics': metrics,
            'charts': {
                'waveform_preview': waveform_preview
            }
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)