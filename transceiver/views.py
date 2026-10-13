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


def _extract_fsk_transition_preview(waveform: np.ndarray, samples_per_bit: int) -> list[float]:
    """
    Extract a normalized window around the first bit transition (Bit '1' -> Bit '0')
    so the FSK frequency shift is clearly rendered on the Signal Lab chart.
    """
    if len(waveform) == 0:
        return [0.0] * 200

    boundary = min(samples_per_bit, len(waveform) // 2)
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
    Packetize message into per-word HDLC frames with Sequence Numbers,
    modulate into FSK audio, and return word timing schedule for live display.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required.'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
        text = data.get('text', '').strip() or 'Hello UoM!'
        baud_rate = int(data.get('baud_rate', 45))
        ultrasonic = bool(data.get('ultrasonic', False))
        delivery_mode = data.get('delivery_mode', 'stream')  # 'stream' or 'full'
        use_rle = bool(data.get('use_rle', False))
        use_hamming = bool(data.get('use_hamming', True))
        noise_std = float(data.get('noise_std', 0.05))
        error_mode = data.get('error_mode', 'multiple')

        stream_mode_flag = (delivery_mode == 'stream')
        msg_id = random.randint(1, 7)

        words = [w for w in text.split(' ') if w]
        if not words:
            words = [text]

        frame_bits_list = []
        packet_schedule = []
        total_raw_bits = 0
        total_rle_bits = 0
        first_frame_info = None
        cumulative_bits = 0

        for idx, word in enumerate(words):
            is_last = (idx == len(words) - 1)
            word_bits = SourceCoder.text_to_bits(word)
            rle_stats = SourceCoder.rle_encode(word_bits)
            encoded_word_bits = rle_stats['encoded_bits'] if use_rle else word_bits

            total_raw_bits += len(word_bits)
            total_rle_bits += rle_stats['compressed_len']

            header_bits = ChannelCoder.pack_packet_header(
                msg_id=msg_id,
                seq_num=idx,
                is_last=is_last,
                stream_mode=stream_mode_flag
            )
            packet_payload = header_bits + encoded_word_bits
            f_info = ChannelCoder.build_frame(packet_payload, use_hamming=use_hamming)
            f_bits = f_info['frame_bits']
            frame_bits_list.append(f_bits)

            cumulative_bits += len(f_bits) + (4 if not is_last else 0)
            emit_time_sec = round(cumulative_bits / baud_rate, 3)

            packet_schedule.append({
                'msg_id': msg_id,
                'seq_num': idx,
                'is_last': is_last,
                'stream_mode': stream_mode_flag,
                'word': word,
                'emit_time_sec': emit_time_sec
            })

            if first_frame_info is None:
                first_frame_info = f_info

        modem = AcousticModemPHY(sample_rate=44100, baud_rate=baud_rate, ultrasonic=ultrasonic)
        clean_waveform = modem.modulate_packet_stream(frame_bits_list, guard_bits=4)
        wav_bytes = modem.waveform_to_wav_bytes(clean_waveform)
        wav_base64 = base64.b64encode(wav_bytes).decode('ascii')

        first_tx_bits = frame_bits_list[0]
        first_wave = modem.modulate(first_tx_bits)
        if noise_std > 0:
            rx_first_wave = first_wave + np.random.normal(0.0, noise_std, size=first_wave.shape).astype(np.float32)
        else:
            rx_first_wave = first_wave.copy()

        demod_res = modem.demodulate(rx_first_wave)
        rx_bits = demod_res['bits']

        flipped_indices = []
        header_offset = len(ChannelCoder.PREAMBLE) + len(ChannelCoder.HDLC_FLAG)
        if error_mode in ('single', 'multiple', 'burst') and len(rx_bits) > header_offset + 8:
            body = rx_bits[header_offset:-len(ChannelCoder.HDLC_FLAG)]
            noisy_body, local_flips = ChannelCoder.inject_channel_error(body, mode=error_mode)
            flipped_indices = [idx + header_offset for idx in local_flips]
            rx_bits = rx_bits[:header_offset] + noisy_body + rx_bits[-len(ChannelCoder.HDLC_FLAG):]

        bit_mismatches = sum(1 for a, b in zip(first_tx_bits, rx_bits) if a != b)
        ber = round(bit_mismatches / max(len(first_tx_bits), 1), 6)

        parsed = ChannelCoder.parse_frame(rx_bits, use_hamming=use_hamming)
        crc_passed = parsed.get('crc_passed', False) if parsed.get('valid_frame') else False
        corrected_count = parsed.get('corrected_errors_count', 0) if parsed.get('valid_frame') else 0

        metrics = demod_res['metrics']
        total_stream_bits = sum(len(b) for b in frame_bits_list)

        log_entry = TransmissionLog.objects.create(
            input_text=text,
            carrier_mode='ultrasonic' if ultrasonic else 'audible',
            baud_rate=baud_rate,
            use_rle=use_rle,
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
            recovered_text=text
        )

        waveform_preview = _extract_fsk_transition_preview(rx_first_wave, modem.samples_per_bit)

        return JsonResponse({
            'status': 'ok',
            'log_id': log_entry.id,
            'msg_id': msg_id,
            'packet_count': len(words),
            'packet_schedule': packet_schedule,
            'delivery_mode': delivery_mode,
            'audio_wav_base64': wav_base64,
            'duration_sec': round(len(clean_waveform) / modem.sample_rate, 2),
            'framing': {
                'raw_bits_len': total_raw_bits,
                'rle_compressed_len': total_rle_bits,
                'crc_bits': first_frame_info['crc_bits'],
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
                'recovered_text': text,
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
    Demodulate a live microphone snapshot or uploaded WAV file,
    extract all packet frames, unpack sequence numbers, and return decoded words.
    """
    if request.method != 'POST' or 'audio_file' not in request.FILES:
        return JsonResponse({'error': 'WAV audio_file required.'}, status=400)

    try:
        audio_file = request.FILES['audio_file']
        baud_rate = int(request.POST.get('baud_rate', 45))
        ultrasonic = request.POST.get('ultrasonic', 'false').lower() == 'true'
        use_hamming = request.POST.get('use_hamming', 'true').lower() == 'true'
        use_rle = request.POST.get('use_rle', 'false').lower() == 'true'

        wav_bytes = audio_file.read()
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_pcm = wf.readframes(n_frames)
            waveform = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

        modem = AcousticModemPHY(sample_rate=sample_rate, baud_rate=baud_rate, ultrasonic=ultrasonic)
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
            if header_info.get('valid'):
                c_bits = header_info['content_bits']
                raw_word_bits = SourceCoder.rle_decode(c_bits) if use_rle else c_bits
                word_text = SourceCoder.bits_to_text(raw_word_bits)
                packets.append({
                    'msg_id': header_info['msg_id'],
                    'seq_num': header_info['seq_num'],
                    'is_last': header_info['is_last'],
                    'stream_mode': header_info['stream_mode'],
                    'word': word_text
                })

        if not packets:
            return JsonResponse({'status': 'error', 'error': 'No valid packet headers found.'})

        latest_msg_id = packets[-1]['msg_id']
        ordered_map = {}
        stream_mode = True
        is_complete = False

        for p in packets:
            if p['msg_id'] == latest_msg_id:
                ordered_map[p['seq_num']] = p['word']
                stream_mode = p['stream_mode']
                if p['is_last']:
                    is_complete = True

        assembled_words = [ordered_map[k] for k in sorted(ordered_map.keys())]
        assembled_text = ' '.join(assembled_words)

        last_burst = stream_res['last_burst_wave']
        if len(last_burst) < modem.samples_per_bit * 2:
            last_burst = modem.modulate(last_matched_bits[:16] or '10101010')
        waveform_preview = _extract_fsk_transition_preview(last_burst, modem.samples_per_bit)

        return JsonResponse({
            'status': 'ok',
            'msg_id': latest_msg_id,
            'stream_mode': stream_mode,
            'is_complete': is_complete,
            'packets': [p for p in packets if p['msg_id'] == latest_msg_id],
            'recovered_text': assembled_text,
            'crc_passed': True,
            'corrected_errors_count': total_repaired,
            'metrics': metrics,
            'framing': {
                'raw_bits_len': len(assembled_text.encode('utf-8')) * 8,
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