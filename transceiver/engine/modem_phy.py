# transceiver/engine/modem_phy.py
import io
import math
import wave
import numpy as np
from .channel_coding import ChannelCoder


class AcousticModemPHY:
    """
    Physical Layer (PHY) Acoustic Modem.
    Implements Binary Frequency Shift Keying (BFSK) modulation/demodulation,
    16-phase symbol timing synchronization across multi-packet streams,
    and Nyquist/Shannon channel capacity calculations.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        baud_rate: int = 45,
        ultrasonic: bool = False
    ):
        self.sample_rate = sample_rate
        self.baud_rate = baud_rate
        self.ultrasonic = ultrasonic

        if ultrasonic:
            self.f0 = 17500.0
            self.f1 = 18500.0
            self.bandwidth = 2000.0
        else:
            self.f0 = 1500.0
            self.f1 = 2500.0
            self.bandwidth = 2000.0

        self.samples_per_bit = int(self.sample_rate / self.baud_rate)

        t = np.arange(self.samples_per_bit, dtype=np.float32) / self.sample_rate
        self._cos_f0 = np.cos(2.0 * np.pi * self.f0 * t)
        self._sin_f0 = np.sin(2.0 * np.pi * self.f0 * t)
        self._cos_f1 = np.cos(2.0 * np.pi * self.f1 * t)
        self._sin_f1 = np.sin(2.0 * np.pi * self.f1 * t)

    def modulate(self, bits: str) -> np.ndarray:
        """Modulate a binary bit string into a continuous-phase FSK audio waveform."""
        if not bits:
            return np.array([], dtype=np.float32)

        total_samples = len(bits) * self.samples_per_bit
        waveform = np.zeros(total_samples, dtype=np.float32)

        phase = 0.0
        t_step = 1.0 / self.sample_rate

        edge_len = max(1, int(self.samples_per_bit * 0.08))
        window = np.ones(self.samples_per_bit, dtype=np.float32)
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, edge_len)))
        window[:edge_len] = ramp
        window[-edge_len:] = ramp[::-1]

        for idx, bit in enumerate(bits):
            freq = self.f1 if bit == '1' else self.f0
            phase_increments = 2.0 * np.pi * freq * np.arange(self.samples_per_bit) * t_step
            symbol_wave = np.sin(phase + phase_increments) * window

            start = idx * self.samples_per_bit
            waveform[start:start + self.samples_per_bit] = symbol_wave
            phase = (phase + 2.0 * np.pi * freq * self.samples_per_bit * t_step) % (2.0 * np.pi)

        return waveform * 0.85

    def modulate_packet_stream(self, frame_bits_list: list[str], guard_bits: int = 4) -> np.ndarray:
        """
        Modulate a sequence of word frames separated by alternating sync bits ('1010')
        so phase synchronization remains locked across the entire multi-word stream.
        """
        guard_pattern = "10" * (guard_bits // 2)
        combined_bits = guard_pattern.join(frame_bits_list)
        return self.modulate(combined_bits)

    def waveform_to_wav_bytes(self, waveform: np.ndarray) -> bytes:
        """Convert normalized float32 waveform into 16-bit mono PCM WAV bytes."""
        pcm_int16 = np.int16(np.clip(waveform, -1.0, 1.0) * 32767)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_int16.tobytes())
        return buffer.getvalue()

    def _isolate_carrier_burst(self, waveform: np.ndarray) -> np.ndarray:
        """
        Locate the active FSK carrier region without clipping the leading Preamble
        or trailing HDLC flags.
        """
        n_sym = len(waveform) // self.samples_per_bit
        if n_sym < 8:
            return waveform

        trimmed = waveform[:n_sym * self.samples_per_bit]
        matrix = trimmed.reshape(n_sym, self.samples_per_bit)
        norm = float(self.samples_per_bit ** 2)

        p0 = ((matrix @ self._cos_f0) ** 2 + (matrix @ self._sin_f0) ** 2) / norm
        p1 = ((matrix @ self._cos_f1) ** 2 + (matrix @ self._sin_f1) ** 2) / norm
        carrier_power = p0 + p1

        peak_power = float(np.max(carrier_power))
        if peak_power < 1e-7:
            return np.array([], dtype=np.float32)

        active_idx = np.where(carrier_power > peak_power * 0.05)[0]
        if len(active_idx) < 8:
            return np.array([], dtype=np.float32)

        # Keep a generous 4-symbol margin on both sides so flags are never cut off
        start_sym = max(0, int(active_idx[0]) - 4)
        end_sym = min(n_sym, int(active_idx[-1]) + 5)
        return waveform[start_sym * self.samples_per_bit:end_sym * self.samples_per_bit]

    def demodulate(self, received_waveform: np.ndarray) -> dict:
        """Fast vectorized non-coherent FSK demodulator."""
        n_sym = len(received_waveform) // self.samples_per_bit
        if n_sym == 0:
            return {
                "bits": "",
                "f0_energies": [],
                "f1_energies": [],
                "metrics": self.compute_channel_metrics(self.bandwidth, 1.0, 2)
            }

        trimmed = received_waveform[:n_sym * self.samples_per_bit]
        matrix = trimmed.reshape(n_sym, self.samples_per_bit)
        norm = float(self.samples_per_bit ** 2)

        p0 = ((matrix @ self._cos_f0) ** 2 + (matrix @ self._sin_f0) ** 2) / norm
        p1 = ((matrix @ self._cos_f1) ** 2 + (matrix @ self._sin_f1) ** 2) / norm

        total_p = p0 + p1
        peak_p = float(np.max(total_p)) if len(total_p) > 0 else 0.0
        squelch_threshold = peak_p * 0.03

        detected_bits = []
        signal_powers = []
        noise_powers = []

        for i in range(n_sym):
            if total_p[i] < squelch_threshold:
                detected_bits.append('0')
                continue
            if p1[i] > p0[i]:
                detected_bits.append('1')
                signal_powers.append(float(p1[i]))
                noise_powers.append(float(max(p0[i], 1e-12)))
            else:
                detected_bits.append('0')
                signal_powers.append(float(p0[i]))
                noise_powers.append(float(max(p1[i], 1e-12)))

        avg_sig = float(np.mean(signal_powers)) if signal_powers else 1.0
        avg_noise = float(np.mean(noise_powers)) if noise_powers else 1e-6
        snr_linear = avg_sig / max(avg_noise, 1e-12)

        return {
            "bits": ''.join(detected_bits),
            "f0_energies": [round(float(x), 6) for x in p0[:60]],
            "f1_energies": [round(float(x), 6) for x in p1[:60]],
            "metrics": self.compute_channel_metrics(self.bandwidth, snr_linear, 2)
        }

    def demodulate_stream(self, received_waveform: np.ndarray, use_hamming: bool = True) -> dict:
        """
        Scan all 16 sub-symbol timing phases across the active waveform and
        aggregate all decoded word frames across phases.
        """
        if len(received_waveform) > 0:
            received_waveform = received_waveform - np.mean(received_waveform)

        active_wave = self._isolate_carrier_burst(received_waveform)
        if len(active_wave) < self.samples_per_bit * 16:
            empty_demod = self.demodulate(np.array([], dtype=np.float32))
            return {
                "valid_frames": [],
                "metrics": empty_demod["metrics"],
                "last_burst_wave": np.array([], dtype=np.float32)
            }

        collected_by_seq = {}
        fallback_frames = []
        best_metrics = None

        num_phases = 16
        step = max(1, self.samples_per_bit // num_phases)

        for phase_idx in range(num_phases):
            offset = phase_idx * step
            if offset >= len(active_wave) - self.samples_per_bit:
                break

            shifted_wave = active_wave[offset:]
            demod_candidate = self.demodulate(shifted_wave)
            if best_metrics is None:
                best_metrics = demod_candidate["metrics"]

            frames = ChannelCoder.parse_all_frames(demod_candidate["bits"], use_hamming=use_hamming)
            for f in frames:
                if f.get("crc_passed"):
                    hdr = ChannelCoder.unpack_packet_header(f["payload_bits"])
                    if hdr.get("valid"):
                        key = (hdr["msg_id"], hdr["seq_num"])
                        if key not in collected_by_seq:
                            collected_by_seq[key] = f
                            best_metrics = demod_candidate["metrics"]
                else:
                    fallback_frames.append(f)

        final_frames = list(collected_by_seq.values())
        if not final_frames and fallback_frames:
            final_frames = fallback_frames[:1]

        return {
            "valid_frames": final_frames,
            "metrics": best_metrics,
            "last_burst_wave": active_wave
        }

    def compute_channel_metrics(
        self,
        bandwidth_hz: float,
        snr_linear: float,
        levels_m: int = 2
    ) -> dict:
        """Compute SNR (dB), Nyquist bit rate, and Shannon capacity."""
        snr_linear = max(snr_linear, 1e-9)
        snr_db = 10.0 * math.log10(snr_linear)
        nyquist_rate = 2.0 * bandwidth_hz * math.log2(levels_m)
        shannon_capacity = bandwidth_hz * math.log2(1.0 + snr_linear)
        spectral_efficiency = self.baud_rate / bandwidth_hz

        return {
            "bandwidth_hz": round(bandwidth_hz, 2),
            "snr_linear": round(snr_linear, 2),
            "snr_db": round(snr_db, 2),
            "nyquist_max_bps": round(nyquist_rate, 2),
            "shannon_capacity_bps": round(shannon_capacity, 2),
            "actual_bit_rate_bps": self.baud_rate,
            "spectral_efficiency_bps_hz": round(spectral_efficiency, 4)
        }