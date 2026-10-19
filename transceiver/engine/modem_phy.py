# transceiver/engine/modem_phy.py
import io
import math
import wave
import numpy as np
from .channel_coding import ChannelCoder


class AcousticModemPHY:
    """
    Cross-Device Physical Layer (PHY) Acoustic Modem.
    Features ISI Guard-Interval Windowing, Per-Carrier Spectral Equalization
    for real laptop/phone microphones, and hardware-friendly Near-Ultrasonic frequencies.
    """

    SYMBOL_BITS_4FSK = ("00", "01", "10", "11")
    ACK_FREQ_HZ = 6800.0
    NACK_FREQ_HZ = 3400.0

    def __init__(
        self,
        sample_rate: int = 44100,
        baud_rate: int = 45,
        ultrasonic: bool = False,
        levels_m: int = 4
    ):
        self.sample_rate = sample_rate
        self.baud_rate = baud_rate
        self.ultrasonic = ultrasonic
        self.levels_m = 4 if levels_m == 4 else 2
        self.bits_per_symbol = int(math.log2(self.levels_m))
        self.bit_rate_bps = self.baud_rate * self.bits_per_symbol

        # Cross-device acoustic frequencies tuned for phone/laptop speakers & mics
        if self.levels_m == 4:
            if ultrasonic:
                # 14.4 - 16.2 kHz: Near-ultrasonic band that passes through mobile/laptop mics without roll-off
                self.frequencies = [14400.0, 15000.0, 15600.0, 16200.0]
                self.bandwidth = 2400.0
            else:
                # 2.4 - 4.5 kHz: Sweet-spot audible band with flat mic response and low speech interference
                self.frequencies = [2400.0, 3100.0, 3800.0, 4500.0]
                self.bandwidth = 2600.0
        else:
            if ultrasonic:
                self.frequencies = [14800.0, 15800.0]
                self.bandwidth = 2000.0
            else:
                self.frequencies = [2600.0, 3800.0]
                self.bandwidth = 2000.0

        self.f0 = self.frequencies[0]
        self.f1 = self.frequencies[-1]
        self.samples_per_symbol = int(self.sample_rate / self.baud_rate)
        self.samples_per_bit = self.samples_per_symbol

        # Core 72% symbol integration window: strips 14% leading/trailing edges to eliminate room echo ISI
        self._guard_trim = max(2, int(self.samples_per_symbol * 0.14))
        self._core_len = max(16, self.samples_per_symbol - 2 * self._guard_trim)

        t_core = np.arange(self._core_len, dtype=np.float32) / self.sample_rate
        hann_core = np.hanning(self._core_len).astype(np.float32)
        self._cos_bank = [np.cos(2.0 * np.pi * f * t_core) * hann_core for f in self.frequencies]
        self._sin_bank = [np.sin(2.0 * np.pi * f * t_core) * hann_core for f in self.frequencies]

    def _apply_bandpass_filter(self, waveform: np.ndarray) -> np.ndarray:
        """Sharp FFT Band-Pass Filter + Automatic Gain Normalization (AGC)."""
        n = len(waveform)
        if n < self.samples_per_symbol:
            return waveform

        low_cut = min(self.frequencies) - 400.0
        high_cut = max(self.frequencies) + 400.0

        spectrum = np.fft.rfft(waveform)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        mask = (freqs >= low_cut) & (freqs <= high_cut)
        filtered = np.fft.irfft(spectrum * mask, n=n).astype(np.float32)

        peak = float(np.max(np.abs(filtered)))
        if peak > 1e-5:
            filtered = filtered / peak
        return filtered

    def modulate(self, bits: str) -> np.ndarray:
        """Modulate a binary bit string into a continuous-phase M-FSK waveform."""
        if not bits:
            return np.array([], dtype=np.float32)

        if self.levels_m == 4:
            if len(bits) % 2 != 0:
                bits = bits + '0'
            symbols = [int(bits[i:i+2], 2) for i in range(0, len(bits), 2)]
        else:
            symbols = [1 if b == '1' else 0 for b in bits]

        total_samples = len(symbols) * self.samples_per_symbol
        waveform = np.zeros(total_samples, dtype=np.float32)

        phase = 0.0
        t_step = 1.0 / self.sample_rate

        edge_len = max(1, int(self.samples_per_symbol * 0.12))
        window = np.ones(self.samples_per_symbol, dtype=np.float32)
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, edge_len)))
        window[:edge_len] = ramp
        window[-edge_len:] = ramp[::-1]

        for idx, sym_val in enumerate(symbols):
            freq = self.frequencies[sym_val]
            phase_increments = 2.0 * np.pi * freq * np.arange(self.samples_per_symbol) * t_step
            symbol_wave = np.sin(phase + phase_increments) * window

            start = idx * self.samples_per_symbol
            waveform[start:start + self.samples_per_symbol] = symbol_wave
            phase = (phase + 2.0 * np.pi * freq * self.samples_per_symbol * t_step) % (2.0 * np.pi)

        return waveform * 0.90

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

    def _compute_symbol_powers(self, waveform: np.ndarray) -> np.ndarray:
        """
        Compute M-carrier energy matrix using only the ISI-free core of each symbol.
        """
        n_sym = len(waveform) // self.samples_per_symbol
        if n_sym == 0:
            return np.zeros((self.levels_m, 0), dtype=np.float32)

        trimmed = waveform[:n_sym * self.samples_per_symbol]
        full_matrix = trimmed.reshape(n_sym, self.samples_per_symbol)
        # Slice out the center of each symbol to reject room reverberation (ISI) at edges
        core_matrix = full_matrix[:, self._guard_trim:self._guard_trim + self._core_len]
        norm = float(self._core_len ** 2)

        powers = np.zeros((self.levels_m, n_sym), dtype=np.float32)
        for m_idx in range(self.levels_m):
            c_vec = self._cos_bank[m_idx]
            s_vec = self._sin_bank[m_idx]
            powers[m_idx] = ((core_matrix @ c_vec) ** 2 + (core_matrix @ s_vec) ** 2) / norm

        return powers

    def _isolate_carrier_burst(self, waveform: np.ndarray) -> np.ndarray:
        """Isolate the active M-FSK signal burst from microphone audio."""
        powers = self._compute_symbol_powers(waveform)
        n_sym = powers.shape[1]
        if n_sym < 6:
            return np.array([], dtype=np.float32)

        max_tone_power = np.max(powers, axis=0)
        total_power = np.sum(powers, axis=0)
        other_power = (total_power - max_tone_power) / max(1, self.levels_m - 1)
        dominance_ratio = max_tone_power / np.maximum(other_power, 1e-10)

        peak_power = float(np.max(max_tone_power))
        if peak_power < 1e-7:
            return np.array([], dtype=np.float32)

        valid_tone_mask = (max_tone_power > peak_power * 0.04) & (dominance_ratio > 1.45)
        active_idx = np.where(valid_tone_mask)[0]

        if len(active_idx) < 6:
            return np.array([], dtype=np.float32)

        start_sym = max(0, int(active_idx[0]) - 6)
        end_sym = min(n_sym, int(active_idx[-1]) + 6)
        return waveform[start_sym * self.samples_per_symbol:end_sym * self.samples_per_symbol]

    def demodulate(self, received_waveform: np.ndarray) -> dict:
        """
        Non-coherent M-FSK demodulator with Per-Carrier Spectral Equalization
        to compensate for microphone frequency roll-off on external devices.
        """
        powers = self._compute_symbol_powers(received_waveform)
        n_sym = powers.shape[1]
        if n_sym == 0:
            return {
                "bits": "",
                "f0_energies": [],
                "f1_energies": [],
                "metrics": self.compute_channel_metrics(self.bandwidth, 1.0, self.levels_m)
            }

        # Per-Carrier Spectral Equalization:
        # Compensates when a laptop/phone speaker or mic attenuates higher frequencies (e.g., f3 vs f0)
        eq_powers = powers.copy()
        if n_sym >= 8:
            for m_idx in range(self.levels_m):
                tone_ref = float(np.percentile(powers[m_idx], 85))
                if tone_ref > 1e-8:
                    eq_powers[m_idx] = powers[m_idx] / tone_ref

        total_raw_p = np.sum(powers, axis=0)
        peak_raw_p = float(np.max(total_raw_p)) if len(total_raw_p) > 0 else 0.0
        squelch_threshold = peak_raw_p * 0.025

        best_tone_indices = np.argmax(eq_powers, axis=0)
        detected_chunks = []
        signal_powers = []
        noise_powers = []

        for i in range(n_sym):
            if total_raw_p[i] < squelch_threshold:
                detected_chunks.append("00" if self.levels_m == 4 else "0")
                continue

            winner = int(best_tone_indices[i])
            win_power = float(powers[winner, i])
            other_power = float(total_raw_p[i] - win_power) / max(1, self.levels_m - 1)

            signal_powers.append(win_power)
            noise_powers.append(max(other_power, 1e-12))

            if self.levels_m == 4:
                detected_chunks.append(self.SYMBOL_BITS_4FSK[winner])
            else:
                detected_chunks.append('1' if winner == 1 else '0')

        avg_sig = float(np.mean(signal_powers)) if signal_powers else 0.0
        avg_noise = float(np.mean(noise_powers)) if noise_powers else 1e-6
        snr_linear = avg_sig / max(avg_noise, 1e-12) if avg_sig > 0 else 0.1

        return {
            "bits": ''.join(detected_chunks),
            "f0_energies": [round(float(x), 6) for x in powers[0, :60]],
            "f1_energies": [round(float(x), 6) for x in powers[-1, :60]],
            "metrics": self.compute_channel_metrics(self.bandwidth, snr_linear, self.levels_m)
        }

    def demodulate_stream(self, received_waveform: np.ndarray, use_hamming: bool = True) -> dict:
        """
        Band-pass filter the recording and scan 20 sub-symbol timing phases
        to lock onto over-the-air frames from external devices.
        """
        if len(received_waveform) > 0:
            received_waveform = received_waveform - np.mean(received_waveform)
            received_waveform = self._apply_bandpass_filter(received_waveform)

        active_wave = self._isolate_carrier_burst(received_waveform)
        if len(active_wave) < self.samples_per_symbol * 8:
            empty_demod = self.demodulate(np.array([], dtype=np.float32))
            return {
                "valid_frames": [],
                "corrupted_frames": [],
                "metrics": empty_demod["metrics"],
                "last_burst_wave": np.array([], dtype=np.float32)
            }

        collected_by_seq = {}
        best_metrics = None

        num_phases = 20
        step = max(1, self.samples_per_symbol // num_phases)

        for phase_idx in range(num_phases):
            offset = phase_idx * step
            if offset >= len(active_wave) - self.samples_per_symbol:
                break

            shifted_wave = active_wave[offset:]
            demod_candidate = self.demodulate(shifted_wave)
            if best_metrics is None:
                best_metrics = demod_candidate["metrics"]

            if demod_candidate["metrics"]["snr_db"] < 2.0:
                continue

            bit_candidates = [demod_candidate["bits"]]
            if self.levels_m == 4 and len(demod_candidate["bits"]) > 2:
                bit_candidates.append(demod_candidate["bits"][1:])

            for cand_bits in bit_candidates:
                frames = ChannelCoder.parse_all_frames(
                    cand_bits,
                    use_hamming=use_hamming,
                    strict_crc=True
                )
                for f in frames:
                    hdr = ChannelCoder.unpack_packet_header(f["payload_bits"])
                    if not hdr.get("valid"):
                        continue
                    key = (hdr["msg_id"], hdr["seq_num"])
                    if key not in collected_by_seq:
                        collected_by_seq[key] = f
                        best_metrics = demod_candidate["metrics"]

        return {
            "valid_frames": list(collected_by_seq.values()),
            "corrupted_frames": [],
            "metrics": best_metrics,
            "last_burst_wave": active_wave
        }

    def compute_channel_metrics(
        self,
        bandwidth_hz: float,
        snr_linear: float,
        levels_m: int = 4
    ) -> dict:
        """Compute SNR (dB), Nyquist bit rate, and Shannon capacity."""
        snr_linear = max(snr_linear, 1e-9)
        snr_db = 10.0 * math.log10(snr_linear)
        nyquist_rate = 2.0 * bandwidth_hz * math.log2(levels_m)
        shannon_capacity = bandwidth_hz * math.log2(1.0 + snr_linear)
        effective_bit_rate = int(self.baud_rate * math.log2(levels_m))
        spectral_efficiency = effective_bit_rate / bandwidth_hz

        return {
            "levels_m": levels_m,
            "bandwidth_hz": round(bandwidth_hz, 2),
            "snr_linear": round(snr_linear, 2),
            "snr_db": round(snr_db, 2),
            "nyquist_max_bps": round(nyquist_rate, 2),
            "shannon_capacity_bps": round(shannon_capacity, 2),
            "symbol_rate_baud": self.baud_rate,
            "actual_bit_rate_bps": effective_bit_rate,
            "spectral_efficiency_bps_hz": round(spectral_efficiency, 4)
        }