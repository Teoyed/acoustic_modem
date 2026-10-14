# transceiver/engine/modem_phy.py
import io
import math
import wave
import numpy as np
from .channel_coding import ChannelCoder


class AcousticModemPHY:
    """
    Noise-Immune Physical Layer (PHY) Acoustic Modem.
    Uses a High-Clarity Audible Band (4.2 kHz - 6.0 kHz) above human speech,
    FFT Band-Pass Filtering, and a Spectral Tone-Dominance Gate to reject room noise.
    """

    SYMBOL_BITS_4FSK = ("00", "01", "10", "11")

    def __init__(
        self,
        sample_rate: int = 44100,
        baud_rate: int = 65,
        ultrasonic: bool = False,
        levels_m: int = 4
    ):
        self.sample_rate = sample_rate
        self.baud_rate = baud_rate
        self.ultrasonic = ultrasonic
        self.levels_m = 4 if levels_m == 4 else 2
        self.bits_per_symbol = int(math.log2(self.levels_m))
        self.bit_rate_bps = self.baud_rate * self.bits_per_symbol

        # High-Clarity Anti-Noise Carrier Frequencies (Far above human voice < 3.2 kHz)
        if self.levels_m == 4:
            if ultrasonic:
                self.frequencies = [16800.0, 17400.0, 18000.0, 18600.0]
                self.bandwidth = 2400.0
            else:
                self.frequencies = [4200.0, 4800.0, 5400.0, 6000.0]
                self.bandwidth = 2400.0
        else:
            if ultrasonic:
                self.frequencies = [17500.0, 18500.0]
                self.bandwidth = 2000.0
            else:
                self.frequencies = [4500.0, 5700.0]
                self.bandwidth = 2000.0

        self.f0 = self.frequencies[0]
        self.f1 = self.frequencies[-1]
        self.samples_per_symbol = int(self.sample_rate / self.baud_rate)
        self.samples_per_bit = self.samples_per_symbol

        t = np.arange(self.samples_per_symbol, dtype=np.float32) / self.sample_rate
        self._cos_bank = [np.cos(2.0 * np.pi * f * t) for f in self.frequencies]
        self._sin_bank = [np.sin(2.0 * np.pi * f * t) for f in self.frequencies]

    def _apply_bandpass_filter(self, waveform: np.ndarray) -> np.ndarray:
        """
        Sharp FFT Band-Pass Filter:
        Completely zeroes out human speech (< 3.8 kHz), low-frequency room rumble,
        and out-of-band interference before demodulation.
        """
        n = len(waveform)
        if n < self.samples_per_symbol:
            return waveform

        low_cut = min(self.frequencies) - 350.0
        high_cut = max(self.frequencies) + 350.0

        spectrum = np.fft.rfft(waveform)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        mask = (freqs >= low_cut) & (freqs <= high_cut)
        filtered = np.fft.irfft(spectrum * mask, n=n)
        return filtered.astype(np.float32)

    def modulate(self, bits: str) -> np.ndarray:
        """
        Modulate a binary bit string into a continuous-phase M-FSK waveform
        using a raised-cosine envelope for clean spectral containment.
        """
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

        return waveform * 0.85

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
        Isolate the active M-FSK signal burst only if narrowband carrier tones
        dominate over background broadband noise.
        """
        n_sym = len(waveform) // self.samples_per_symbol
        if n_sym < 6:
            return np.array([], dtype=np.float32)

        trimmed = waveform[:n_sym * self.samples_per_symbol]
        matrix = trimmed.reshape(n_sym, self.samples_per_symbol)
        norm = float(self.samples_per_symbol ** 2)

        powers = np.zeros((self.levels_m, n_sym), dtype=np.float32)
        for m_idx in range(self.levels_m):
            c_vec = self._cos_bank[m_idx]
            s_vec = self._sin_bank[m_idx]
            powers[m_idx] = ((matrix @ c_vec) ** 2 + (matrix @ s_vec) ** 2) / norm

        max_tone_power = np.max(powers, axis=0)
        total_power = np.sum(powers, axis=0)
        other_power = (total_power - max_tone_power) / max(1, self.levels_m - 1)

        # Tone Dominance Ratio: True FSK tones concentrate energy in a single carrier,
        # whereas claps/speech spread energy across multiple frequencies.
        dominance_ratio = max_tone_power / np.maximum(other_power, 1e-10)

        peak_power = float(np.max(max_tone_power))
        if peak_power < 1e-6:
            return np.array([], dtype=np.float32)

        # Require both sufficient amplitude AND single-tone spectral dominance (> 2.2x)
        valid_tone_mask = (max_tone_power > peak_power * 0.08) & (dominance_ratio > 2.2)
        active_idx = np.where(valid_tone_mask)[0]

        if len(active_idx) < 8:
            return np.array([], dtype=np.float32)

        start_sym = max(0, int(active_idx[0]) - 4)
        end_sym = min(n_sym, int(active_idx[-1]) + 5)
        return waveform[start_sym * self.samples_per_symbol:end_sym * self.samples_per_symbol]

    def demodulate(self, received_waveform: np.ndarray) -> dict:
        """
        Fast vectorized non-coherent M-FSK demodulator with Tone-Dominance gating.
        """
        n_sym = len(received_waveform) // self.samples_per_symbol
        if n_sym == 0:
            return {
                "bits": "",
                "f0_energies": [],
                "f1_energies": [],
                "metrics": self.compute_channel_metrics(self.bandwidth, 1.0, self.levels_m)
            }

        trimmed = received_waveform[:n_sym * self.samples_per_symbol]
        matrix = trimmed.reshape(n_sym, self.samples_per_symbol)
        norm = float(self.samples_per_symbol ** 2)

        powers = np.zeros((self.levels_m, n_sym), dtype=np.float32)
        for m_idx in range(self.levels_m):
            c_vec = self._cos_bank[m_idx]
            s_vec = self._sin_bank[m_idx]
            powers[m_idx] = ((matrix @ c_vec) ** 2 + (matrix @ s_vec) ** 2) / norm

        total_p = np.sum(powers, axis=0)
        peak_p = float(np.max(total_p)) if len(total_p) > 0 else 0.0
        squelch_threshold = peak_p * 0.05

        best_tone_indices = np.argmax(powers, axis=0)
        detected_chunks = []
        signal_powers = []
        noise_powers = []

        for i in range(n_sym):
            if total_p[i] < squelch_threshold:
                detected_chunks.append("00" if self.levels_m == 4 else "0")
                continue

            winner = int(best_tone_indices[i])
            win_power = float(powers[winner, i])
            other_power = float(total_p[i] - win_power) / max(1, self.levels_m - 1)

            # Reject broadband impulse noise (e.g. clapping/coughing) that excites all tones equally
            if win_power < other_power * 1.8:
                detected_chunks.append("00" if self.levels_m == 4 else "0")
                continue

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
        Band-pass filter the microphone recording, isolate genuine FSK bursts,
        and strictly extract only CRC-8 verified frames with high SNR.
        """
        if len(received_waveform) > 0:
            received_waveform = received_waveform - np.mean(received_waveform)
            received_waveform = self._apply_bandpass_filter(received_waveform)

        active_wave = self._isolate_carrier_burst(received_waveform)
        if len(active_wave) < self.samples_per_symbol * 10:
            empty_demod = self.demodulate(np.array([], dtype=np.float32))
            return {
                "valid_frames": [],
                "metrics": empty_demod["metrics"],
                "last_burst_wave": np.array([], dtype=np.float32)
            }

        collected_by_seq = {}
        best_metrics = None

        num_phases = 16
        step = max(1, self.samples_per_symbol // num_phases)

        for phase_idx in range(num_phases):
            offset = phase_idx * step
            if offset >= len(active_wave) - self.samples_per_symbol:
                break

            shifted_wave = active_wave[offset:]
            demod_candidate = self.demodulate(shifted_wave)
            if best_metrics is None:
                best_metrics = demod_candidate["metrics"]

            # Noise Gate: Require at least 4.5 dB in-band SNR to consider frames valid
            if demod_candidate["metrics"]["snr_db"] < 4.5:
                continue

            bit_candidates = [demod_candidate["bits"]]
            if self.levels_m == 4 and len(demod_candidate["bits"]) > 2:
                bit_candidates.append(demod_candidate["bits"][1:])

            for cand_bits in bit_candidates:
                # Strict CRC-8 enforcement: never accept unverified fallback frames from mic noise!
                frames = ChannelCoder.parse_all_frames(
                    cand_bits,
                    use_hamming=use_hamming,
                    strict_crc=True
                )
                for f in frames:
                    hdr = ChannelCoder.unpack_packet_header(f["payload_bits"])
                    if hdr.get("valid"):
                        key = (hdr["msg_id"], hdr["seq_num"])
                        if key not in collected_by_seq:
                            collected_by_seq[key] = f
                            best_metrics = demod_candidate["metrics"]

        return {
            "valid_frames": list(collected_by_seq.values()),
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