# transceiver/engine/modem_phy.py
import io
import math
import wave
import numpy as np
from .channel_coding import ChannelCoder


class AcousticModemPHY:
    """
    Physical Layer (PHY) Acoustic Modem.
    Supports both Binary FSK (2-FSK, M=2, 1 bit/symbol) and
    Quadrature/4-Level FSK (4-FSK, M=4, 2 bits/symbol) for high-speed transmission.
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

        if self.levels_m == 4:
            if ultrasonic:
                self.frequencies = [16800.0, 17400.0, 18000.0, 18600.0]
                self.bandwidth = 2400.0
            else:
                self.frequencies = [1200.0, 1800.0, 2400.0, 3000.0]
                self.bandwidth = 2400.0
        else:
            if ultrasonic:
                self.frequencies = [17500.0, 18500.0]
                self.bandwidth = 2000.0
            else:
                self.frequencies = [1500.0, 2500.0]
                self.bandwidth = 2000.0

        self.f0 = self.frequencies[0]
        self.f1 = self.frequencies[-1]
        self.samples_per_symbol = int(self.sample_rate / self.baud_rate)
        self.samples_per_bit = self.samples_per_symbol  # Alias for chart preview helpers

        # Precompute quadrature reference bases for all M carrier frequencies
        t = np.arange(self.samples_per_symbol, dtype=np.float32) / self.sample_rate
        self._cos_bank = [np.cos(2.0 * np.pi * f * t) for f in self.frequencies]
        self._sin_bank = [np.sin(2.0 * np.pi * f * t) for f in self.frequencies]

    def modulate(self, bits: str) -> np.ndarray:
        """
        Modulate a binary bit string into a continuous-phase M-FSK (2-FSK or 4-FSK) waveform.
        In 4-FSK mode, every 2 bits are grouped into a single acoustic symbol tone.
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

        edge_len = max(1, int(self.samples_per_symbol * 0.08))
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
        """Locate the active M-FSK carrier region with safe margins on both ends."""
        n_sym = len(waveform) // self.samples_per_symbol
        if n_sym < 6:
            return waveform

        trimmed = waveform[:n_sym * self.samples_per_symbol]
        matrix = trimmed.reshape(n_sym, self.samples_per_symbol)
        norm = float(self.samples_per_symbol ** 2)

        carrier_power = np.zeros(n_sym, dtype=np.float32)
        for c_vec, s_vec in zip(self._cos_bank, self._sin_bank):
            carrier_power += ((matrix @ c_vec) ** 2 + (matrix @ s_vec) ** 2) / norm

        peak_power = float(np.max(carrier_power))
        if peak_power < 1e-7:
            return np.array([], dtype=np.float32)

        active_idx = np.where(carrier_power > peak_power * 0.05)[0]
        if len(active_idx) < 6:
            return np.array([], dtype=np.float32)

        start_sym = max(0, int(active_idx[0]) - 4)
        end_sym = min(n_sym, int(active_idx[-1]) + 5)
        return waveform[start_sym * self.samples_per_symbol:end_sym * self.samples_per_symbol]

    def demodulate(self, received_waveform: np.ndarray) -> dict:
        """
        Fast vectorized non-coherent M-FSK demodulator (supports M=2 and M=4).
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

        # Compute energy matrix of shape (levels_m, n_sym)
        powers = np.zeros((self.levels_m, n_sym), dtype=np.float32)
        for m_idx in range(self.levels_m):
            c_vec = self._cos_bank[m_idx]
            s_vec = self._sin_bank[m_idx]
            powers[m_idx] = ((matrix @ c_vec) ** 2 + (matrix @ s_vec) ** 2) / norm

        total_p = np.sum(powers, axis=0)
        peak_p = float(np.max(total_p)) if len(total_p) > 0 else 0.0
        squelch_threshold = peak_p * 0.03

        best_Tone_indices = np.argmax(powers, axis=0)
        detected_chunks = []
        signal_powers = []
        noise_powers = []

        for i in range(n_sym):
            if total_p[i] < squelch_threshold:
                detected_chunks.append("00" if self.levels_m == 4 else "0")
                continue

            winner = int(best_Tone_indices[i])
            win_power = float(powers[winner, i])
            other_power = float(total_p[i] - win_power) / max(1, self.levels_m - 1)

            signal_powers.append(win_power)
            noise_powers.append(max(other_power, 1e-12))

            if self.levels_m == 4:
                detected_chunks.append(self.SYMBOL_BITS_4FSK[winner])
            else:
                detected_chunks.append('1' if winner == 1 else '0')

        avg_sig = float(np.mean(signal_powers)) if signal_powers else 1.0
        avg_noise = float(np.mean(noise_powers)) if noise_powers else 1e-6
        snr_linear = avg_sig / max(avg_noise, 1e-12)

        return {
            "bits": ''.join(detected_chunks),
            "f0_energies": [round(float(x), 6) for x in powers[0, :60]],
            "f1_energies": [round(float(x), 6) for x in powers[-1, :60]],
            "metrics": self.compute_channel_metrics(self.bandwidth, snr_linear, self.levels_m)
        }

    def demodulate_stream(self, received_waveform: np.ndarray, use_hamming: bool = True) -> dict:
        """
        Scan all 16 sub-symbol timing phases across the active waveform and
        aggregate all decoded word frames.
        """
        if len(received_waveform) > 0:
            received_waveform = received_waveform - np.mean(received_waveform)

        active_wave = self._isolate_carrier_burst(received_waveform)
        if len(active_wave) < self.samples_per_symbol * 10:
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
        step = max(1, self.samples_per_symbol // num_phases)

        for phase_idx in range(num_phases):
            offset = phase_idx * step
            if offset >= len(active_wave) - self.samples_per_symbol:
                break

            shifted_wave = active_wave[offset:]
            demod_candidate = self.demodulate(shifted_wave)
            if best_metrics is None:
                best_metrics = demod_candidate["metrics"]

            # In 4-FSK, also check a 1-bit shifted alignment in case a frame starts on an odd bit
            bit_candidates = [demod_candidate["bits"]]
            if self.levels_m == 4 and len(demod_candidate["bits"]) > 2:
                bit_candidates.append(demod_candidate["bits"][1:])

            for cand_bits in bit_candidates:
                frames = ChannelCoder.parse_all_frames(cand_bits, use_hamming=use_hamming)
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
        levels_m: int = 4
    ) -> dict:
        """
        Compute theoretical digital communication parameters (Slides 39-56):
        - SNR (dB)           : 10 * log10(SNR)
        - Nyquist Bit Rate   : Rb = 2 * B * log2(M)
        - Shannon Capacity   : C = B * log2(1 + SNR)
        - Effective Bit Rate : Rs * log2(M)
        """
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