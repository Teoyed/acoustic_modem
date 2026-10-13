# transceiver/engine/modem_phy.py
import math
import wave
import io
import numpy as np

class AcousticModemPHY:
    """
    Physical Layer (PHY) Acoustic Modem.
    Implements Binary Frequency Shift Keying (BFSK) modulation/demodulation,
    WAV generation, and Nyquist/Shannon channel capacity calculations.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        baud_rate: int = 20,
        ultrasonic: bool = False
    ):
        self.sample_rate = sample_rate  # Standard CD Audio 44.1 kHz
        self.baud_rate = baud_rate      # Symbols (bits) per second
        self.ultrasonic = ultrasonic

        # Set FSK carrier frequencies (f0 for bit '0', f1 for bit '1')
        if ultrasonic:
            self.f0 = 17500.0  # Near-ultrasonic frequency for '0' (Hz)
            self.f1 = 18500.0  # Near-ultrasonic frequency for '1' (Hz)
            self.bandwidth = 2000.0
        else:
            self.f0 = 1500.0   # Audible frequency for '0' (Hz)
            self.f1 = 2500.0   # Audible frequency for '1' (Hz)
            self.bandwidth = 2000.0

        self.samples_per_bit = int(self.sample_rate / self.baud_rate)

    # ==========================================
    # 1. FSK Digital Modulator (Transmitter)
    # ==========================================
    def modulate(self, bits: str) -> np.ndarray:
        """
        Modulate a binary string into a continuous-phase FSK audio waveform.
        Returns a normalized float32 numpy array in range [-1.0, 1.0].
        """
        if not bits:
            return np.array([], dtype=np.float32)

        total_samples = len(bits) * self.samples_per_bit
        waveform = np.zeros(total_samples, dtype=np.float32)

        # Continuous phase accumulator to prevent clicking noise between bit transitions
        phase = 0.0
        t_step = 1.0 / self.sample_rate

        # Smooth envelope window at the edges of each symbol to reduce spectral leakage
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

            # Update phase for the next symbol
            phase = (phase + 2.0 * np.pi * freq * self.samples_per_bit * t_step) % (2.0 * np.pi)

        return waveform * 0.8  # Scale amplitude to avoid audio clipping

    def waveform_to_wav_bytes(self, waveform: np.ndarray) -> bytes:
        """Convert normalized float32 waveform into 16-bit PCM WAV bytes (for Django HTTP response)."""
        pcm_int16 = np.int16(np.clip(waveform, -1.0, 1.0) * 32767)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)       # Mono
            wav_file.setsampwidth(2)       # 16-bit (2 bytes per sample)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_int16.tobytes())
        return buffer.getvalue()

    # ==========================================
    # 2. FSK Digital Demodulator (Receiver)
    # ==========================================
    @staticmethod
    def _goertzel_power(samples: np.ndarray, target_freq: float, sample_rate: int) -> float:
        """Compute signal power at a specific target frequency using the Goertzel algorithm."""
        n = len(samples)
        if n == 0:
            return 0.0
        k = int(0.5 + (n * target_freq) / sample_rate)
        omega = (2.0 * math.pi * k) / n
        coeff = 2.0 * math.cos(omega)

        s_prev = 0.0
        s_prev2 = 0.0
        for x in samples:
            s = x + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s

        power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
        return float(power / (n * n))

    def demodulate(self, received_waveform: np.ndarray) -> dict:
        """
        Demodulate an incoming audio waveform back into a binary string.
        Computes symbol energies at f0 and f1, estimates SNR, and outputs constellation points.
        """
        num_symbols = len(received_waveform) // self.samples_per_bit
        detected_bits = []
        signal_powers = []
        noise_powers = []
        f0_energies = []
        f1_energies = []

        for i in range(num_symbols):
            chunk = received_waveform[i * self.samples_per_bit:(i + 1) * self.samples_per_bit]
            p0 = self._goertzel_power(chunk, self.f0, self.sample_rate)
            p1 = self._goertzel_power(chunk, self.f1, self.sample_rate)

            f0_energies.append(round(p0, 6))
            f1_energies.append(round(p1, 6))

            if p1 > p0:
                detected_bits.append('1')
                signal_powers.append(p1)
                noise_powers.append(max(p0, 1e-12))
            else:
                detected_bits.append('0')
                signal_powers.append(p0)
                noise_powers.append(max(p1, 1e-12))

        avg_signal = float(np.mean(signal_powers)) if signal_powers else 1.0
        avg_noise = float(np.mean(noise_powers)) if noise_powers else 1e-6
        snr_linear = avg_signal / max(avg_noise, 1e-12)

        metrics = self.compute_channel_metrics(
            bandwidth_hz=self.bandwidth,
            snr_linear=snr_linear,
            levels_m=2
        )

        return {
            "bits": ''.join(detected_bits),
            "f0_energies": f0_energies,
            "f1_energies": f1_energies,
            "metrics": metrics
        }

    # ==========================================
    # 3. Theoretical Channel Metrics (Shannon & Nyquist)
    # ==========================================
    def compute_channel_metrics(self, bandwidth_hz: float, snr_linear: float, levels_m: int = 2) -> dict:
        """
        Calculate theoretical digital transmission parameters:
        - SNR in dB: 10 * log10(SNR)
        - Nyquist Bit Rate: Rb = 2 * B * log2(M)
        - Shannon Capacity: C = B * log2(1 + SNR)
        - Spectral Efficiency: eta = Rb / B
        """
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