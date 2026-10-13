# transceiver/engine/test_audio.py
import sounddevice as sd
import numpy as np
from source_coding import SourceCoder
from channel_coding import ChannelCoder
from modem_phy import AcousticModemPHY

def run_live_audio_test():
    print("=" * 65)
    print("LIVE ACOUSTIC MODEM TEST (SPEAKER TRANSMISSION)")
    print("=" * 65)

    # 1. Prepare message and encode frame
    text = "Hi UoM!"
    raw_bits = SourceCoder.text_to_bits(text)
    frame_data = ChannelCoder.build_frame(raw_bits, use_hamming=True)
    tx_bits = frame_data["frame_bits"]

    # 2. Modulate bits into audible FSK sound wave (baud_rate=25 bits/sec)
    # Set ultrasonic=True if you want to test silent ultrasonic transmission!
    modem = AcousticModemPHY(sample_rate=44100, baud_rate=25, ultrasonic=False)
    tx_waveform = modem.modulate(tx_bits)

    duration_sec = len(tx_waveform) / modem.sample_rate
    print(f"Message to Transmit : '{text}'")
    print(f"Total Frame Bits    : {len(tx_bits)} bits")
    print(f"Carrier Frequencies : f0={modem.f0} Hz ('0'), f1={modem.f1} Hz ('1')")
    print(f"Audio Duration      : {duration_sec:.2f} seconds")
    print("\n[🔊] Playing FSK modulated audio through your laptop speaker...")

    # Play sound out loud through the laptop speaker
    sd.play(tx_waveform, samplerate=modem.sample_rate)
    sd.wait()
    print("[✓] Audio playback finished!")

    # 3. Simulate acoustic environment noise (AWGN) & Demodulate
    noise = np.random.normal(0, 0.15, size=tx_waveform.shape).astype(np.float32)
    rx_waveform = tx_waveform + noise

    demod_res = modem.demodulate(rx_waveform)
    parsed_frame = ChannelCoder.parse_frame(demod_res["bits"], use_hamming=True)
    recovered_text = SourceCoder.bits_to_text(parsed_frame["payload_bits"])

    metrics = demod_res["metrics"]
    print("\n--- Receiver & Channel Analysis (Shannon / Nyquist) ---")
    print(f"Estimated SNR       : {metrics['snr_db']} dB")
    print(f"Nyquist Max Rate    : {metrics['nyquist_max_bps']} bps")
    print(f"Shannon Capacity    : {metrics['shannon_capacity_bps']} bps")
    print(f"CRC-8 Check         : {'PASSED' if parsed_frame['crc_passed'] else 'FAILED'}")
    print(f"Recovered Message   : '{recovered_text}'")

if __name__ == "__main__":
    run_live_audio_test()