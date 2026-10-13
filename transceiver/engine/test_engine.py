# transceiver/engine/test_engine.py
from source_coding import SourceCoder
from channel_coding import ChannelCoder

def run_tests():
    print("=" * 65)
    print("TEST 1: Text Transmission with HDLC Framing & Hamming(7,4) FEC")
    print("=" * 65)

    message = "University of Mazandaran - Data Communications"
    raw_bits = SourceCoder.text_to_bits(message)
    print(f"Original Text: {message}")
    print(f"Raw UTF-8 Payload Length: {len(raw_bits)} bits")

    # Build Data Link Layer Frame
    frame_info = ChannelCoder.build_frame(raw_bits, use_hamming=True)
    frame_bits = frame_info["frame_bits"]
    print(f"Computed CRC-8 (FCS): {frame_info['crc_bits']}")
    print(f"Total Frame Length (with FEC & HDLC Flags): {frame_info['total_frame_len']} bits")

    # Inject simulated channel noise into the frame body (between HDLC flags)
    header_offset = len(ChannelCoder.PREAMBLE) + len(ChannelCoder.HDLC_FLAG)
    body = frame_bits[header_offset:-len(ChannelCoder.HDLC_FLAG)]
    noisy_body, error_indices = ChannelCoder.inject_channel_error(body, mode="multiple")
    noisy_frame = frame_bits[:header_offset] + noisy_body + ChannelCoder.HDLC_FLAG

    print(f"\n[!] Channel Noise Injected! Flipped bit indices in body: {error_indices}")

    # Parse and decode at Receiver
    rx_result = ChannelCoder.parse_frame(noisy_frame, use_hamming=True)
    recovered_text = SourceCoder.bits_to_text(rx_result["payload_bits"])

    print(f"[+] Errors Corrected by Hamming(7,4) FEC: {rx_result['corrected_errors_count']}")
    print(f"[+] Final CRC-8 Integrity Check: {'PASSED' if rx_result['crc_passed'] else 'FAILED'}")
    print(f"[+] Recovered Text at Receiver: {recovered_text}")

    print("\n" + "=" * 65)
    print("TEST 2: Binary Fax Image Lossless Compression using RLE")
    print("=" * 65)

    # 8x8 binary image matrix with large uniform white/black regions
    sample_fax_image = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    img_bits = SourceCoder.binary_matrix_to_bits(sample_fax_image)
    rle_res = SourceCoder.rle_encode(img_bits)
    decoded_img_bits = SourceCoder.rle_decode(rle_res["encoded_bits"])

    print(f"Raw Image Size: {rle_res['original_len']} bits")
    print(f"RLE Compressed Size: {rle_res['compressed_len']} bits (Saved: {rle_res['saving_percent']}%)")
    print(f"Lossless Reconstruction Verified: {img_bits == decoded_img_bits}")

if __name__ == "__main__":
    run_tests()