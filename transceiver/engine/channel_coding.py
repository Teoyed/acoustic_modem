# transceiver/engine/channel_coding.py
import random


class ChannelCoder:
    """
    Channel Coding and Data Link Layer Framing module.
    Includes CRC-8 error detection, Hamming(7,4) Forward Error Correction (FEC),
    HDLC framing with shared flags, packet sequence headers (Text, Fax & Encryption),
    and channel error injection.
    """
    PREAMBLE = "10101010"      # 1-byte alternating preamble for receiver synchronization
    HDLC_FLAG = "01111110"     # Standard HDLC frame delimiter flag (0x7E)
    CRC8_POLY = "100000111"    # Standard CRC-8 generator polynomial (x^8 + x^2 + x + 1)

    @staticmethod
    def _xor(a: str, b: str) -> str:
        """Perform bitwise XOR between two equal-length bit strings (excluding MSB)."""
        return ''.join('0' if x == y else '1' for x, y in zip(a[1:], b[1:]))

    @classmethod
    def compute_crc(cls, data_bits: str, polynomial: str = None) -> str:
        """Compute the CRC remainder (Frame Check Sequence - FCS) for a bit string."""
        if polynomial is None:
            polynomial = cls.CRC8_POLY

        poly_len = len(polynomial)
        padded_data = data_bits + '0' * (poly_len - 1)
        pick = poly_len
        tmp = padded_data[:pick]

        while pick < len(padded_data):
            if tmp[0] == '1':
                tmp = cls._xor(polynomial, tmp) + padded_data[pick]
            else:
                tmp = cls._xor('0' * pick, tmp) + padded_data[pick]
            pick += 1

        if tmp[0] == '1':
            tmp = cls._xor(polynomial, tmp)
        else:
            tmp = cls._xor('0' * pick, tmp)

        return tmp

    @classmethod
    def verify_crc(cls, data_with_crc: str, polynomial: str = None) -> bool:
        """Verify frame integrity at the receiver; returns True if remainder is all zeros."""
        remainder = cls.compute_crc(data_with_crc, polynomial)
        return set(remainder) == {'0'}

    @staticmethod
    def hamming74_encode(data_bits: str) -> str:
        """Encode data bits using Hamming(7,4) block code."""
        pad_len = (4 - (len(data_bits) % 4)) % 4
        padded_bits = data_bits + ('0' * pad_len)

        encoded_blocks = []
        for i in range(0, len(padded_bits), 4):
            d1, d2, d3, d4 = [int(b) for b in padded_bits[i:i+4]]
            p1 = d1 ^ d2 ^ d4
            p2 = d1 ^ d3 ^ d4
            p4 = d2 ^ d3 ^ d4
            encoded_blocks.append(f"{p1}{p2}{d1}{p4}{d2}{d3}{d4}")

        return ''.join(encoded_blocks)

    @staticmethod
    def hamming74_decode(encoded_bits: str) -> dict:
        """Decode Hamming(7,4) codewords and automatically correct single-bit errors per block."""
        decoded_data = []
        corrected_positions = []
        blocks_with_error = 0

        valid_len = (len(encoded_bits) // 7) * 7
        for block_idx, i in enumerate(range(0, valid_len, 7)):
            block = [int(b) for b in encoded_bits[i:i+7]]
            b1, b2, b3, b4, b5, b6, b7 = block

            s1 = b1 ^ b3 ^ b5 ^ b7
            s2 = b2 ^ b3 ^ b6 ^ b7
            s4 = b4 ^ b5 ^ b6 ^ b7

            error_pos = (s4 << 2) | (s2 << 1) | s1

            if error_pos != 0:
                blocks_with_error += 1
                global_bit_index = i + (error_pos - 1)
                corrected_positions.append({
                    "block_index": block_idx,
                    "bit_in_block": error_pos,
                    "global_bit_index": global_bit_index,
                    "flipped_from": block[error_pos - 1],
                    "corrected_to": 1 - block[error_pos - 1]
                })
                block[error_pos - 1] ^= 1

            decoded_data.append(f"{block[2]}{block[4]}{block[5]}{block[6]}")

        return {
            "decoded_bits": ''.join(decoded_data),
            "corrected_count": blocks_with_error,
            "corrected_positions": corrected_positions
        }

    @staticmethod
    def bit_stuff(bits: str) -> str:
        """Insert a '0' bit after every 5 consecutive '1' bits to prevent false flag detection."""
        stuffed = []
        ones_count = 0
        for b in bits:
            stuffed.append(b)
            if b == '1':
                ones_count += 1
                if ones_count == 5:
                    stuffed.append('0')
                    ones_count = 0
            else:
                ones_count = 0
        return ''.join(stuffed)

    @staticmethod
    def bit_unstuff(stuffed_bits: str) -> str:
        """Remove stuffed '0' bits that follow 5 consecutive '1' bits at the receiver."""
        unstuffed = []
        ones_count = 0
        i = 0
        while i < len(stuffed_bits):
            b = stuffed_bits[i]
            unstuffed.append(b)
            if b == '1':
                ones_count += 1
                if ones_count == 5:
                    if i + 1 < len(stuffed_bits) and stuffed_bits[i + 1] == '0':
                        i += 1
                    ones_count = 0
            else:
                ones_count = 0
            i += 1
        return ''.join(unstuffed)

    @staticmethod
    def pack_packet_header(
        msg_id: int,
        seq_num: int,
        is_last: bool,
        stream_mode: bool,
        is_fax: bool = False,
        is_encrypted: bool = False
    ) -> str:
        """
        Build a 12-bit Data Link packet header:
        - Bits 0..2 (3 bits) : Message Session ID (1..7)
        - Bit 3     (1 bit)  : Delivery Mode (1 = Live Stream, 0 = Full at End)
        - Bit 4     (1 bit)  : Last Packet Flag (1 = Final frame, 0 = More follow)
        - Bit 5     (1 bit)  : Payload Type (1 = 16x16 RLE Fax Image, 0 = UTF-8 Text)
        - Bit 6     (1 bit)  : Encryption Flag (1 = Passkey Encrypted, 0 = Plaintext)
        - Bits 7..11(5 bits) : Sequence Number (0..31)
        """
        msg_id_bits = f"{msg_id & 0x07:03b}"
        mode_bit = '1' if stream_mode else '0'
        last_bit = '1' if is_last else '0'
        fax_bit = '1' if is_fax else '0'
        enc_bit = '1' if is_encrypted else '0'
        seq_bits = f"{seq_num & 0x1F:05b}"
        return f"{msg_id_bits}{mode_bit}{last_bit}{fax_bit}{enc_bit}{seq_bits}"

    @staticmethod
    def unpack_packet_header(payload_bits: str) -> dict:
        """Extract and validate the 12-bit packet header and remaining payload bits."""
        if len(payload_bits) < 20:
            return {"valid": False}
        header = payload_bits[:12]
        content_bits = payload_bits[12:]
        msg_id = int(header[0:3], 2)
        seq_num = int(header[7:12], 2)

        if msg_id == 0 or seq_num > 31:
            return {"valid": False}

        return {
            "valid": True,
            "msg_id": msg_id,
            "stream_mode": header[3] == '1',
            "is_last": header[4] == '1',
            "is_fax": header[5] == '1',
            "is_encrypted": header[6] == '1',
            "seq_num": seq_num,
            "content_bits": content_bits
        }

    @classmethod
    def encode_frame_body(cls, payload_bits: str, use_hamming: bool = True) -> dict:
        """Compute CRC-8, apply optional Hamming(7,4) FEC, and perform HDLC bit stuffing."""
        crc = cls.compute_crc(payload_bits)
        data_with_crc = payload_bits + crc

        if use_hamming:
            protected_bits = cls.hamming74_encode(data_with_crc)
        else:
            protected_bits = data_with_crc

        stuffed_body = cls.bit_stuff(protected_bits)
        return {
            "stuffed_body": stuffed_body,
            "crc_bits": crc,
            "protected_len": len(protected_bits),
            "stuffed_len": len(stuffed_body)
        }

    @classmethod
    def build_frame(cls, payload_bits: str, use_hamming: bool = True) -> dict:
        """Construct a standalone Data Link Layer frame with Preamble and HDLC flags."""
        body_info = cls.encode_frame_body(payload_bits, use_hamming=use_hamming)
        stuffed_body = body_info["stuffed_body"]
        full_frame = f"{cls.PREAMBLE}{cls.HDLC_FLAG}{stuffed_body}{cls.HDLC_FLAG}"

        return {
            "frame_bits": full_frame,
            "payload_len": len(payload_bits),
            "crc_bits": body_info["crc_bits"],
            "protected_len": body_info["protected_len"],
            "stuffed_len": body_info["stuffed_len"],
            "total_frame_len": len(full_frame),
            "overhead_bits": len(full_frame) - len(payload_bits)
        }

    @classmethod
    def build_shared_flag_stream(cls, payloads: list[str], use_hamming: bool = True) -> dict:
        """Build a back-to-back HDLC frame stream using Flag Sharing."""
        bodies = []
        cumulative_bits = []
        current_len = len(cls.PREAMBLE) + len(cls.HDLC_FLAG)
        first_crc = "00000000"

        for idx, p_bits in enumerate(payloads):
            b_info = cls.encode_frame_body(p_bits, use_hamming=use_hamming)
            if idx == 0:
                first_crc = b_info["crc_bits"]
            bodies.append(b_info["stuffed_body"])
            current_len += len(b_info["stuffed_body"]) + len(cls.HDLC_FLAG)
            cumulative_bits.append(current_len)

        stream_bits = cls.PREAMBLE + cls.HDLC_FLAG + cls.HDLC_FLAG.join(bodies) + cls.HDLC_FLAG
        first_single_frame = f"{cls.PREAMBLE}{cls.HDLC_FLAG}{bodies[0]}{cls.HDLC_FLAG}" if bodies else ""

        return {
            "stream_bits": stream_bits,
            "first_frame_bits": first_single_frame,
            "first_crc": first_crc,
            "cumulative_bits": cumulative_bits,
            "total_stream_len": len(stream_bits)
        }

    @classmethod
    def parse_all_frames(
        cls,
        received_frame_bits: str,
        use_hamming: bool = True,
        strict_crc: bool = True
    ) -> list[dict]:
        """Scan a bit stream and extract HDLC frames between 01111110 flags."""
        flag = cls.HDLC_FLAG
        flag_len = len(flag)
        flag_positions = []
        pos = received_frame_bits.find(flag)
        while pos != -1:
            flag_positions.append(pos)
            pos = received_frame_bits.find(flag, pos + flag_len)

        if len(flag_positions) < 2:
            return []

        valid_frames = []
        fallback_frames = []
        crc_len = len(cls.CRC8_POLY) - 1
        used_up_to = -1

        for idx in range(len(flag_positions) - 1):
            if flag_positions[idx] < used_up_to:
                continue
            body_start = flag_positions[idx] + flag_len

            for end_idx in range(idx + 1, min(idx + 4, len(flag_positions))):
                body_end = flag_positions[end_idx]
                stuffed_body = received_frame_bits[body_start:body_end]
                if len(stuffed_body) < 28:
                    continue

                protected_bits = cls.bit_unstuff(stuffed_body)
                if use_hamming:
                    hamming_res = cls.hamming74_decode(protected_bits)
                    data_with_crc = hamming_res["decoded_bits"]
                    corrected_info = hamming_res["corrected_positions"]
                else:
                    data_with_crc = protected_bits
                    corrected_info = []

                if len(data_with_crc) < crc_len + 20:
                    continue

                crc_ok = cls.verify_crc(data_with_crc)
                payload_bits = data_with_crc[:-crc_len]
                extracted_crc = data_with_crc[-crc_len:]

                candidate = {
                    "valid_frame": True,
                    "crc_passed": crc_ok,
                    "payload_bits": payload_bits,
                    "extracted_crc": extracted_crc,
                    "corrected_errors_count": len(corrected_info),
                    "corrected_positions": corrected_info,
                    "matched_frame_bits": received_frame_bits[max(0, flag_positions[idx] - 8):body_end + flag_len]
                }

                if crc_ok:
                    valid_frames.append(candidate)
                    used_up_to = body_end
                    break
                elif not strict_crc:
                    fallback_frames.append(candidate)

        if valid_frames:
            return valid_frames
        return [] if strict_crc else fallback_frames[:1]

    @classmethod
    def parse_frame(cls, received_frame_bits: str, use_hamming: bool = True) -> dict:
        """Parse the first HDLC frame from a bit stream."""
        frames = cls.parse_all_frames(received_frame_bits, use_hamming=use_hamming, strict_crc=False)
        if frames:
            return frames[0]
        return {"valid_frame": False, "error": "HDLC frame flags (01111110) not found."}

    @staticmethod
    def inject_channel_error(bits: str, mode: str = "single", burst_len: int = 3) -> tuple[str, list[int]]:
        """Inject simulated transmission errors into a bit sequence."""
        bit_list = list(bits)
        flipped_indices = []
        n = len(bit_list)
        if n == 0:
            return bits, []

        if mode == "single":
            idx = random.randint(0, n - 1)
            bit_list[idx] = '1' if bit_list[idx] == '0' else '0'
            flipped_indices.append(idx)
        elif mode == "multiple":
            for block_start in range(0, min(n - 7, 21), 7):
                idx = block_start + random.randint(0, 6)
                bit_list[idx] = '1' if bit_list[idx] == '0' else '0'
                flipped_indices.append(idx)
        elif mode == "burst":
            start_idx = random.randint(0, max(0, n - burst_len))
            for idx in range(start_idx, min(n, start_idx + burst_len)):
                bit_list[idx] = '1' if bit_list[idx] == '0' else '0'
                flipped_indices.append(idx)

        return ''.join(bit_list), flipped_indices