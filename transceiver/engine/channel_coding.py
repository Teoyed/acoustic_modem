# transceiver/engine/channel_coding.py
import random

class ChannelCoder:
    """
    Channel Coding and Data Link Layer Framing module.
    Includes CRC-8 error detection, Hamming(7,4) Forward Error Correction (FEC),
    HDLC framing with bit stuffing, and channel error injection.
    """
    PREAMBLE = "10101010"      # 1-byte alternating preamble for receiver synchronization
    HDLC_FLAG = "01111110"     # Standard HDLC frame delimiter flag (0x7E)
    CRC8_POLY = "100000111"    # Standard CRC-8 generator polynomial (x^8 + x^2 + x + 1)

    # ==========================================
    # 1. Cyclic Redundancy Check (CRC)
    # ==========================================
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

    # ==========================================
    # 2. Forward Error Correction: Hamming(7,4)
    # ==========================================
    @staticmethod
    def hamming74_encode(data_bits: str) -> str:
        """
        Encode data bits using Hamming(7,4) block code.
        Maps every 4 data bits (d1, d2, d3, d4) to a 7-bit codeword:
        [p1, p2, d1, p4, d2, d3, d4]
        """
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
        """
        Decode Hamming(7,4) codewords and automatically correct single-bit errors per block.
        Returns the decoded payload bits and metadata of all corrected bit positions.
        """
        decoded_data = []
        corrected_positions = []
        blocks_with_error = 0

        valid_len = (len(encoded_bits) // 7) * 7
        for block_idx, i in enumerate(range(0, valid_len, 7)):
            block = [int(b) for b in encoded_bits[i:i+7]]
            b1, b2, b3, b4, b5, b6, b7 = block

            # Calculate 3-bit syndrome (s4, s2, s1)
            s1 = b1 ^ b3 ^ b5 ^ b7
            s2 = b2 ^ b3 ^ b6 ^ b7
            s4 = b4 ^ b5 ^ b6 ^ b7

            # 1-based index of the corrupted bit within the 7-bit block (0 means no error)
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
                # Flip the corrupted bit back to its original value (FEC)
                block[error_pos - 1] ^= 1

            # Extract the 4 original data bits at indices 2, 4, 5, 6
            decoded_data.append(f"{block[2]}{block[4]}{block[5]}{block[6]}")

        return {
            "decoded_bits": ''.join(decoded_data),
            "corrected_count": blocks_with_error,
            "corrected_positions": corrected_positions
        }

    # ==========================================
    # 3. HDLC Framing and Bit Stuffing
    # ==========================================
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

    @classmethod
    def build_frame(cls, payload_bits: str, use_hamming: bool = True) -> dict:
        """
        Construct a complete Data Link Layer frame:
        1. Compute and append CRC-8 (FCS) to payload
        2. Encode with Hamming(7,4) if enabled
        3. Apply HDLC bit stuffing
        4. Prepend Preamble + Start Flag and append End Flag
        """
        crc = cls.compute_crc(payload_bits)
        data_with_crc = payload_bits + crc

        if use_hamming:
            protected_bits = cls.hamming74_encode(data_with_crc)
        else:
            protected_bits = data_with_crc

        stuffed_body = cls.bit_stuff(protected_bits)
        full_frame = f"{cls.PREAMBLE}{cls.HDLC_FLAG}{stuffed_body}{cls.HDLC_FLAG}"

        return {
            "frame_bits": full_frame,
            "payload_len": len(payload_bits),
            "crc_bits": crc,
            "protected_len": len(protected_bits),
            "stuffed_len": len(stuffed_body),
            "total_frame_len": len(full_frame),
            "overhead_bits": len(full_frame) - len(payload_bits)
        }

    @classmethod
    def parse_frame(cls, received_frame_bits: str, use_hamming: bool = True) -> dict:
        """
        Parse a received frame stream:
        Locate HDLC flags, unstuff bits, apply Hamming(7,4) error correction, and verify CRC-8.
        """
        first_flag = received_frame_bits.find(cls.HDLC_FLAG)
        if first_flag == -1:
            return {"valid_frame": False, "error": "HDLC start flag (01111110) not found."}

        body_start = first_flag + len(cls.HDLC_FLAG)
        second_flag = received_frame_bits.find(cls.HDLC_FLAG, body_start)
        if second_flag == -1:
            return {"valid_frame": False, "error": "HDLC end flag (01111110) not found."}

        stuffed_body = received_frame_bits[body_start:second_flag]
        protected_bits = cls.bit_unstuff(stuffed_body)

        corrected_info = []
        if use_hamming:
            hamming_res = cls.hamming74_decode(protected_bits)
            data_with_crc = hamming_res["decoded_bits"]
            corrected_info = hamming_res["corrected_positions"]
        else:
            data_with_crc = protected_bits

        crc_len = len(cls.CRC8_POLY) - 1
        if len(data_with_crc) < crc_len:
            return {"valid_frame": False, "error": "Frame payload is shorter than CRC length."}

        crc_ok = cls.verify_crc(data_with_crc)
        payload_bits = data_with_crc[:-crc_len]
        extracted_crc = data_with_crc[-crc_len:]

        return {
            "valid_frame": True,
            "crc_passed": crc_ok,
            "payload_bits": payload_bits,
            "extracted_crc": extracted_crc,
            "corrected_errors_count": len(corrected_info),
            "corrected_positions": corrected_info
        }

    # ==========================================
    # 4. Channel Error Simulator
    # ==========================================
    @staticmethod
    def inject_channel_error(bits: str, mode: str = "single", burst_len: int = 3) -> tuple[str, list[int]]:
        """
        Inject simulated transmission errors into a bit sequence:
        - 'single': Flip 1 random bit
        - 'multiple': Flip 1 bit in multiple separate 7-bit blocks
        - 'burst': Flip a contiguous sequence of bits of length `burst_len`
        """
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