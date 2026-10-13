# transceiver/engine/source_coding.py

class SourceCoder:
    """
    Source Encoder and Decoder module.
    Handles UTF-8 text conversion, binary image serialization,
    and lossless Run-Length Encoding (RLE) compression.
    """

    @staticmethod
    def text_to_bits(text: str) -> str:
        """Convert a UTF-8 string into a binary bit string."""
        byte_data = text.encode('utf-8')
        return ''.join(f'{byte:08b}' for byte in byte_data)

    @staticmethod
    def bits_to_text(bits: str) -> str:
        """Convert a binary bit string back into a UTF-8 string."""
        valid_len = (len(bits) // 8) * 8
        bits = bits[:valid_len]
        byte_array = bytearray(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))
        return byte_array.decode('utf-8', errors='replace')

    @staticmethod
    def binary_matrix_to_bits(matrix: list[list[int]]) -> str:
        """Flatten a 2D binary image matrix (0s and 1s) into a 1D bit string."""
        return ''.join(str(pixel) for row in matrix for pixel in row)

    @staticmethod
    def bits_to_binary_matrix(bits: str, width: int, height: int) -> list[list[int]]:
        """Reconstruct a 2D binary image matrix from a 1D bit string."""
        expected_len = width * height
        bits = bits[:expected_len].ljust(expected_len, '0')
        return [
            [int(bits[row * width + col]) for col in range(width)]
            for row in range(height)
        ]

    @staticmethod
    def rle_encode(bits: str) -> dict:
        """
        Compress a bit string using Run-Length Encoding (RLE).
        Each run is encoded as an 8-bit chunk:
        - 1st bit: bit value ('0' or '1')
        - Next 7 bits: run length (1 to 127)
        """
        if not bits:
            return {
                "encoded_bits": "",
                "original_len": 0,
                "compressed_len": 0,
                "saving_percent": 0.0,
                "ratio": 1.0
            }

        encoded_chunks = []
        current_bit = bits[0]
        count = 1

        for bit in bits[1:]:
            if bit == current_bit and count < 127:
                count += 1
            else:
                encoded_chunks.append(f"{current_bit}{count:07b}")
                current_bit = bit
                count = 1

        encoded_chunks.append(f"{current_bit}{count:07b}")
        encoded_bits = ''.join(encoded_chunks)

        orig_len = len(bits)
        comp_len = len(encoded_bits)
        ratio = round(comp_len / orig_len, 3) if orig_len > 0 else 1.0

        return {
            "encoded_bits": encoded_bits,
            "original_len": orig_len,
            "compressed_len": comp_len,
            "saving_percent": round((1.0 - ratio) * 100, 1),
            "ratio": ratio
        }

    @staticmethod
    def rle_decode(encoded_bits: str) -> str:
        """Decompress an RLE-encoded bit string back into the original bit sequence."""
        decoded_chunks = []
        valid_len = (len(encoded_bits) // 8) * 8
        for i in range(0, valid_len, 8):
            chunk = encoded_bits[i:i+8]
            bit_val = chunk[0]
            count = int(chunk[1:], 2)
            decoded_chunks.append(bit_val * count)
        return ''.join(decoded_chunks)