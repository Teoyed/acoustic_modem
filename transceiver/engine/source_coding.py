# transceiver/engine/source_coding.py
import hashlib


class SourceCoder:
    """
    Source Encoder, Decoder, and Presentation Layer Security (OSI Layer 6) module.
    Handles UTF-8 text conversion, binary image serialization, lossless RLE compression,
    and symmetric stream-cipher encryption/decryption with key verification.
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
    def is_clean_text(text: str) -> bool:
        """
        Verify that decoded text contains only valid printable UTF-8 characters
        and no replacement chars ('\\ufffd') or control codes caused by noise.
        """
        if not text or '\ufffd' in text:
            return False
        for ch in text:
            if not (ch.isprintable() or ch in ('\n', '\t')):
                return False
        return True

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

    # ==========================================
    # Presentation Layer: Symmetric Encryption
    # ==========================================
    @staticmethod
    def _generate_keystream_bits(passkey: str, num_bits: int, salt: int = 0) -> str:
        """Generate a deterministic pseudo-random bitstream from a passkey using SHA-256."""
        stream_bits = []
        counter = 0
        while len(stream_bits) < num_bits:
            seed = f"sonicdrop-v1|{passkey}|{salt}|{counter}".encode('utf-8')
            digest = hashlib.sha256(seed).digest()
            for byte in digest:
                stream_bits.append(f"{byte:08b}")
            counter += 1
        return ''.join(stream_bits)[:num_bits]

    @classmethod
    def _compute_key_auth_tag(cls, plain_bits: str, passkey: str, salt: int = 0) -> str:
        """Compute an 8-bit authentication tag to detect wrong decryption passkeys."""
        tag_input = f"auth|{passkey}|{salt}|{plain_bits}".encode('utf-8')
        tag_byte = hashlib.sha256(tag_input).digest()[0]
        return f"{tag_byte:08b}"

    @classmethod
    def encrypt_bits(cls, plain_bits: str, passkey: str, salt: int = 0) -> str:
        """
        Encrypt a bit string with a symmetric passkey (OSI Presentation Layer).
        Prepends an 8-bit Key Verification Tag and XORs with a SHA-256 keystream.
        """
        if not passkey:
            return plain_bits
        auth_tag = cls._compute_key_auth_tag(plain_bits, passkey, salt=salt)
        combined = auth_tag + plain_bits
        keystream = cls._generate_keystream_bits(passkey, len(combined), salt=salt)
        return ''.join('1' if a != b else '0' for a, b in zip(combined, keystream))

    @classmethod
    def decrypt_bits(cls, cipher_bits: str, passkey: str, salt: int = 0) -> dict:
        """
        Attempt to decrypt an encrypted bit payload using the receiver's passkey.
        Returns `key_valid=True` only if the 8-bit authentication tag matches.
        """
        if len(cipher_bits) < 8:
            return {"plain_bits": "", "key_valid": False, "cipher_preview": "🔒"}

        if not passkey:
            return {
                "plain_bits": cipher_bits[8:],
                "key_valid": False,
                "cipher_preview": cls.bits_to_cipher_preview(cipher_bits[8:])
            }

        keystream = cls._generate_keystream_bits(passkey, len(cipher_bits), salt=salt)
        decrypted = ''.join('1' if a != b else '0' for a, b in zip(cipher_bits, keystream))
        extracted_tag = decrypted[:8]
        plain_bits = decrypted[8:]

        expected_tag = cls._compute_key_auth_tag(plain_bits, passkey, salt=salt)
        is_valid = (extracted_tag == expected_tag)

        return {
            "plain_bits": plain_bits,
            "key_valid": is_valid,
            "cipher_preview": cls.bits_to_cipher_preview(plain_bits if not is_valid else cipher_bits[8:])
        }

    @staticmethod
    def bits_to_cipher_preview(bits: str) -> str:
        """Render unauthenticated/wrong-key bits as visible scrambled ciphertext symbols."""
        glyphs = "0123456789ABCDEF#@$%&*?!~+="
        out = []
        for i in range(0, min(len(bits), 96), 8):
            chunk = bits[i:i+8]
            val = int(chunk, 2)
            out.append(glyphs[val % len(glyphs)])
        return "🔒 " + (''.join(out) or "****")

    # ==========================================
    # Lossless Run-Length Encoding (RLE)
    # ==========================================
    @staticmethod
    def rle_encode(bits: str) -> dict:
        """Compress a bit string using Run-Length Encoding (RLE)."""
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