# transceiver/models.py
from django.db import models

class TransmissionLog(models.Model):
    """
    Database model to store experimental transmission results,
    channel metrics (SNR, BER, Shannon, Nyquist), and FEC performance.
    """
    CARRIER_CHOICES = [
        ('audible', 'Audible FSK (1500 / 2500 Hz)'),
        ('ultrasonic', 'Ultrasonic FSK (17500 / 18500 Hz)'),
    ]

    created_at = models.DateTimeField(auto_now_add=True)
    input_text = models.TextField()
    carrier_mode = models.CharField(max_length=20, choices=CARRIER_CHOICES, default='audible')
    baud_rate = models.IntegerField(default=25)

    # Compression & Framing Metrics
    use_rle = models.BooleanField(default=False)
    use_hamming = models.BooleanField(default=True)
    raw_bits_len = models.IntegerField()
    total_frame_bits = models.IntegerField()

    # Error Control & Channel Metrics
    injected_errors = models.IntegerField(default=0)
    corrected_errors = models.IntegerField(default=0)
    ber = models.FloatField(default=0.0)
    snr_db = models.FloatField(default=0.0)
    nyquist_bps = models.FloatField(default=0.0)
    shannon_bps = models.FloatField(default=0.0)
    crc_passed = models.BooleanField(default=True)
    recovered_text = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        status = "PASS" if self.crc_passed else "FAIL"
        return f"[{status}] {self.carrier_mode} @ {self.baud_rate} Bd | SNR: {self.snr_db} dB"