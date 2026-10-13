# Acoustic Software Modem (Over-the-Air Data Transceiver)

An end-to-end digital communication system built with **Python** and **Django** that transmits UTF-8 text and binary fax images over audible and ultrasonic sound waves without relying on Wi-Fi, Bluetooth, or physical cables.

## System Architecture

```text
[Input Data] -> [Source Encoder (UTF-8 / RLE)] -> [Channel Encoder (CRC-8 + Hamming 7,4 + HDLC)] -> [FSK Modulator]
                                                                                                           |
                                                                                                  (Noisy Acoustic Channel)
                                                                                                           |
[Output Data] <- [Source Decoder] <- [Channel Decoder (FEC Repair + CRC Check)] <- [FSK Demodulator] <-----+