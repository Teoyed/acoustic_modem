# Acoustic Software Modem

A Django web application that sends UTF-8 text as FSK audio and decodes received WAV recordings. It demonstrates acoustic data transmission without Wi-Fi, Bluetooth, or a physical data cable.

## Features

- FSK modulation and demodulation over audible or ultrasonic carriers
- HDLC framing, CRC-8 error detection, and optional Hamming(7,4) error correction
- Optional RLE source encoding
- Multi-packet text transmission with packet timing information
- WAV audio output, microphone recording, and WAV file decoding
- Signal Lab charts and channel/FEC metrics

## Requirements

- Python 3.10 or newer
- A modern browser for the web interface
- A microphone and speakers/headphones for live acoustic transmission (optional)

## Run locally

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Teoyed/acoustic_modem.git
cd acoustic_modem
```

Create and activate a virtual environment, then install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Initialize the local SQLite database and start Django's development server:

```bash
python manage.py migrate
python manage.py runserver
```

Open <http://127.0.0.1:8000/> in your browser. The microphone feature works on localhost; browsers generally require HTTPS for microphone access on non-localhost hosts.

## Checks and examples

Run Django's configuration check and the engine demonstration:

```bash
python manage.py check
python transceiver/engine/test_engine.py
```

`transceiver/engine/test_audio.py` is an optional live speaker/microphone demonstration. It requires the additional `sounddevice` package and access to an audio output device.

## Development settings

The project uses SQLite and Django's development server by default. `DJANGO_SECRET_KEY` can be set in the environment to override the local development key. Do not use the development settings or fallback key for a production deployment; configure a private secret key, `DEBUG=False`, and appropriate `ALLOWED_HOSTS` before deploying.

The local database (`db.sqlite3`) is created by `migrate` and is not part of the source checkout.

## System architecture

```text
[UTF-8 Text] -> [Optional RLE] -> [HDLC + CRC-8 + Optional Hamming(7,4)]
             -> [FSK Modulator] -> (Acoustic Channel)
             -> [FSK Demodulator] -> [Frame/Error Decoder] -> [Recovered Text]
```
