# Raspberry Pi IMU Collector

The Raspberry Pi is an optional post-training storage station.  It should not
sit in the live recording path.  During training the ESP32 wearables log to
their own flash; after training they reboot into WiFi download mode and the Pi
pulls the `.bin` logs over the MikroTik LAN.

## Recommended Hardware

Minimum:

- Raspberry Pi 3B+, 4, or 5
- 32 GB high-endurance microSD card for Raspberry Pi OS
- MikroTik hAP lite or any stable 2.4 GHz access point named `JudoNet`
- Ethernet cable from Pi to MikroTik
- Small HDMI screen, terminal, or tiny Pi display for status

Better:

- 64 GB high-endurance microSD for OS
- 64-128 GB USB flash drive or small USB SSD mounted at `/data/lionofjudo`
- Pi 4/5 if you want faster copying to the Mac

Do not store Sony/iPhone videos on the Pi in v1.  Video goes directly to the
Mac.  The Pi only stores IMU logs and collector status.

## Operating-System Image

Use Raspberry Pi Imager to flash the currently offered **Raspberry Pi OS Lite
(64-bit)** image for a Pi 3, 4, 5, or Zero 2 W. It is a command-line-only
image, which is exactly right for this collector; status remains visible in an
HDMI terminal or over SSH. For an original Pi Zero / Zero W, choose **Raspberry
Pi OS Lite (32-bit)** instead. Flashing a new card overwrites Pwnagotchi, so
use a separate microSD card if you want to keep that installation.

## Expected IMU Volume

Current firmware log format:

```text
16 bytes/sample * 200 samples/s = 3.2 KB/s
~11.5 MB/hour/unit
~23 MB/hour for chest + hip
```

Storage examples:

```text
2-hour session, 2 units      ~46 MB
100 two-hour sessions        ~4.6 GB
100 sessions + status files  still comfortably below 16 GB
```

A 32 GB card is enough for IMU logs.  Use a USB flash/SSD anyway so the OS card
is not the only copy.

## Network

Use the same SSID in `firmware/lionimu/config.h`:

```cpp
#define HOME_SSID "JudoNet"
#define HOME_PASS "your-password"
```

Important operational rule:

```text
Training boot: JudoNet OFF -> wearable does not find WiFi -> logs to flash.
After training: JudoNet ON -> reboot wearable -> download mode.
```

If JudoNet is already on when the wearable boots, the current firmware enters
download mode instead of logging mode.

## Pi Setup

On the Pi:

```bash
sudo mkdir -p /data/lionofjudo
sudo chown -R "$USER":"$USER" /data/lionofjudo
git clone https://github.com/pashadude/LionOfJudo.git ~/LionOfJudo
cd ~/LionOfJudo
python3 -m venv .venv
. .venv/bin/activate
```

The collector uses only Python standard library modules.  No heavy model
dependencies are needed on the Pi.

## After-Training Collection

1. Stop training and power off the wearables.
2. Turn on MikroTik `JudoNet` and the Pi.
3. Power on chest and hip wearables so they join WiFi download mode.
4. Run:

```bash
cd ~/LionOfJudo
. .venv/bin/activate
python tools/rpi_collect_imu.py \
  --root /data/lionofjudo \
  --session 2026-07-12 \
  --wipe
```

Output looks like:

```text
LionOfJudo IMU Collector
Session: 2026-07-12
IMU dir: /data/lionofjudo/sessions/2026-07-12/imu
Updated: 2026-07-12T21:34:02

chest: OK  battery 4.05V  free 8012KB  files 1  downloaded 18.42MB
  OK: chest_001.bin 18.42MB wiped
hip: OK  battery 4.01V  free 7976KB  files 1  downloaded 18.35MB
  OK: hip_001.bin 18.35MB wiped

TOTAL: 36.77MB
Next: copy this session folder to Mac and run pipeline.run_session.
```

The same information is written to:

```text
/data/lionofjudo/sessions/<date>/collector_status.txt
/data/lionofjudo/sessions/<date>/collector_status.json
```

Show the text file on a screen with:

```bash
watch -n 2 cat /data/lionofjudo/sessions/2026-07-12/collector_status.txt
```

## Copy To Mac

From the Mac:

```bash
rsync -av pi@raspberrypi.local:/data/lionofjudo/sessions/2026-07-12/ \
  sessions/2026-07-12/
```

Then run final sync/analysis on the Mac:

```bash
python -m pipeline.run_session \
  --sony /path/to/sony.MP4 \
  --iphone /path/to/iphone.MOV \
  --imu sessions/2026-07-12/imu \
  --out sessions/2026-07-12/
```

Each throw folder will contain:

```text
gi_biomechanics.csv
```

That CSV is the chest/hip IMU signal resampled onto the Sony video frame
timeline.
