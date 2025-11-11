# LionOfJudo: 3-Camera Hardware Setup Guide

**Last Updated:** 2025-11-11
**System Cost:** ~$431 USD
**Build Time:** 4-6 hours

This guide walks you through building a synchronized 3-camera video capture system for judo training analysis. This is the **Phase 1 MVP** - start here before scaling up.

---

## Why Start with 3 Cameras?

- **Cost:** Half the price of 6-camera system ($431 vs $832)
- **Coverage:** Front + Side + Back angles capture 80% of techniques
- **Still Synced:** Hardware-synchronized via Pi Pico (same precision as 6-camera)
- **Expandable:** Add 3 more cameras later if needed (same Pi Pico can handle 6+)

---

## Bill of Materials

### Core Components

| Item | Specification | Qty | Unit Price | Total | Where to Buy |
|------|--------------|-----|------------|-------|--------------|
| Raspberry Pi 4 Model B | 4GB RAM | 3 | $65 | $195 | [Adafruit](https://www.adafruit.com), [CanaKit](https://www.canakit.com) |
| Arducam Global Shutter Camera | With trigger pin support | 3 | $40 | $120 | [Arducam Store](https://www.arducam.com) |
| MicroSD Card | 64GB, Class 10, A1 rated | 3 | $12 | $36 | Amazon, local electronics store |
| Raspberry Pi Pico | Any variant | 1 | $20 | $20 | Adafruit, Pimoroni |
| USB-C Power Supply | Official Pi 4 PSU (5V 3A) | 3 | $10 | $30 | Same as Pi source |
| Ethernet Switch | 8-port Gigabit | 1 | $25 | $25 | Amazon, TP-Link |
| Ethernet Cables | Cat6, 2m length | 4 | $1.25 | $5 | Amazon |
| **TOTAL** | | | | **$431** | |

### Additional Supplies (Not in Budget)

- Soldering iron + solder (if you don't have one)
- Jumper wires (male-to-female, 20cm) - pack of 40 for $5
- Heat shrink tubing or electrical tape
- Camera mounts/tripods (DIY with PVC pipe is cheapest)
- Extension cords for power

---

## Camera Placement Strategy

For a standard 8×8m judo mat:

```
                   [Camera 2: SIDE VIEW]
                          │
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    │                     │                     │
    │        JUDO MAT (8m x 8m)                │
    │                     │                     │
    │                     │                     │
    └─────────────────────┼─────────────────────┘
                          │
    [Camera 1: FRONT]     │      [Camera 3: BACK]
         (main view)
```

**Camera Heights:**
- Camera 1 (Front): 1.5m high, angled down 20°
- Camera 2 (Side): 1.5m high, angled down 20°
- Camera 3 (Back): 1.5m high, angled down 20°

**Why These Angles:**
- High enough to see foot placement and ground techniques
- Low enough to capture grips and upper body detail
- 20° downward angle reduces floor reflection from mats

---

## Step 1: Prepare Raspberry Pis (×3)

### 1.1 Flash Operating System

**For Each Pi:**

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Insert microSD card into your computer
3. Flash **Raspberry Pi OS Lite (64-bit)** (no desktop needed)
4. Before writing:
   - Click "Edit Settings"
   - Set hostname: `judo-cam-1`, `judo-cam-2`, `judo-cam-3`
   - Enable SSH (use password authentication)
   - Set username: `judo` / password: `[your-password]`
   - Configure WiFi (for initial setup only - will use Ethernet later)
   - Set locale/timezone

5. Write to SD card (takes ~5 minutes)

### 1.2 Initial Pi Configuration

**Repeat for each Pi:**

1. Insert SD card, power on Pi
2. SSH into Pi from your computer:
   ```bash
   ssh judo@judo-cam-1.local
   ```

3. Update system:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

4. Install required packages:
   ```bash
   sudo apt install -y python3-pip python3-opencv ffmpeg rsync
   pip3 install picamera2 --break-system-packages
   ```

5. Enable camera:
   ```bash
   sudo raspi-config
   # Navigate to: Interface Options → Camera → Enable
   # Reboot
   ```

6. Test camera is detected:
   ```bash
   libcamera-hello --list-cameras
   # Should show: "0 : arducam..."
   ```

---

## Step 2: Build Pi Pico Sync Generator

### 2.1 Hardware Connections

**Pico Pinout for 3 Cameras:**

```
Raspberry Pi Pico
┌─────────────────┐
│                 │
│  GP0 ──────────┼───→ Camera 1 Trigger (GPIO 4)
│  GP1 ──────────┼───→ Camera 2 Trigger (GPIO 4)
│  GP2 ──────────┼───→ Camera 3 Trigger (GPIO 4)
│                 │
│  GND ──────────┼───→ Common Ground (all 3 Pis)
│                 │
└─────────────────┘
```

**Connection Details:**
- Use female-to-female jumper wires
- Pico GP0/GP1/GP2 → Arducam trigger pin (labeled "FSIN" on camera board)
- Pico GND → Raspberry Pi GPIO Pin 6 (ground) on each Pi
- Keep wires short (<30cm) to reduce electrical noise

### 2.2 Pico Programming

1. Download [Thonny IDE](https://thonny.org/)
2. Connect Pico via USB while holding BOOTSEL button
3. Install MicroPython firmware (Thonny will prompt)
4. Copy this code to `main.py` on the Pico:

```python
from machine import Pin, Timer
import time

# Define trigger pins for 3 cameras
trigger_pins = [
    Pin(0, Pin.OUT),  # Camera 1
    Pin(1, Pin.OUT),  # Camera 2
    Pin(2, Pin.OUT),  # Camera 3
]

# Initialize all triggers LOW
for pin in trigger_pins:
    pin.value(0)

def trigger_cameras():
    """Send simultaneous pulse to all 3 cameras"""
    # Set all HIGH simultaneously
    for pin in trigger_pins:
        pin.value(1)

    # Hold HIGH for 10ms (camera exposure trigger)
    time.sleep_ms(10)

    # Set all LOW simultaneously
    for pin in trigger_pins:
        pin.value(0)

def start_recording():
    """Trigger at 30fps for synchronized video"""
    # 30 fps = trigger every 33.33ms
    timer = Timer()
    timer.init(freq=30, mode=Timer.PERIODIC, callback=lambda t: trigger_cameras())
    print("Recording started at 30fps...")

# Auto-start on power-up
print("Pi Pico Sync Generator Ready")
print("Starting in 5 seconds...")
time.sleep(5)
start_recording()
```

5. Save and disconnect Pico
6. When powered, Pico will auto-start triggering after 5 seconds

---

## Step 3: Camera Setup on Raspberry Pis

### 3.1 Attach Arducam to Each Pi

1. Power OFF the Pi
2. Connect Arducam ribbon cable to Pi camera port (blue tab faces USB ports)
3. Ensure cable is fully inserted (you'll feel a click)
4. **Important:** Connect the trigger wire:
   - Arducam FSIN pin → Pi GPIO 4
   - This receives the sync pulse from Pico

### 3.2 Camera Capture Script

Create `/home/judo/capture.py` on each Pi:

```python
#!/usr/bin/env python3
"""
Synchronized video capture for LionOfJudo
Triggered by external pulse from Pi Pico
"""

import time
from datetime import datetime
from picamera2 import Picamera2
from picamera2.encoders import H265Encoder
from picamera2.outputs import FileOutput
import RPi.GPIO as GPIO

# Configuration
TRIGGER_PIN = 4  # GPIO pin receiving sync pulse
OUTPUT_DIR = "/home/judo/recordings"
CAMERA_ID = "cam1"  # Change to cam2, cam3 for other Pis

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIGGER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Initialize camera
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1920, 1080), "format": "RGB888"},
    controls={"FrameRate": 30}
)
picam2.configure(config)

# H.265 encoder for small file sizes
encoder = H265Encoder(bitrate=4000000)  # 4 Mbps = ~2GB/hour

# Generate filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"{OUTPUT_DIR}/{CAMERA_ID}_{timestamp}.h265"

print(f"Camera {CAMERA_ID} ready")
print(f"Waiting for trigger pulse on GPIO {TRIGGER_PIN}...")
print(f"Output: {output_file}")

# Wait for trigger pulse to start
while GPIO.input(TRIGGER_PIN) == 0:
    time.sleep(0.1)

print("TRIGGER DETECTED - Recording started!")

# Start recording
picam2.start_recording(encoder, output_file)

try:
    # Record until interrupted (Ctrl+C)
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping recording...")
    picam2.stop_recording()
    GPIO.cleanup()
    print(f"Video saved: {output_file}")
```

Make it executable:
```bash
chmod +x /home/judo/capture.py
mkdir -p /home/judo/recordings
```

---

## Step 4: Networking Setup

### 4.1 Connect Everything

```
Internet Router
      │
      ├─ Laptop (for SSH access)
      │
  Ethernet Switch
      ├─ judo-cam-1 (192.168.1.101)
      ├─ judo-cam-2 (192.168.1.102)
      └─ judo-cam-3 (192.168.1.103)
```

### 4.2 Assign Static IPs

On **each Pi**, edit network config:

```bash
sudo nano /etc/dhcpcd.conf
```

Add at the end (adjust for cam-2 and cam-3):

```
interface eth0
static ip_address=192.168.1.101/24
static routers=192.168.1.1
static domain_name_servers=8.8.8.8
```

**For cam-2:** Use `192.168.1.102`
**For cam-3:** Use `192.168.1.103`

Reboot:
```bash
sudo reboot
```

---

## Step 5: Testing the System

### 5.1 Power-On Sequence

1. Connect all Pis to Ethernet switch
2. Power on all 3 Raspberry Pis
3. Wait 30 seconds for boot
4. Power on Pi Pico (triggers will start after 5 seconds)

### 5.2 Start Recording

On your laptop, SSH to each camera and start capture:

**Terminal 1:**
```bash
ssh judo@192.168.1.101
python3 /home/judo/capture.py
```

**Terminal 2:**
```bash
ssh judo@192.168.1.102
python3 /home/judo/capture.py
```

**Terminal 3:**
```bash
ssh judo@192.168.1.103
python3 /home/judo/capture.py
```

All three should say "TRIGGER DETECTED - Recording started!" simultaneously.

### 5.3 Stop Recording

Press `Ctrl+C` in all three terminals when session ends.

### 5.4 Verify Sync

Download all 3 videos to your laptop:

```bash
scp judo@192.168.1.101:/home/judo/recordings/*.h265 ./cam1.h265
scp judo@192.168.1.102:/home/judo/recordings/*.h265 ./cam2.h265
scp judo@192.168.1.103:/home/judo/recordings/*.h265 ./cam3.h265
```

Play side-by-side in VLC:
- All videos should show the same instant
- Test with a visible event (clap, jump)
- Sync should be <50ms across all cameras

---

## Step 6: Upload to Hetzner (After Recording)

### 6.1 Setup Hetzner Storage Box

(See HETZNER_SETUP.md for detailed server setup)

1. Order 1TB Storage Box from Hetzner (~€3.81/month)
2. Note your credentials: `username`, `hostname`

### 6.2 Configure rsync Upload

Create `/home/judo/upload.sh` on **cam-1** (primary Pi):

```bash
#!/bin/bash
# Upload all recordings from 3 cameras to Hetzner

STORAGE_BOX="u123456@u123456.your-storagebox.de"
SESSION_DIR="/judo/$(date +%Y%m%d_%H%M%S)"

echo "Creating session directory on Hetzner..."
ssh $STORAGE_BOX "mkdir -p $SESSION_DIR"

echo "Uploading from Camera 1..."
rsync -avz /home/judo/recordings/ $STORAGE_BOX:$SESSION_DIR/cam1/

echo "Uploading from Camera 2..."
ssh judo@192.168.1.102 "rsync -avz /home/judo/recordings/ $STORAGE_BOX:$SESSION_DIR/cam2/"

echo "Uploading from Camera 3..."
ssh judo@192.168.1.103 "rsync -avz /home/judo/recordings/ $STORAGE_BOX:$SESSION_DIR/cam3/"

echo "Upload complete! Session: $SESSION_DIR"
echo "Cleaning local files..."
rm -rf /home/judo/recordings/*
ssh judo@192.168.1.102 "rm -rf /home/judo/recordings/*"
ssh judo@192.168.1.103 "rm -rf /home/judo/recordings/*"

echo "Triggering cloud processing..."
curl https://your-hetzner-server.com/api/process?session=$SESSION_DIR
```

Make executable:
```bash
chmod +x /home/judo/upload.sh
```

---

## Troubleshooting

### Cameras Not Syncing

**Problem:** Videos show different timings

**Solutions:**
1. Check Pi Pico is powered and running (LED should blink)
2. Verify trigger wires connected to correct GPIO pins
3. Check all Pis have same camera config (framerate, resolution)
4. Test Pico trigger with multimeter (should pulse 30 times/second)

### Camera Not Detected

**Problem:** `libcamera-hello` shows no cameras

**Solutions:**
1. Check ribbon cable fully inserted (both camera and Pi ends)
2. Try different camera port on Pi (if Pi 4 has two)
3. Update Pi firmware: `sudo rpi-update`
4. Check camera not damaged (test on different Pi)

### Upload Too Slow

**Problem:** 24 GB upload takes >2 hours

**Solutions:**
1. Check using Ethernet not WiFi
2. Test network speed: `iperf3 -c your-hetzner-server.com`
3. Compress before upload: `tar -czf session.tar.gz recordings/`
4. Schedule upload overnight if bandwidth limited

### Running Out of SD Card Space

**Problem:** Recording stops after 30 minutes

**Solutions:**
1. Use larger SD cards (128GB recommended for long sessions)
2. Upload and clean after each session (don't accumulate)
3. Monitor space: `df -h` before recording
4. Reduce bitrate in capture.py (trade quality for size)

---

## Next Steps

After successful testing:

1. **Build Camera Mounts:** PVC pipe or wood stands at 1.5m height
2. **Cable Management:** Secure power and network cables to avoid tripping
3. **Automate Startup:** Make capture script run on Pi boot (systemd service)
4. **Test Full Session:** Record 2-hour training session and upload
5. **Phase 0 Complete:** If OpenRouter recognition works, this hardware is production-ready!

---

## Safety Notes

- Never connect/disconnect cameras while Pi is powered
- Use official power supplies (cheap ones cause voltage drops = bad sync)
- Keep Pis cool (attach heatsinks in warm dojos)
- Backup SD card images after configuration (avoid rebuilding)
- Label all cables (you'll thank yourself later)

---

## Cost Optimization Tips for Serbian Schools

1. **Buy in Bulk:** Order 3-5 kits together for better shipping
2. **Local Alternatives:**
   - Check if Serbia has local Pi distributors (avoid import fees)
   - Generic cameras work if they have trigger pins (test first)
3. **Shared Equipment:** One school can build first, others watch and learn
4. **Second-Hand:** Older Pi 3B+ works fine (save $20/unit)
5. **DIY Mounts:** Wood or PVC is 90% cheaper than commercial tripods

---

## Maintenance Schedule

**After Each Session:**
- Upload videos and clean SD cards
- Check camera lenses for dust/smudges

**Monthly:**
- Update Pi OS: `sudo apt update && sudo apt upgrade`
- Check network cables for damage
- Verify sync accuracy with test video

**Every 6 Months:**
- Replace SD cards (they wear out with constant writing)
- Clean camera sensors with air blower
- Tighten all connections

---

## Appendix: Camera Calibration (Optional - Phase 2)

For 3D reconstruction, you'll need camera calibration data. This is NOT required for Phase 1 (OpenRouter works with uncalibrated cameras).

See CAMERA_CALIBRATION.md when ready for advanced features.

---

**Need Help?**
- GitHub Issues: https://github.com/pashadude/LionOfJudo/issues
- Serbian Judo Community: [add forum link]

**Good luck building your system! You're giving kids in Serbia a huge advantage!**
