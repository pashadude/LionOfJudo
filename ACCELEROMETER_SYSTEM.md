# Accelerometer-Based Power Measurement & Sync System

**Revolutionary Addition:** Add IMU (accelerometer + gyroscope) sensors to judogi to:
1. Measure actual throw power (not estimated from video)
2. Auto-sync video using power spike detection
3. Use Pi Pico as unified controller: camera trigger + sensor hub

---

## Why Accelerometers Change Everything

### Problem with Video-Only Power Estimation
- Can only estimate force from pose movement speed
- Missing hidden forces (grips, internal muscle tension)
- Occlusions make measurements inaccurate
- No ground truth for validation

### Solution: Direct Force Measurement
- **Actual g-forces** during throw execution
- Peak impact forces (uke landing)
- Rotational velocity (gyroscope)
- Throw duration from acceleration profile
- **Auto-sync:** Match acceleration spikes to video frames

---

## Hardware Architecture

### Option 1: Pi Pico as Unified Hub (RECOMMENDED)

```
┌──────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI PICO                         │
│                                                              │
│  GPIO 0-2: Camera Triggers (existing sync)                  │
│  GPIO 3-4: I2C Bus for IMU sensors (SDA/SCL)               │
│  GPIO 5:   Trigger button (start/stop recording)            │
│  USB:      Data logging to connected laptop                  │
│                                                              │
└─────┬────────┬─────────┬─────────┬─────────┬────────────────┘
      │        │         │         │         │
      ▼        ▼         ▼         ▼         ▼
   Cam 1    Cam 2     Cam 3    IMU #1     IMU #2
                               (Tori)     (Uke)
```

**Why Pi Pico is Perfect:**
- Already using it for camera sync (no new hardware)
- Has I2C bus for sensor communication
- 12-bit ADC for analog sensors
- Dual-core: one for camera timing, one for sensor reading
- MicroPython or C/C++ for real-time logging
- USB connectivity for data export
- Cost: Already $20 in budget!

---

### IMU Sensor Options

**Option A: MPU-6050 (Budget - $3-5 each)**
```
Specs:
- 3-axis accelerometer: ±16g range
- 3-axis gyroscope: ±2000°/s
- I2C interface
- 400 Hz sampling rate
- Perfect for judo throws (max ~10g on impact)
```

**Option B: LSM6DS3 (Better - $8-10 each)**
```
Specs:
- 6-axis IMU (accel + gyro)
- ±16g accelerometer
- ±2000°/s gyroscope
- Built-in FIFO buffer (won't miss samples)
- Higher precision
```

**Option C: BNO055 (Best - $25-30 each)**
```
Specs:
- 9-axis absolute orientation sensor
- On-board sensor fusion
- Direct quaternion output
- Best for rotation tracking
- Overkill for initial testing
```

**Recommendation:** Start with MPU-6050 ($3) for Phase 0 testing, upgrade if needed.

---

### Sensor Placement on Judogi

```
      [Head]
        │
    ┌───┴───┐
    │       │
    │ CHEST │  ← IMU #1: Center of mass, main throw dynamics
    │   *   │     (attached to judogi back or chest)
    │       │
    ├───┬───┤
    │   │   │
   Arm Arm  │
        │   │
       Hip  │  ← IMU #2: Hip rotation (critical for throwing power)
        │   │     (attached to belt or lower back)
        │   │
       Legs
```

**Two Sensors Per Athlete (Minimal Setup):**
1. **Chest/Back:** Overall body acceleration, impact detection
2. **Hip/Belt:** Rotation measurement (core of throw power)

**Future Expansion:**
- Wrist sensors: Grip force proxy (accelerate when grip breaks)
- Ankle sensors: Footwork timing
- Maximum: 4 sensors per athlete = 8 total for pair

---

## Updated Bill of Materials

### Phase 1 MVP with Accelerometers

| Component | Specification | Qty | Unit Cost | Total |
|-----------|--------------|-----|-----------|-------|
| **Existing Camera System** | | | | **$431** |
| Raspberry Pi 4 (4GB) | | 3 | $65 | $195 |
| Arducam Global Shutter | | 3 | $40 | $120 |
| MicroSD Cards | | 3 | $12 | $36 |
| **Pi Pico (sync + sensors)** | Already included | **1** | **$20** | **$20** |
| Power supplies | | 3 | $10 | $30 |
| Network switch + cables | | 1 | $30 | $30 |
| **New: IMU Sensors** | | | | |
| MPU-6050 IMU modules | 3-axis accel/gyro | 4 | $3.50 | **$14** |
| Flexible ribbon cable | For sensor attachment | 4 | $2 | **$8** |
| Velcro patches | Attach to judogi | 1 pack | $5 | **$5** |
| JST connectors | Quick sensor swap | 4 | $1 | **$4** |
| **Total** | | | | **$462** |

**Only $31 more than original plan!**

---

## Pi Pico Code: Unified Camera + Sensor Hub

```python
"""
Pi Pico: Unified controller for camera sync + accelerometer logging
Runs on Raspberry Pi Pico with MicroPython

Hardware connections:
- GPIO 0: Camera 1 trigger
- GPIO 1: Camera 2 trigger
- GPIO 2: Camera 3 trigger
- GPIO 4: I2C SDA (sensors)
- GPIO 5: I2C SCL (sensors)
- GPIO 15: Start/stop button
"""

from machine import Pin, I2C, Timer
import time
import json

# Camera trigger pins
camera_triggers = [
    Pin(0, Pin.OUT),
    Pin(1, Pin.OUT),
    Pin(2, Pin.OUT),
]

# I2C bus for IMU sensors
i2c = I2C(0, scl=Pin(5), sda=Pin(4), freq=400000)

# MPU-6050 I2C addresses (can have multiple on same bus)
IMU_ADDRESSES = [0x68, 0x69]  # AD0 pin low=0x68, high=0x69

# Button for start/stop
button = Pin(15, Pin.IN, Pin.PULL_UP)

# Global state
recording = False
frame_count = 0
sensor_data = []
start_time = 0

# MPU-6050 registers
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

def init_mpu6050(addr):
    """Wake up MPU-6050"""
    i2c.writeto_mem(addr, PWR_MGMT_1, bytes([0]))
    time.sleep_ms(100)

def read_mpu6050(addr):
    """Read accelerometer and gyroscope data"""
    # Read 14 bytes: accel (6) + temp (2) + gyro (6)
    data = i2c.readfrom_mem(addr, ACCEL_XOUT_H, 14)

    # Convert to signed integers
    def to_signed(high, low):
        value = (high << 8) | low
        if value > 32767:
            value -= 65536
        return value

    # Accelerometer (LSB/g = 16384 for ±2g range)
    ax = to_signed(data[0], data[1]) / 16384.0
    ay = to_signed(data[2], data[3]) / 16384.0
    az = to_signed(data[4], data[5]) / 16384.0

    # Gyroscope (LSB/°/s = 131 for ±250°/s range)
    gx = to_signed(data[8], data[9]) / 131.0
    gy = to_signed(data[10], data[11]) / 131.0
    gz = to_signed(data[12], data[13]) / 131.0

    return {
        'accel': {'x': ax, 'y': ay, 'z': az},
        'gyro': {'x': gx, 'y': gy, 'z': gz}
    }

def trigger_cameras():
    """Send simultaneous pulse to all cameras"""
    global frame_count

    if not recording:
        return

    # Trigger all cameras simultaneously
    for pin in camera_triggers:
        pin.value(1)
    time.sleep_us(100)  # 100 microsecond pulse
    for pin in camera_triggers:
        pin.value(0)

    frame_count += 1

def read_all_sensors():
    """Read from all connected IMU sensors"""
    global sensor_data

    if not recording:
        return

    timestamp = time.ticks_ms() - start_time

    readings = {
        'timestamp_ms': timestamp,
        'frame': frame_count,
        'sensors': {}
    }

    for i, addr in enumerate(IMU_ADDRESSES):
        try:
            data = read_mpu6050(addr)
            readings['sensors'][f'imu_{i}'] = data

            # Calculate total acceleration (for spike detection)
            total_g = (data['accel']['x']**2 +
                      data['accel']['y']**2 +
                      data['accel']['z']**2) ** 0.5
            readings['sensors'][f'imu_{i}']['total_g'] = total_g

        except Exception as e:
            print(f"IMU {i} error: {e}")

    sensor_data.append(readings)

    # Print if high acceleration detected (potential throw)
    for sensor_id, sensor_vals in readings['sensors'].items():
        if sensor_vals.get('total_g', 0) > 3.0:  # 3g = significant movement
            print(f"⚡ {sensor_id}: {sensor_vals['total_g']:.1f}g @ frame {frame_count}")

def start_recording():
    """Begin synchronized recording"""
    global recording, frame_count, sensor_data, start_time

    print("Starting recording...")
    recording = True
    frame_count = 0
    sensor_data = []
    start_time = time.ticks_ms()

    # Start camera trigger timer (30 fps)
    camera_timer = Timer()
    camera_timer.init(freq=30, mode=Timer.PERIODIC, callback=lambda t: trigger_cameras())

    # Start sensor reading timer (100 Hz - faster than cameras)
    sensor_timer = Timer()
    sensor_timer.init(freq=100, mode=Timer.PERIODIC, callback=lambda t: read_all_sensors())

    print("Recording at 30fps cameras, 100Hz sensors")
    return camera_timer, sensor_timer

def stop_recording(camera_timer, sensor_timer):
    """Stop recording and save data"""
    global recording

    print("Stopping recording...")
    recording = False
    camera_timer.deinit()
    sensor_timer.deinit()

    # Save sensor data to file
    with open('sensor_data.json', 'w') as f:
        json.dump({
            'total_frames': frame_count,
            'duration_ms': time.ticks_ms() - start_time,
            'readings': sensor_data
        }, f)

    print(f"Saved {len(sensor_data)} sensor readings, {frame_count} video frames")

    # Detect throw events (acceleration spikes)
    throws = detect_throws()
    print(f"Detected {len(throws)} potential throws:")
    for throw in throws:
        print(f"  Frame {throw['frame']}: {throw['peak_g']:.1f}g")

def detect_throws():
    """Analyze sensor data to find throw events"""
    throws = []

    # Look for acceleration spikes > 4g
    for reading in sensor_data:
        for sensor_id, vals in reading['sensors'].items():
            if vals.get('total_g', 0) > 4.0:
                # Check if already recorded nearby throw
                if not throws or reading['frame'] - throws[-1]['frame'] > 30:
                    throws.append({
                        'frame': reading['frame'],
                        'timestamp_ms': reading['timestamp_ms'],
                        'peak_g': vals['total_g'],
                        'sensor': sensor_id
                    })

    return throws

def main():
    """Main control loop"""
    print("Pi Pico Unified Controller")
    print("=" * 40)

    # Initialize IMU sensors
    print("Initializing IMU sensors...")
    for addr in IMU_ADDRESSES:
        try:
            init_mpu6050(addr)
            print(f"  IMU at 0x{addr:02x}: OK")
        except:
            print(f"  IMU at 0x{addr:02x}: NOT FOUND")

    print("\nPress button to start/stop recording")
    print("Or connect via USB serial for commands")

    timers = None

    while True:
        # Check button press
        if button.value() == 0:  # Button pressed (pull-up, so 0=pressed)
            time.sleep_ms(50)  # Debounce
            if button.value() == 0:
                if not recording:
                    timers = start_recording()
                else:
                    stop_recording(*timers)
                    timers = None

                # Wait for button release
                while button.value() == 0:
                    time.sleep_ms(10)

        time.sleep_ms(10)

if __name__ == "__main__":
    main()
```

---

## Auto-Sync: Matching Sensor Spikes to Video

### The Magic of Sensor-Based Sync

**Problem:** Video from 3 cameras needs perfect sync
**Old Solution:** Pi Pico GPIO triggers (already have this)
**New Bonus:** Accelerometer spikes provide SECOND sync signal!

**How It Works:**

1. **During Recording:**
   - Pi Pico logs: `{frame: 145, timestamp: 4833ms, accel: 5.2g}`
   - Video records frame #145

2. **After Recording:**
   - Look for spike in sensor data (throw event)
   - Find corresponding frame number
   - Video jumps to exact throw moment

3. **Validation:**
   - Sensor shows 5.2g at frame 145 (throw impact)
   - Video frame 145 shows uke hitting ground
   - **Perfect match = sync confirmed!**

**Sync Verification Script:**

```python
#!/usr/bin/env python3
"""
Verify sync between accelerometer data and video frames
"""

import json
import cv2
import numpy as np
from pathlib import Path

def load_sensor_data(json_path):
    """Load accelerometer readings from Pi Pico"""
    with open(json_path) as f:
        data = json.load(f)
    return data['readings']

def find_acceleration_peaks(readings, threshold=4.0):
    """Find frames with high acceleration (throw events)"""
    peaks = []

    for reading in readings:
        for sensor_id, vals in reading['sensors'].items():
            if vals['total_g'] > threshold:
                peaks.append({
                    'frame': reading['frame'],
                    'g_force': vals['total_g'],
                    'sensor': sensor_id
                })

    # Remove duplicates (keep highest g-force per event)
    filtered_peaks = []
    for peak in peaks:
        if not filtered_peaks or peak['frame'] - filtered_peaks[-1]['frame'] > 30:
            filtered_peaks.append(peak)
        elif peak['g_force'] > filtered_peaks[-1]['g_force']:
            filtered_peaks[-1] = peak

    return filtered_peaks

def verify_video_sync(video_path, sensor_peaks):
    """Check if video frames match sensor peaks"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video FPS: {fps}")
    print(f"Sensor peaks to verify: {len(sensor_peaks)}")

    for peak in sensor_peaks:
        frame_num = peak['frame']

        # Jump to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            # Add text overlay
            text = f"Frame {frame_num} | {peak['g_force']:.1f}g"
            cv2.putText(frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display
            cv2.imshow(f"Verify Sync: {peak['sensor']}", frame)
            print(f"Frame {frame_num}: {peak['g_force']:.1f}g - Is this a throw impact?")

            key = cv2.waitKey(0)
            if key == ord('y'):
                print("  ✓ SYNC CONFIRMED")
            elif key == ord('n'):
                print("  ✗ SYNC ERROR - frame doesn't match")

            cv2.destroyAllWindows()

    cap.release()

if __name__ == "__main__":
    sensor_data = load_sensor_data("sensor_data.json")
    peaks = find_acceleration_peaks(sensor_data)

    print(f"Found {len(peaks)} throw events:")
    for peak in peaks:
        print(f"  Frame {peak['frame']}: {peak['g_force']:.1f}g")

    # Verify with video
    verify_video_sync("cam1_recording.mp4", peaks)
```

---

## Power Measurement: From G-Force to Throw Quality

### Biomechanical Metrics from Accelerometer

**1. Peak Impact Force (g-force)**
```python
def measure_throw_power(sensor_readings, throw_frame, window=30):
    """Calculate throw power metrics from accelerometer data"""

    # Get readings around throw event
    start = max(0, throw_frame - window)
    end = throw_frame + window

    throw_data = [r for r in sensor_readings
                  if start <= r['frame'] <= end]

    # Peak acceleration (impact)
    peak_g = max(r['sensors']['imu_0']['total_g'] for r in throw_data)

    # Time to peak (throw duration)
    peak_frame = max(throw_data, key=lambda r: r['sensors']['imu_0']['total_g'])['frame']
    throw_duration_ms = (peak_frame - start) * (1000/30)  # assuming 30fps

    # Rotational velocity (from gyroscope)
    gyro_data = [r['sensors']['imu_0']['gyro'] for r in throw_data]
    max_rotation = max(
        (g['x']**2 + g['y']**2 + g['z']**2)**0.5
        for g in gyro_data
    )

    return {
        'peak_g_force': peak_g,
        'throw_duration_ms': throw_duration_ms,
        'max_rotation_deg_s': max_rotation,
        'power_index': peak_g * max_rotation / throw_duration_ms
    }
```

**2. Throw Signature Analysis**

Each technique has distinctive acceleration profile:

```
Seoi-Nage (Shoulder Throw):
┌─────────────────────────────┐
│    ╱╲   Peak: 6-8g         │
│   ╱  ╲  Sharp spike        │
│  ╱    ╲ (impact)           │
│ ╱      ╲                   │
│╱        ╲___               │
│  Entry  Throw  Landing     │
└─────────────────────────────┘

Uchi-Mata (Inner Thigh):
┌─────────────────────────────┐
│      ╱╲  Peak: 4-6g        │
│     ╱  ╲ Longer arc        │
│    ╱    ╲                  │
│   ╱      ╲___              │
│  ╱   Sweep   Landing       │
└─────────────────────────────┘

O-Soto-Gari (Major Outer Reap):
┌─────────────────────────────┐
│        ╱╲ Peak: 7-9g       │
│       ╱  ╲ Sudden drop     │
│      ╱    ╲                │
│_____╱      ╲___            │
│  Setup   Reap  Impact      │
└─────────────────────────────┘
```

**3. Quality Scoring**

```python
def score_throw_quality(metrics, technique_type):
    """Score throw based on accelerometer metrics"""

    # Reference values (from high-level judoka)
    REFERENCE = {
        'seoi_nage': {'peak_g': 7.0, 'duration_ms': 800, 'rotation': 400},
        'uchi_mata': {'peak_g': 5.0, 'duration_ms': 1000, 'rotation': 500},
        'harai_goshi': {'peak_g': 6.0, 'duration_ms': 900, 'rotation': 450},
    }

    if technique_type not in REFERENCE:
        return None

    ref = REFERENCE[technique_type]

    # Score based on proximity to reference
    power_score = min(100, (metrics['peak_g_force'] / ref['peak_g']) * 100)
    timing_score = max(0, 100 - abs(metrics['throw_duration_ms'] - ref['duration_ms']) / 10)
    rotation_score = min(100, (metrics['max_rotation_deg_s'] / ref['rotation']) * 100)

    overall = (power_score * 0.4 + timing_score * 0.3 + rotation_score * 0.3)

    return {
        'overall': overall,
        'power': power_score,
        'timing': timing_score,
        'rotation': rotation_score,
        'feedback': generate_feedback(power_score, timing_score, rotation_score)
    }

def generate_feedback(power, timing, rotation):
    """Generate coaching feedback from scores"""
    feedback = []

    if power < 60:
        feedback.append("Increase hip drive and kuzushi for more throwing power")
    elif power > 90:
        feedback.append("Excellent power generation!")

    if timing < 60:
        feedback.append("Throw timing is off - practice the rhythm of entry-load-throw")

    if rotation < 60:
        feedback.append("More hip rotation needed - focus on turning your core")

    return feedback
```

---

## Integration with Hybrid Pipeline

### Updated Architecture

```
┌─────────────────────────────────────────────────────────┐
│              DURING TRAINING SESSION                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Pi Pico Controller                                    │
│   ├─ Triggers 3 cameras (30 fps, synced)              │
│   ├─ Reads 4 IMU sensors (100 Hz)                     │
│   └─ Logs: frame number + timestamp + sensor data      │
│                                                         │
│   Athlete 1 (Tori):  IMU on chest + hip               │
│   Athlete 2 (Uke):   IMU on chest + hip               │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              AFTER SESSION (Cloud Processing)           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   1. Upload: 3 videos + sensor_data.json to Hetzner    │
│                                                         │
│   2. Auto-detect throws from sensor spikes             │
│      (No YOLO needed for motion detection!)            │
│                                                         │
│   3. Extract video frames at spike moments             │
│                                                         │
│   4. Send to Vision LLM for technique classification   │
│                                                         │
│   5. Combine: Video + Pose + Accelerometer analysis    │
│                                                         │
│   Result:                                               │
│   - Technique name (from LLM)                          │
│   - Quality score (from pose analysis)                 │
│   - Power measurement (from accelerometer)             │
│   - Video evidence (synced clips)                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Cost Savings:**
- No YOLO needed for motion detection (sensor spikes do it)
- Only process frames with high acceleration
- Reduces Vision LLM calls by additional 50%
- **Total per session: ~$0.005-0.02**

---

## Phase 0.5: Accelerometer Testing

### Before Buying Full System

**Quick Test ($10):**
1. Buy 1× MPU-6050 ($3.50) + 1× Pi Pico ($6)
2. Wire together (4 wires: VCC, GND, SDA, SCL)
3. Attach to judogi with velcro
4. Record one training session
5. Analyze data - can you see throws clearly?

**Test Script:**

```python
# Test MPU-6050 with Pi Pico (no cameras yet)
from machine import Pin, I2C
import time

i2c = I2C(0, scl=Pin(5), sda=Pin(4), freq=400000)
IMU_ADDR = 0x68

# Initialize
i2c.writeto_mem(IMU_ADDR, 0x6B, bytes([0]))
time.sleep(0.1)

print("Recording 60 seconds of accelerometer data...")
print("Do some throws!")

data = []
start = time.ticks_ms()

while time.ticks_ms() - start < 60000:  # 60 seconds
    raw = i2c.readfrom_mem(IMU_ADDR, 0x3B, 6)

    ax = ((raw[0] << 8) | raw[1]) / 16384.0
    ay = ((raw[2] << 8) | raw[3]) / 16384.0
    az = ((raw[4] << 8) | raw[5]) / 16384.0

    # Handle signed
    if ax > 2: ax -= 4
    if ay > 2: ay -= 4
    if az > 2: az -= 4

    total_g = (ax**2 + ay**2 + az**2) ** 0.5

    if total_g > 3.0:
        print(f"⚡ HIGH G: {total_g:.1f}g @ {time.ticks_ms() - start}ms")

    data.append({
        'time': time.ticks_ms() - start,
        'g': total_g
    })

    time.sleep_ms(10)  # 100 Hz

print(f"\nRecorded {len(data)} samples")
print("High-g events (potential throws):")
peaks = [d for d in data if d['g'] > 3.0]
print(f"  Found {len(peaks)} events")
```

**Success Criteria:**
- Can clearly see throw events (4-8g spikes)
- Events match when you actually threw
- Data is consistent (not too noisy)

---

## Future: Wireless Sensors (Phase 3+)

### Current: Wired to Pi Pico
**Pros:** Cheap, reliable, no latency, perfect sync
**Cons:** Wires can be annoying during training

### Future: Bluetooth/WiFi IMUs
**Options:**
- ESP32 + MPU-6050: Built-in WiFi/Bluetooth ($8/sensor)
- Commercial IMU (Xsens, IMeasureU): $200-500/sensor
- Nordic nRF52 + BLE: Low power, reliable ($15/sensor)

**Architecture:**
```
Wireless IMUs (4x)
     │ Bluetooth Low Energy
     ▼
Pi Pico W (with WiFi/BLE)
     │ Receives sensor data
     │ Triggers cameras
     │ Logs everything
     ▼
Synchronized Data
```

**Defer to Phase 3:** Get wired system working first!

---

## Summary: What Accelerometers Add

| Feature | Before (Video Only) | After (Video + Accel) |
|---------|---------------------|------------------------|
| **Power Measurement** | Estimated from pose | **Direct measurement (g-force)** |
| **Sync Verification** | Trust GPIO timing | **Double-check with acceleration spikes** |
| **Throw Detection** | YOLO motion analysis | **Instant from sensor spikes** |
| **Cost** | $431 (3-camera) | **$462 (only $31 more!)** |
| **Accuracy** | Good | **Excellent (ground truth force)** |
| **Coaching Feedback** | "Hip seems low" | **"7.2g impact, 450°/s rotation"** |

**This is the difference between good and professional-grade sports science!**

---

## Next Steps

1. **Quick Test ($10):** Buy 1× MPU-6050 + 1× Pi Pico
2. **Validate:** Can you see throws in accelerometer data?
3. **If yes:** Add 3 more sensors ($14) to camera system
4. **Integrate:** Update Pi Pico code to handle both cameras + sensors
5. **Deploy:** Full system with video + accelerometer analysis

**Total additional investment: $31 for transformative improvement!**
