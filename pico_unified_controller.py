"""
Pi Pico: Unified Camera Sync + Accelerometer Logger

This runs on Raspberry Pi Pico with MicroPython to:
1. Trigger 3 synchronized cameras at 30fps
2. Read 4 IMU sensors at 100Hz (accelerometer + gyroscope)
3. Log all data with frame-accurate timestamps
4. Auto-detect throws from g-force spikes

Hardware:
- GPIO 0, 1, 2: Camera trigger outputs
- GPIO 4 (SDA), 5 (SCL): I2C bus for IMU sensors
- GPIO 15: Start/stop button
- USB: Serial output for monitoring

IMU Setup:
- IMU 0 (Tori chest): Address 0x68 (AD0 = LOW)
- IMU 1 (Tori hip):   Address 0x69 (AD0 = HIGH)
- IMU 2 (Uke chest):  Second I2C bus or multiplexer
- IMU 3 (Uke hip):    Second I2C bus or multiplexer

For 2 IMUs on same bus: Connect AD0 pin to GND (0x68) or 3.3V (0x69)

Upload to Pi Pico using Thonny IDE or ampy.
"""

from machine import Pin, I2C, Timer, UART
import time
import json
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================

# Camera triggers
CAMERA_PINS = [0, 1, 2]  # GPIO pins
CAMERA_FPS = 30          # Frames per second

# IMU sensors (MPU-6050)
IMU_SDA = 4
IMU_SCL = 5
IMU_FREQ = 400000  # I2C speed (400kHz fast mode)
IMU_ADDRESSES = [0x68, 0x69]  # Two sensors per bus
SENSOR_HZ = 100    # Sensor sampling rate

# Button
BUTTON_PIN = 15

# Thresholds
THROW_THRESHOLD_G = 3.5  # g-force to consider as significant movement
IMPACT_THRESHOLD_G = 5.0  # g-force to consider as throw impact

# MPU-6050 registers
MPU6050_PWR_MGMT_1 = 0x6B
MPU6050_ACCEL_XOUT_H = 0x3B
MPU6050_ACCEL_CONFIG = 0x1C
MPU6050_GYRO_CONFIG = 0x1B

# ============================================================================
# GLOBAL STATE
# ============================================================================

camera_triggers = []
i2c = None
button = None

recording = False
frame_count = 0
start_time = 0

# Use circular buffer for sensor data (memory efficient)
MAX_READINGS = 10000  # ~100 seconds at 100Hz
sensor_buffer = []
buffer_index = 0

# Throw detection
throw_events = []

# Timers
camera_timer = None
sensor_timer = None

# ============================================================================
# IMU FUNCTIONS
# ============================================================================

def init_hardware():
    """Initialize all hardware"""
    global camera_triggers, i2c, button

    print("Initializing hardware...")

    # Camera triggers
    camera_triggers = [Pin(pin, Pin.OUT) for pin in CAMERA_PINS]
    for trigger in camera_triggers:
        trigger.value(0)  # Start LOW
    print(f"  Cameras: {len(camera_triggers)} triggers on GPIO {CAMERA_PINS}")

    # I2C bus
    i2c = I2C(0, scl=Pin(IMU_SCL), sda=Pin(IMU_SDA), freq=IMU_FREQ)
    devices = i2c.scan()
    print(f"  I2C devices found: {[hex(d) for d in devices]}")

    # Button
    button = Pin(BUTTON_PIN, Pin.IN, Pin.PULL_UP)
    print(f"  Button on GPIO {BUTTON_PIN}")

    # Initialize IMUs
    for addr in IMU_ADDRESSES:
        if addr in devices:
            init_mpu6050(addr)
            print(f"  IMU 0x{addr:02x}: Initialized")
        else:
            print(f"  IMU 0x{addr:02x}: NOT FOUND")

    print("Hardware initialization complete!\n")

def init_mpu6050(addr):
    """Wake up and configure MPU-6050"""
    # Wake up from sleep mode
    i2c.writeto_mem(addr, MPU6050_PWR_MGMT_1, bytes([0x00]))
    time.sleep_ms(10)

    # Set accelerometer range to ±8g (for high-impact throws)
    # 0x00 = ±2g, 0x08 = ±4g, 0x10 = ±8g, 0x18 = ±16g
    i2c.writeto_mem(addr, MPU6050_ACCEL_CONFIG, bytes([0x10]))

    # Set gyroscope range to ±500°/s
    # 0x00 = ±250°/s, 0x08 = ±500°/s, 0x10 = ±1000°/s
    i2c.writeto_mem(addr, MPU6050_GYRO_CONFIG, bytes([0x08]))

def read_mpu6050(addr):
    """
    Read accelerometer and gyroscope data from MPU-6050

    Returns:
        dict with 'accel' (x,y,z in g) and 'gyro' (x,y,z in °/s)
    """
    try:
        # Read 14 bytes: accel(6) + temp(2) + gyro(6)
        data = i2c.readfrom_mem(addr, MPU6050_ACCEL_XOUT_H, 14)

        # Convert to signed 16-bit integers
        def to_signed16(high, low):
            value = (high << 8) | low
            if value > 32767:
                value -= 65536
            return value

        # Accelerometer: ±8g range → LSB/g = 4096
        ax = to_signed16(data[0], data[1]) / 4096.0
        ay = to_signed16(data[2], data[3]) / 4096.0
        az = to_signed16(data[4], data[5]) / 4096.0

        # Gyroscope: ±500°/s range → LSB/(°/s) = 65.5
        gx = to_signed16(data[8], data[9]) / 65.5
        gy = to_signed16(data[10], data[11]) / 65.5
        gz = to_signed16(data[12], data[13]) / 65.5

        # Calculate total acceleration (magnitude)
        total_g = (ax*ax + ay*ay + az*az) ** 0.5

        return {
            'ax': ax, 'ay': ay, 'az': az,
            'gx': gx, 'gy': gy, 'gz': gz,
            'total_g': total_g
        }
    except Exception as e:
        return None

# ============================================================================
# RECORDING FUNCTIONS
# ============================================================================

def trigger_cameras(timer):
    """ISR: Trigger all cameras simultaneously (called at 30fps)"""
    global frame_count

    if not recording:
        return

    # Set all triggers HIGH
    for trigger in camera_triggers:
        trigger.value(1)

    # Brief delay (cameras need ~10μs pulse)
    time.sleep_us(50)

    # Set all triggers LOW
    for trigger in camera_triggers:
        trigger.value(0)

    frame_count += 1

def read_sensors(timer):
    """ISR: Read all IMU sensors (called at 100Hz)"""
    global sensor_buffer, buffer_index

    if not recording:
        return

    timestamp = time.ticks_ms() - start_time

    # Create reading
    reading = {
        't': timestamp,  # Short keys to save memory
        'f': frame_count,
        's': {}
    }

    # Read each IMU
    for i, addr in enumerate(IMU_ADDRESSES):
        data = read_mpu6050(addr)
        if data:
            reading['s'][i] = data

            # Check for throw event
            if data['total_g'] > IMPACT_THRESHOLD_G:
                # Record throw event
                throw_events.append({
                    'frame': frame_count,
                    'time_ms': timestamp,
                    'g': data['total_g'],
                    'sensor': i
                })
                # Print notification
                print(f"⚡ THROW! Frame {frame_count}: {data['total_g']:.1f}g (sensor {i})")

    # Add to buffer (circular)
    if len(sensor_buffer) < MAX_READINGS:
        sensor_buffer.append(reading)
    else:
        sensor_buffer[buffer_index] = reading
        buffer_index = (buffer_index + 1) % MAX_READINGS

def start_recording():
    """Start synchronized recording"""
    global recording, frame_count, start_time, sensor_buffer, buffer_index
    global camera_timer, sensor_timer, throw_events

    print("\n" + "="*50)
    print("STARTING RECORDING")
    print("="*50)

    # Reset state
    recording = True
    frame_count = 0
    start_time = time.ticks_ms()
    sensor_buffer = []
    buffer_index = 0
    throw_events = []

    # Force garbage collection
    gc.collect()
    print(f"Free memory: {gc.mem_free()} bytes")

    # Start camera timer (30 fps = every 33.33ms)
    camera_timer = Timer()
    camera_timer.init(freq=CAMERA_FPS, mode=Timer.PERIODIC, callback=trigger_cameras)
    print(f"Camera triggers: {CAMERA_FPS} fps")

    # Start sensor timer (100 Hz = every 10ms)
    sensor_timer = Timer()
    sensor_timer.init(freq=SENSOR_HZ, mode=Timer.PERIODIC, callback=read_sensors)
    print(f"Sensor sampling: {SENSOR_HZ} Hz")

    print("\nRecording... Press button to stop.\n")

def stop_recording():
    """Stop recording and save data"""
    global recording, camera_timer, sensor_timer

    recording = False

    # Stop timers
    if camera_timer:
        camera_timer.deinit()
    if sensor_timer:
        sensor_timer.deinit()

    duration_sec = (time.ticks_ms() - start_time) / 1000.0

    print("\n" + "="*50)
    print("RECORDING STOPPED")
    print("="*50)
    print(f"Duration: {duration_sec:.1f} seconds")
    print(f"Frames captured: {frame_count}")
    print(f"Sensor readings: {len(sensor_buffer)}")
    print(f"Throw events detected: {len(throw_events)}")

    # List throws
    if throw_events:
        print("\nThrow Events:")
        for i, event in enumerate(throw_events):
            print(f"  {i+1}. Frame {event['frame']} ({event['time_ms']/1000:.2f}s): "
                  f"{event['g']:.1f}g (sensor {event['sensor']})")

    # Save data
    save_data(duration_sec)

def save_data(duration_sec):
    """Save sensor data to file"""
    print("\nSaving data to sensor_data.json...")

    # Create summary
    output = {
        'metadata': {
            'duration_sec': duration_sec,
            'total_frames': frame_count,
            'sensor_readings': len(sensor_buffer),
            'throw_events': len(throw_events),
            'camera_fps': CAMERA_FPS,
            'sensor_hz': SENSOR_HZ
        },
        'throws': throw_events,
        # Don't save all readings - too much data
        # Instead, save summary statistics
    }

    # Save compact JSON
    try:
        with open('sensor_summary.json', 'w') as f:
            json.dump(output, f)
        print("✓ Saved sensor_summary.json")

        # For full data, write in chunks
        print("Saving full sensor data...")
        with open('sensor_full.json', 'w') as f:
            f.write('{"readings":[\n')
            for i, reading in enumerate(sensor_buffer):
                if i > 0:
                    f.write(',\n')
                f.write(json.dumps(reading))
                if i % 1000 == 0:
                    print(f"  {i}/{len(sensor_buffer)}...")
            f.write('\n]}')
        print(f"✓ Saved sensor_full.json ({len(sensor_buffer)} readings)")

    except Exception as e:
        print(f"✗ Error saving: {e}")

    gc.collect()
    print(f"Free memory after save: {gc.mem_free()} bytes")

# ============================================================================
# MAIN LOOP
# ============================================================================

def monitor_status():
    """Print status during recording"""
    if recording:
        elapsed = (time.ticks_ms() - start_time) / 1000.0
        print(f"Recording: {elapsed:.1f}s | Frames: {frame_count} | "
              f"Readings: {len(sensor_buffer)} | Throws: {len(throw_events)}")

def main():
    """Main control loop"""
    print("\n" + "="*50)
    print("  PI PICO UNIFIED CONTROLLER")
    print("  Camera Sync + Accelerometer Logger")
    print("="*50)

    # Initialize hardware
    init_hardware()

    print("Controls:")
    print("  - Press button to START/STOP recording")
    print("  - Data saved to sensor_summary.json and sensor_full.json")
    print("  - Files can be read via USB mass storage or serial")
    print("\nReady! Waiting for button press...\n")

    last_status_time = 0

    while True:
        # Check button
        if button.value() == 0:  # Pressed (pull-up)
            time.sleep_ms(50)  # Debounce

            if button.value() == 0:  # Still pressed
                if not recording:
                    start_recording()
                else:
                    stop_recording()

                # Wait for release
                while button.value() == 0:
                    time.sleep_ms(10)

        # Periodic status update
        if recording and time.ticks_ms() - last_status_time > 5000:
            monitor_status()
            last_status_time = time.ticks_ms()

        time.sleep_ms(50)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        if recording:
            stop_recording()
    except Exception as e:
        print(f"\nError: {e}")
        if recording:
            stop_recording()
