# LionOfJudo v1 Field-Test Protocol

This protocol is for the first bench, mattress, and camera sessions. It is not
approval to use the device in unrestricted randori before it has passed the
physical checks.

## Before Each Test

1. Inspect the LiPo pouch: no dents, swelling, creases, exposed foil, or loose
   leads. Retire a suspect cell.
2. Confirm the GY-521 is rigidly fixed, all solder joints are heat-shrunk, and
   the battery cannot touch solder joints or move in the box.
3. With the switch off, use the multimeter to confirm no short between `3V3`
   and GND or between `5V/VIN` and GND.
4. With the switch on, confirm the ESP32 `3V3` rail is 3.25-3.35 V while
   logging and GPIO4 is roughly half the cell voltage. Repeat near a 3.3 V
   cell voltage; a falling or brownout-prone 3V3 rail fails the build.
5. Keep `JudoNet` off before the training boot. Start Sony and iPhone in
   landscape at 1080p60 with audio enabled. Sony is the master angle.

## Sync And Recording

1. Switch on both wearables and wait for their 10-second amber countdown to
   finish. They are not logging during the countdown.
2. Keep the athlete still for 5 seconds. This is the baseline-quality window.
3. In view and within earshot of both cameras, perform three strong,
   deliberately uneven physical heel-drops or direct case taps, about 0.8 s
   then 1.4 s apart. Each event must shake both mounted units and be audible.
4. Run the bench/mattress test. For the first real clothing test, do controlled
   movements only; stop if a case shifts, presses into the body, heats, or has
   any sign of battery damage.
5. End with the same three physical spikes, then stay still for 10 seconds.
   Do not substitute off-body claps: they do not produce an end anchor in the
   IMU data.

## Collection And Acceptance

1. Power the units off. Turn on the Pi and `JudoNet`; power the units back on
   and wait for their slow green download mode.
2. On the Pi, collect without wiping on the first few tests:

   ```bash
   python tools/rpi_collect_imu.py --root /data/lionofjudo --session 2026-07-12
   ```

3. Preflight the copied logs before any video analysis:

   ```bash
   python tools/imu_preflight.py /data/lionofjudo/sessions/2026-07-12/imu/
   ```

4. Accept a test only when both units have the correct distinct identities,
   200 Hz timing close to 5 ms, no accelerometer clipping, an identified
   start ritual, and plausible initial stillness. Inspect with `imu_plot.py`
   if any check looks wrong.
5. Copy the session folder to the Mac, then run `pipeline.run_session`. The
   session report now includes IMU quality data and frame-aligned
   `gi_biomechanics.csv` for each throw.

## Charging

Disconnect the battery from the wearable and connect it to the external
charger on a non-flammable surface. Use the LiPo protection bags for charging
and storage, not inside the gi. Do not use the TP4056 at its usual 1 A setting
unless the cell datasheet explicitly permits it.
