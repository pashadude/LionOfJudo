# LionOfJudo v1 Design Feedback

Use this as the correction brief for the next device and architecture image.
It reflects the hardware actually bought and the software that will read it.

## Must Change

1. Draw **one protected 3.7 V LiPo per wearable**, not two batteries and not
   a separate "protected switch." The switch is simply inline on battery +.
2. Draw the battery power path exactly: `LiPo + -> switch -> ESP32
   5V/VIN/VBUS`; `LiPo - -> common GND`. Never draw the LiPo going to `3V3`.
3. Draw the TP4056 **outside** the wearable. Charging means unplugging the
   battery JST connector from the wearable and plugging it into the external
   charger. No USB-C cutout belongs on the v1 capsule.
4. Do not label a generic TP4056 as a 1 A charger for a 250 mAh pouch cell.
   Its current must first be set to the cell manufacturer's permitted value.
5. Draw `100k from battery+ -> GPIO4 -> 100k -> GND`, plus `100 nF from GPIO4
   to GND`. The midpoint is approximately half the battery voltage. The shown
   shopping list has the 100k resistors but does **not** show a 100 nF capacitor;
   add it to the parts list before assembly.
6. Show heat-shrink only over solder joints and wire splices. It is electrical
   insulation, not impact padding.
7. Show the 3 mm tattoo-silicone material as a **60 x 40 mm body-facing
   impact-spreader pad in the sewn pocket**. Inside the enclosure, place small
   silicone pieces at the battery and cover to stop rattle and prevent solder
   contact. Do not mount the IMU in soft silicone.
8. The GY-521 must be rigidly fixed flat to the case floor. Add an axis mark
   on the outside of both enclosures: `+Y -> head`, `+X -> athlete's right`.
   Both units must use the same orientation. Without this, axis-level chest /
   hip coupling is not interpretable.
9. Use the actually bought **rectangular 64 x 44 x 20 mm box** as the candidate
   v1 enclosure. It is not yet a confirmed fit. Do not draw a 50 x 26 mm
   cylinder or claim final dimensions until a physical dry-fit is photographed
   with a ruler.
10. Draw one physical sync ritual at both start and end: after logging begins,
    5 s still, then three audible heel-drops / mounted-unit impacts. "Three
    claps" alone is wrong: claps provide audio but no spike in either IMU.

## Language That Must Be Accurate

- Call the output **gi inertial measurements** or **frame-aligned gi
  biomechanics**.
- `g`, angular velocity, timing, and a repeatable load index are measured.
- Do not call the result direct impact force, physical power in watts, energy,
  or a medical biomechanical assessment. A loose gi pocket measures garment
  motion relative to the athlete as well as body motion.
- For actual body-axis biomechanics, a future version needs a tight,
  repeatable body mount and calibration. V1 is suitable for relative,
  within-athlete comparisons after the mounting protocol is kept fixed.

## Architecture Labels

- Training: `JudoNet OFF -> wearable boots -> 10 s amber countdown -> local
  flash logging at 200 Hz`.
- Collection: `JudoNet ON -> reboot wearable -> slow green download mode ->
  Raspberry Pi pulls verified files by HTTP -> Pi stores session folder -> Mac
  copies session folder and runs final video/IMU synchronization`.
- The Pi is storage and lightweight quality checking. The Mac remains the
  final video synchronizer and analysis computer.
- Sony is the master video. iPhone is the second angle. Both record landscape
  at 1080p60 with sound enabled; avoid iPhone stabilization/format changes
  during the session.

## Images Still Needed Before Build Approval

1. ESP32-S3 front and back in sharp focus, with every silkscreened pin label.
2. GY-521 front and back, especially its pin row and printed axes.
3. TP4056 front and back, close enough to read its programming-resistor label.
4. LiPo label and its JST connector, next to a ruler.
5. Empty case open beside a ruler, and a dry layout of ESP32 + GY-521 + LiPo
   inside it.
6. Raspberry Pi top side, showing the model printed on the PCB and whether it
   has Ethernet. This decides 32/64-bit image and the network drawing.
