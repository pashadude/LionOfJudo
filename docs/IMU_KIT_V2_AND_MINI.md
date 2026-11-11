# IMU kit: updates to the current build, and a smaller "mini" build

*Written 2026-11-11 from the assembly guide `Lav_assembly_RU.pdf` (01.11.2026),
`firmware/lionimu/` and `docs/IMU_KIT_DESIGN_BRIEF.md`.*

**What this is based on, and what it is not.** I read the 39-page assembly
guide and the firmware. I could not watch the video (a `studio.youtube.com`
link only opens for the channel owner, and YouTube is not reachable from this
session), and I could not open Temu or Seeed's documentation. So:

- Anything about your physical build comes from the photos and text in the PDF.
- Part numbers, pin maps and prices below are from memory of the common
  versions of these boards. Each one that matters is marked **verify** with the
  check to do before soldering or buying.
- Nothing here has been built or compiled.

---

## Part 1. Updates to the current build

The current build, as the guide shows it: a long ESP32-S3 dev board with pin
headers, a GY-521 (MPU6050) on four flying wires, two LiPo pouches wired in
parallel behind one white connector, a harness with a rocker POWER switch and
a resistor branch, battery fed into `5VIN`, optional STOP switch on GPIO7.

### 1.1 Power (highest priority)

| # | Change | Why |
|---|--------|-----|
| P1 | **Keep the two cells in parallel, but build them as one permanent pack** (updated 2026-11-11: one cell was tried and was not enough). Same model and batch; measure both before joining and join only if they are within about 0.05 V; never separate them again; charge them together as one pack. Set the external charger for the combined capacity (for 2 × 250 mAh, Rprog 5 kΩ ≈ 250 mA is right). | Cells joined at different voltages push current into each other. The likely reason one cell fails is peak current, not capacity: the firmware tries WiFi for 8 s at every boot, which draws short bursts of several hundred mA, and a small protected pouch sags under that, on top of the regulator drop in P3. Two cells halve the sag. The alternative with the same effect is **one larger cell** (about 500–1000 mAh, rated for at least 1 A), which is simpler to charge and protect. |
| P1b | **Add a 470–1000 µF low-ESR capacitor across `5VIN` and GND**, close to the board. | Covers the WiFi bursts and may make a single cell sufficient. Test: one cell plus capacitor, ten cold boots, watch for resets. |
| P2 | **Confirm the cell has a protection board**, and write the result into the checklist. | The guide's step 17 leaves this as "установи защиту пакета по документации". A protected pouch has a small PCB under yellow tape at the tab end. No PCB visible means treat it as unprotected and do not put it on a child. |
| P3 | **Check which 3.3 V regulator your dev board has before trusting battery power.** | The battery goes into `5VIN`, so it passes through the board's regulator. Small boards usually carry a low-dropout part (ME6211 class, roughly 0.1–0.3 V drop). Many long DevKit clones carry an AMS1117, which needs about 1 V of headroom: a 3.7 V cell would give a 3V3 rail near 2.7 V and random resets. **Verify:** read the marking on the 3-pin/5-pin regulator near the USB port, and treat step 17's "3V3 = 3.2–3.4 V on battery" as a pass/fail gate, measured again with the cell near 3.5 V, not only when full. |
| P4 | **Raise `BATT_LOW_V` from 3.30 to about 3.45 V.** | At a 3.30 V cell the regulator can no longer hold 3.3 V, so the cutoff fires after the rail has already sagged. Stopping earlier costs a few minutes and protects the file close. |
| P5 | **Switch in the positive lead, divider top on the switched side.** | Then OFF removes power from everything including the divider, and the "which wire does POWER break" uncertainty in the guide disappears for the next unit. |
| P6 | **Add a small series fuse or polyfuse (about 500 mA) at the battery connector.** | Cheap protection against a harness short inside a closed box. |

### 1.2 Sensor and mechanics

| # | Change | Why |
|---|--------|-----|
| M1 | **Remove flying wires between ESP32 and GY-521.** Solder both to one small piece of perfboard, or stack the GY-521 on four short pins. | Wire fatigue at the pad is the most likely field failure, and wire movement shows up as noise in the data. |
| M2 | **Fix the perfboard to the case floor with two M2 nylon screws or standoffs**, not foam or hot glue. | The IMU has to move with the case. The guide already says this; perfboard makes it easy. |
| M3 | **Desolder nothing on the dev board for compactness**; if the box is too tight, change the board (Part 2), not the headers. | Agrees with the guide's step 18. |
| M4 | **Mark the axes and "up" on the outside of the case.** | Step 20 asks for the same orientation every session. |
| M5 | **Plan for accelerometer clipping.** The MPU6050 tops out at ±16 g and a hard landing can exceed it. | `imu_preflight.py` already reports clipping. If it appears regularly, the fix is a different sensor (Part 2, option S2), not a software change. |

### 1.3 Firmware (read from `firmware/lionimu/lionimu.ino`)

These are observations from reading the code, not from running it.

| # | Finding | Suggested change |
|---|---------|------------------|
| F1 | `logSample()` prints `AccelX/Y/Z` over Serial for **every** sample (200 lines per second). | Guard it with a debug flag. It costs time in the sampling loop for no benefit on battery. |
| F2 | `lionimu.ino` includes `secrets.h` unconditionally, while `config.h` handles its absence with `__has_include`. | Drop the include in the `.ino`; a clean checkout does not compile today. |
| F3 | `LittleFS.begin(true, …)` formats the partition if mounting fails. | Pass `false` and show an error colour instead. A transient mount failure should not erase recorded sessions. |
| F4 | No free-space check; `logFile.write()` results are ignored. | Stop and close cleanly when free space drops below about 100 kB, and on a short write. |
| F5 | The STOP button and the 5-minute trial limit exist only in `recording-trial.patch`, which is not in the repository. | Commit the patch (or merge it behind a config flag) so the guide's Appendix A works from a fresh clone. |
| F6 | Each record is 16 bytes (4-byte ms timestamp plus 12 data bytes), so 200 Hz writes 11.5 MB per hour. | Optional: write the timestamp once per 4096-byte block. That is about 25% more recording time on the same flash. Needs a format-version bump and a matching change in `tools/`. |
| F7 | CPU runs at the default clock while logging with WiFi off. | Optional: `setCpuFrequencyMhz(80)` in logging mode to stretch battery life. Check that the 5 ms sample interval still holds afterwards. |

### 1.4 Documentation mismatch to resolve

`config.h` says "Pins (ESP32-S3 SuperMini)" and the brief draws a 22 × 18 mm
SuperMini, but the brief also rejects 4 MB boards, and the common SuperMini is
a 4 MB part. The unit in the guide is a long dev board. Pick one board per
variant and record its flash size, regulator and LED pin in a small table in
`config.h`; Part 2 proposes the values for the mini.

---

## Part 2. The mini build

### 2.1 Goal

Same firmware, same log format, same tools. Roughly a third of the current
volume, one cell, no external charger, no harness.

### 2.2 Recommended parts

| # | Part | Size | Notes |
|---|------|------|-------|
| 1 | **XIAO ESP32-S3 Plus** (16 MB flash) | about 21 × 18 mm | Has a LiPo charger and battery pads on the board, so the cell connects directly and charges over USB-C. **Verify** on the listing: "Plus", 16 MB flash, and that it is the XIAO form factor with `BAT+`/`BAT-` pads on the back. |
| 2 | **GY-521 (MPU6050)** | 21 × 16 mm | Same as now, so no firmware change. Do not fit the header strip. |
| 3 | **One protected 1S LiPo of about 500–600 mAh** (for example 602030 or 552535), rated for at least 1 A | about 30 × 20 × 6 mm | Sized up because one 250 mAh cell was not enough on the current build. Must show a protection PCB. Loose leads solder to the XIAO pads. Add the 470 µF capacitor from P1b across `BAT+`/`BAT−`. Whether one cell is enough on this board is untested; if it is not, use the same permanent two-cell pack as in P1. |
| 4 | **Slide switch SS12D00** or similar, 3-pin | about 9 × 4 mm | In the battery positive lead. |
| 5 | 2 × 100 kΩ resistors, 1 × 100 nF capacitor | — | Battery divider, same as now. |
| 6 | Polyfuse about 500 mA | — | In series with battery positive. |
| 7 | Perfboard offcut about 22 × 40 mm, 30 AWG silicone wire, kapton tape, heat-shrink | — | Carrier for the two boards. |
| 8 | Small rigid plastic box, inner size about 45 × 25 × 12 mm | — | Dry-fit before choosing; see 2.5. |
| 9 | Tactile button (optional) | 6 × 6 mm | STOP on GPIO7. |

**Why the XIAO and not a SuperMini.** The common ESP32-S3 SuperMini carries
4 MB of flash, which your own brief rejects as too short, and it has no
charger. The standard (non-Plus) XIAO ESP32-S3 has 8 MB, which is roughly
25–30 minutes at the current record size. The Plus keeps the 16 MB you have
now. If a seller offers a 16 MB SuperMini-format board, it works too, but then
keep the external charger and unplug-to-charge routine from the current build.

**Temu search terms** (I could not check live listings or prices):
`XIAO ESP32S3 Plus 16MB`, `GY-521 MPU6050 module`, `3.7V lipo 502030 protected`
(or `402030`), `SS12D00 slide switch`, `PPTC polyfuse 500mA`,
`double sided perfboard 2x8cm`, `30AWG silicone wire`, `M2 nylon standoff kit`.
Cheap listings often mislabel flash size and sell cells without protection, so
both need the checks in 2.6 on arrival. If Temu has no XIAO Plus, AliExpress or
Seeed's own store does.

### 2.3 Wiring

```
LiPo +  ── polyfuse ── slide switch ──┬── XIAO BAT+
                                      └── 100k ──┬── GPIO4 (D3)
                                                 ├── 100k ── GND
                                                 └── 100nF ── GND
LiPo −  ─────────────────────────────── XIAO BAT−  (= GND)

XIAO 3V3      ── GY-521 VCC
XIAO GND      ── GY-521 GND
XIAO GPIO8    ── GY-521 SDA
XIAO GPIO9    ── GY-521 SCL
XIAO GPIO7    ── STOP button ── GND      (optional)
```

On the XIAO ESP32-S3 the pads I expect are `D3 = GPIO4`, `D8 = GPIO7`,
`D9 = GPIO8`, `D10 = GPIO9`, which would keep every pin number in `config.h`
unchanged. **Verify** against the pinout card in the box before soldering; the
silk shows `D` numbers, not GPIO numbers.

Differences from the current build:

- The battery goes to the `BAT` pads, **not** to `5V`. The board's own power
  path handles it, which removes the regulator question in P3.
- Charging is over USB-C with the switch ON (the charger sits behind the
  switch in this wiring). The cell stays in the box.
- The divider is still required: the board does not report battery voltage by
  itself.

### 2.4 Firmware changes for the mini

| Item | Current | Mini |
|------|---------|------|
| Status LED | WS2812 on GPIO48 (`Adafruit_NeoPixel`) | Single-colour LED, expected on GPIO21, active low. **Verify.** `ledColor()` needs a second implementation that maps "any colour" to on and "black" to off; the colour language (amber, blue, purple, red) becomes blink patterns. |
| Board profile | `esp32s3`, `FlashSize=16M`, `app3M_fat9M_16MB` | XIAO ESP32S3 board entry with a 16 MB partition scheme that has a partition labelled `ffat`. The firmware mounts by that label. **Verify** with `flash-id` first, as in Appendix A2. |
| Pins | SDA 8, SCL 9, ADC 4, STOP 7 | Same, if the pinout check passes. |
| `BENCH_USB_POWER` | Needed for USB-only tests | Still needed when no cell is fitted. With a cell fitted and USB plugged in, the divider reads the charging cell, so the cutoff behaves normally. |

I have not written this LED variant, because I cannot compile or test it
here. It is a small change, and I can add it as a `BOARD_XIAO_S3` option in
`config.h` if you want it prepared before the parts arrive.

### 2.5 Layout

```
side view (not to scale)

  ┌───────────────────────────────────────────┐  lid
  │  [ XIAO ]──perfboard──[ GY-521 ]          │  boards, parts facing up
  │  ═══════════════════════════════════      │  perfboard on 2 nylon standoffs
  │  ─────────── kapton / thin plastic ────── │  insulating divider
  │  [        LiPo, lying flat          ]     │  not compressed by the lid
  └───────────────────────────────────────────┘  floor → silicone pad → body
        USB-C at one short end        switch at the other
```

- Footprint is set by the perfboard, about 22 × 40 mm, with the cell
  underneath. Expected outer size is around 48 × 28 × 14 mm, against the
  64 × 44 × 20 mm candidate in the brief. This is an estimate until a dry fit.
- Trim every solder tail on the underside flush and cover the underside with
  kapton before the cell goes in. Nothing sharp may face the pouch.
- Keep the XIAO's antenna end at the edge of the box, away from the cell.
- One USB-C cutout and one LED window. The switch sits recessed so a grip
  cannot flip it.

### 2.6 Bring-up order

The guide's method carries over. Only the order of the battery steps changes.

1. Parts on arrival: run `flash-id` on the XIAO and note the flash size. Look
   for the protection PCB on each cell and measure its voltage (3.6–4.2 V).
2. Guide steps 2–14 unchanged (USB only, no cell): sensor wires, continuity,
   3V3, two test files.
3. Build the divider on the perfboard and check it with a bench supply or the
   cell on clip leads before joining it to GPIO4 (guide step 16).
4. With USB unplugged and the switch OFF, solder polyfuse, switch and cell.
   Solder the cell leads one at a time with the other one insulated.
5. Switch ON, no USB: measure `3V3` and sensor `VCC` (3.2–3.4 V).
6. Plug USB: the charge LED should light and go out when full. Feel the cell
   during the first charge; warm is a stop.
7. Compare `/status` `battery_v` with the multimeter, then three 5-minute
   files on battery.
8. Close the box and repeat one file while flexing the box gently.

### 2.7 What does not change

The limits on the last page of the guide still apply to the mini: a smaller
box is not a tested box. Table test, adult, then calm solo movements, and no
throws or falls until the case and cell have been checked for impact and
crushing.

---

## Part 3. Options if the mini works

- **S1. Custom carrier PCB.** One small board that takes the XIAO and a
  GY-521 as castellated modules, with divider, fuse, switch and button
  footprints. Removes all hand wiring and makes units identical.
- **S2. Higher-range sensor.** If preflight shows clipping, an IMU with a
  ±32 g range (for example LSM6DSO32) or a second high-g accelerometer. This
  needs a new driver and a format change, so it is a separate piece of work.
- **S3. Longer recording.** The block-timestamp change (F6) first; a microSD
  breakout only if 16 MB is still too short, since it adds size and a
  connector that can shake loose.

## Notes from the photo of the new unit (IMG_4498) and the "v1 blueprints" infographic

- **Board:** "ESP32-S3 2025-V1.4", 44 pins, two USB-C ports (COM and USB-OTG).
  The regulator is on the other side and is not visible, so P3 is still open.
- **Joints:** the wires are twisted around the header pins and pushed through
  the GY-521 holes; no solder is visible. A twisted joint in the power path
  adds resistance and drops out under vibration, which looks exactly like
  "not enough power". Solder every joint before judging the battery.
- **Colours:** both sensor power wires are red and both signal wires are
  black. Mark GND and SCL with tape before soldering.
- **Infographic vs firmware.** Three items differ from the code and the
  assembly guide; the code is what actually runs:

  | Infographic | Firmware / guide |
  |---|---|
  | RGB status LED on GPIO15/16/17 | One WS2812 on GPIO48 (`PIN_LED`) |
  | End-session button on GPIO5 | STOP on GPIO7 (guide step 12) |
  | "No LED" while logging | Dim green heartbeat, blue above 3 g |

  Its power wiring (LiPo → switch → `5V/VIN`, never to `3V3`), divider values,
  GPIO4 range and the 5 kΩ / 250 mA charger note agree with the guide.

## Open questions

1. What is in the video that the PDF does not show? A public or unlisted
   `youtube.com/watch?v=…` link will not open from this session either, so a
   few sentences or photos of the new version would let me check this document
   against it.
2. Which regulator is on the current long dev board (P3)? A photo of the
   component side near the USB ports would answer it.
4. What exactly happens on one cell: resets at power-on, resets later, or too
   short a runtime? The first two point to peak current (P1b, P3); the third
   to capacity.
3. How long must one recording be? 54 minutes on 16 MB is under a 90-minute
   session, which decides whether F6 is optional or required.
