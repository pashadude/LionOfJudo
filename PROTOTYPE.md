# Budget Prototype: Sony X3000 + iPhone + Wearable IMUs

The working prototype built around gear on hand (~$41 of Temu parts).
Replaces the 3×Raspberry-Pi rig described in HARDWARE_SETUP.md.

```
DOJO (zero infrastructure)                 HOME (MacBook M2)
──────────────────────────                 ─────────────────────────
Sony X3000 on tripod  ─┐                   1. Copy both video files
iPhone on tripod      ─┼─ record normally  2. Plug sensors into USB →
2 IMU units in gi     ─┘                      python tools/fetch_imu.py
                                           3. python -m pipeline.run_session
Son does 3 jumps at start,                 4. Pick your athlete: 1 keypress
3 claps at the end (sync ritual)              per clip, then walk away
                                           5. Read session_report.md
```

## Hardware (per unit — build 2: chest + hip)

| Part | Notes |
|---|---|
| ESP32-S3 SuperMini | **must be 16 MB flash (N16R8)** — 4 MB fits only ~35 min |
| MPU-6050 (GY-521) | 4 wires to ESP32: 3V3, GND, SDA→GPIO8, SCL→GPIO9 |
| Protected LiPo 502030 250 mAh | via mini slide switch; TP4056 USB-C board for charging |
| Rigid pill capsule ~50×25 mm | lined with 2 mm silicone sheet; unit must not rattle |

Battery ADC: 100k/100k divider from cell + to GPIO4 (low-batt cutoff closes
the log file cleanly). Safer chemistry option: 10440 LiFePO4 (AAA size, no
thermal runaway) fed to the 3V3 pin.

**Placement:** chest capsule on the sternum under the crossed lapels; hip
capsule at the FRONT of the belt behind the knot. Never on the spine.
Inspect cells for puffing before every session.

## Firmware (`firmware/lionimu/`)

Edit `config.h` per unit (`UNIT_ID`/`UNIT_NAME`, home WiFi credentials), then:

```bash
arduino-cli compile --fqbn esp32:esp32:esp32s3:FlashSize=16M,PartitionScheme=app3M_fat9M_16MB firmware/lionimu
arduino-cli upload -p /dev/cu.usbmodem* --fqbn esp32:esp32:esp32s3:FlashSize=16M,PartitionScheme=app3M_fat9M_16MB firmware/lionimu
```

Boot logic: sees home WiFi → download mode (slow green blink,
`http://lion-chest.local`). No home WiFi (= at the dojo) → 10 s amber
countdown, then logs at 200 Hz until power-off (blue flash on impacts,
dim green heartbeat). Purple blink = MPU-6050 not wired right. Red = flash
or battery problem.

## Session workflow

1. **At the dojo:** switch both units on, start both cameras. Son does
   **3 sharp jumps** in front of the cameras (must be audible — landing slap
   or a parent clap on each jump). Train normally. **3 claps** at the end.
2. **At home:** plug units into USB (charges + enables WiFi download):
   ```bash
   python tools/fetch_imu.py --out sessions/$(date +%F)/imu/ --wipe
   ```
3. **Run the pipeline** (overnight for a long session):
   ```bash
   python -m pipeline.run_session \
       --sony /path/to/C0012.MP4 --iphone /path/to/IMG_4411.MOV \
       --imu sessions/2026-07-05/imu/ --out sessions/2026-07-05/
   ```
   The only interactive step comes early: for each detected throw you get one
   window — press the number over your athlete (0 if he's not in frame).
   Everyone else's face is always blurred; if his track is lost mid-clip his
   face gets blurred too (noted in the report), never the reverse.
4. **Review** `sessions/<date>/session_report.md`: per throw — technique
   guess + reasoning, peak g / rotation / power index from both sensors, and
   annotated clips per camera.

## Useful knobs

- `--threshold-g 2.5` — throw detection threshold. Default 3.0 is tuned for
  adults; a light 8-year-old may need 2.0–2.5. Re-running segmentation is
  cheap. Inspect a session first with:
  `python tools/imu_plot.py sessions/<date>/imu/chest_001.bin --threshold-g 2.5`
- `--blur-all` — skip the picker, blur everyone (fastest, safest to share).
- `--no-blur` — private review only. Never share this output.
- `--scale-height 0` — keep full 4K in clips (slower inference).
- `--vision-llm` is intentionally NOT implemented yet; the rule-based
  classifier (`phase0_judo_analysis.py`) runs offline at $0.

## Bench-test checklist (Phase A/D, after parts arrive)

1. Breadboard one unit, flash, power via USB away from home WiFi → logs.
2. Clap it 3×, throw it onto a mattress a few times while filming on iPhone.
3. `python tools/imu_plot.py <log>` — ritual found? impacts visible?
4. `python -m pipeline.run_session --sony <iphone_video> --imu <dir> ...`
   → report should contain exactly the mattress throws.
