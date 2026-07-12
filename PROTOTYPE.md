# Budget Prototype: Sony X3000 + iPhone + Wearable IMUs

The working prototype built around gear on hand (~$41 of Temu parts).
Replaces the 3×Raspberry-Pi rig described in HARDWARE_SETUP.md.

```
DOJO (zero infrastructure)                 HOME (MacBook M2)
──────────────────────────                 ─────────────────────────
Sony X3000 on tripod  ─┐                   1. Copy both video files
iPhone on tripod      ─┼─ record normally  2. Pi pulls IMU logs on JudoNet →
2 IMU units in gi     ─┘                      Mac copies session folder
                                           3. python -m pipeline.run_session
Son does 3 physical spikes at start/end,   4. Pick your athlete: 1 keypress
not ordinary off-body claps                   per clip, then walk away
                                           5. Read session_report.md
```

## Hardware (per unit — build 2: chest + hip)

| Part | Notes |
|---|---|
| ESP32-S3 SuperMini | **must be 16 MB flash (N16R8)** — 4 MB fits only ~35 min |
| MPU-6050 (GY-521) | 4 wires to ESP32: 3V3, GND, SDA→GPIO8, SCL→GPIO9 |
| Protected LiPo 502030 250 mAh | via mini slide switch; disconnect and charge with an external TP4056 only |
| Rigid rectangular case 64×44×20 mm | 3 mm silicone pad belongs on the body-facing pocket side; unit must not rattle |

Battery ADC: 100k/100k divider from cell + to GPIO4 with a 100 nF capacitor
from GPIO4 to GND (low-batt cutoff closes the log file cleanly). Never feed a
LiPo or LiFePO4 cell directly to the ESP32 `3V3` pin. A LiFePO4 alternative
needs its own compatible charger and a correctly designed 3.3 V power stage.

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

1. **At the dojo:** switch both units on, wait until the amber countdown has
   finished, then hold still for 5 seconds. Do **3 deliberate heel-drops or
   mounted-unit taps** in front of the cameras. Every event must be audible
   and create a physical spike in both wearables. Train normally. End with the
   same physical three-spike ritual, then leave 10 seconds still.
2. **After training:** power the units off. Turn on JudoNet/Pi, power the
   units back on so they enter download mode, and collect:
   ```bash
   python tools/rpi_collect_imu.py --root /data/lionofjudo --session $(date +%F)
   ```
   Before video processing, check the raw data:
   ```bash
   python tools/imu_preflight.py sessions/2026-07-05/imu/
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

## Training your own technique classifier

The default rule-based recognition is weak. Train a real classifier from
your own clips — no cloud, no cost:

```bash
# 1. Dump your clips into one folder, then auto-sort by filename
#    (any file containing a technique slug like "o-goshi" gets filed):
python tools/build_corpus.py --auto ~/JudoClips/inbox/

# 2. Keypress-label whatever had no technique in the name:
python tools/build_corpus.py --label ~/JudoClips/inbox/

# 3. Split multi-repetition drill clips into one clip per rep:
python tools/build_corpus.py --split-reps dataset/o-goshi/long_drill.mp4

# 4. Extract pose features + train + see per-class accuracy:
python -m pipeline.train_classifier --extract
```

Dataset layout: `dataset/<technique-slug>/clip.mp4` — **folder = label**,
filenames inside don't matter. Count repetitions, not videos: a 60s drill
video with 5 throws = 5 samples after `--split-reps`.

**How much data:** 20 reps/technique = minimum, 50 = solid, ~100 =
diminishing returns. 5–8 techniques → 150–400 samples total. Use your own
footage (same dojo, same kid-sized bodies); web videos only to patch a
class stuck under 20. The fastest data source is filming 10 minutes of
nagekomi of whatever the confusion matrix says is the weakest class.

**Two-tier recognition.** Besides the trained classifier (precise on the
techniques with ≥20 reps), the full catalog — even waza with a single demo
clip — powers a nearest-neighbor reference matcher:

```bash
python -m pipeline.pose_features dataset/       # extract everything (hours, cached)
python -m pipeline.reference_bank --build       # build models/reference_bank.npz
```

Every detected throw then also gets "nearest waza" matches from the whole
catalog in the report, each tagged `[throw]` (nage-waza) or `[hold]`
(katame-waza, detected from the name: -gatame/-jime/-garami/...). Rebuild the
bank whenever you add clips.

`run_session.py` picks up `models/technique_clf.joblib` automatically and
reports `method: learned` (falling back to rules when no model exists).
After each session, feed confirmed throws back into the dataset:

```bash
python tools/build_corpus.py --from-session sessions/2026-07-05/
```

## Bench-test checklist (Phase A/D, after parts arrive)

1. Breadboard one unit, flash, power via USB away from home WiFi → logs.
2. Clap it 3×, throw it onto a mattress a few times while filming on iPhone.
3. `python tools/imu_plot.py <log>` — ritual found? impacts visible?
4. `python -m pipeline.run_session --sony <iphone_video> --imu <dir> ...`
   → report should contain exactly the mattress throws.
