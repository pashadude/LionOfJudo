# 🥋 LionOfJudo: AI-Powered Judo Training Analysis

**Mission:** Bring professional sports-science analysis to underfunded judo schools using affordable hardware and AI.

**Status:** ✅ **Working prototype** — software pipeline verified end-to-end; wearable sensor firmware compiled and ready to flash (hardware bench test pending parts delivery).

---

## What the System Does

Record a normal training session with two ordinary cameras while the athlete wears two matchbox-sized motion sensors inside the judogi. Afterwards, one command on a laptop produces:

1. **Automatic throw detection** — every throw found from the sensors' g-force spikes (no manual scrubbing through an hour of footage)
2. **Per-throw video clips** from both camera angles, perfectly synchronized
3. **Skeleton overlay** — 17-keypoint pose tracking drawn on every clip with biomechanical measurements (hip height, torso angle, knee angle)
4. **Technique classification** — rule-based recognition (o-soto-gari, o-goshi, ippon-seoi-nagi, uki-goshi, ...) with confidence and reasoning
5. **Direct power measurement** — peak g-force, rotation speed (°/s), throw duration, and a power index from the chest and hip sensors (measured, not estimated from video)
6. **Privacy protection** — every face in frame is blurred except the designated athlete's; failure mode always errs toward blurring more, never exposing anyone

Everything runs **offline** on a MacBook (Apple Silicon / MPS). No cloud, no subscriptions, no per-session cost.

---

## Architecture

```
AT THE DOJO (zero infrastructure)          AT HOME (MacBook M2)
─────────────────────────────────          ───────────────────────────────
Sony FDR-X3000 on tripod   ─┐              1. Copy both video files
iPhone on tripod           ─┼─ just record 2. Plug sensors into USB power →
2 IMU units in the judogi  ─┘                 they auto-join home WiFi →
                                              python tools/fetch_imu.py
Sync ritual:                               3. python -m pipeline.run_session
  3 sharp jumps at session start           4. One keypress per throw to mark
  3 claps at session end                      your athlete, then walk away
                                           5. Read sessions/<date>/session_report.md
```

**Three clocks, one timeline.** The two cameras are aligned by cross-correlating their audio tracks (the master timeline is the Sony's). The sensors' internal clocks are aligned to the same timeline via the jump ritual: the three g-spikes in the sensor log are matched against the three corresponding audio transients by their spacing fingerprint. The optional end-of-session claps give a second anchor point, which absorbs the ESP32 crystal drift (±20 ppm ≈ ±0.1 s over 90 min) with a two-point linear fit. Verified residuals: **under 4 ms**.

**Why sensors instead of video motion detection?** A g-spike on the athlete's body is a far more reliable throw signal than pixel-space hip tracking when six kids share the mat. It also measures what video can only guess: actual impact force and hip rotation speed.

**Why flash logging instead of live streaming?** A 2.4 GHz radio strapped to a child being thrown onto a mat drops packets constantly. The units log to their own flash (2+ hours at 200 Hz on 16 MB) and upload over WiFi at home. Radio-off logging also triples battery life. Nothing at the dojo needs setting up, powering, or debugging.

---

## Hardware

### Cameras
Any two cameras with audio work. This build uses a **Sony FDR-X3000** (4K action cam) and an **iPhone**. No sync cables, no genlock — audio cross-correlation replaces all of it.

### Wearable IMU units (2× — chest + hip)

| Part | Spec | Why |
|---|---|---|
| ESP32-S3 SuperMini | **16 MB flash (N16R8)** — verify before ordering! | Samples the IMU at 200 Hz, logs to flash, serves WiFi download. 4 MB variants only hold ~35 min. |
| MPU-6050 (GY-521) | ±16 g accel, ±2000 °/s gyro | The actual sensor. Judo impacts peak well under 16 g. |
| Protected LiPo 502030, 250 mAh | built-in protection circuit (PCM) | ~40 mA draw → 5–6 h per charge. **Do not use unprotected cells on a child.** |
| TP4056 USB-C board | 1 A charger | Plug USB-C at home: charges + enables WiFi download simultaneously. |
| Mini slide switch, JST-PH pair, 30 AWG silicone wire | | On/off + detachable battery. |
| Rigid pill capsule ~50×25 mm + 2 mm silicone sheet | | Rigid shell stops puncture; silicone spreads impact. The board must not rattle inside (loose mounting smears the g readings). |

**Total: ≈ $41 including spares** (Temu prices). Safer-chemistry option: 10440 LiFePO4 (AAA-size, no thermal runaway when crushed) + holder + LiFePO4 charger, ~$6 more and slightly bulkier.

**Wiring:** MPU-6050 → ESP32: `VCC→3V3, GND→GND, SDA→GPIO8, SCL→GPIO9`. Battery + → 100k/100k divider → GPIO4 (low-battery cutoff closes the log file cleanly before brownout).

**Placement on the judogi:**
- **Chest unit:** centered on the sternum, under the crossed lapels (ukemi lands on the back/side; the lapels add padding)
- **Hip unit:** front of the belt, tucked behind the knot
- **Never on the spine or lower back** — direct mat impact zone
- Sewn silicone-lined pockets with velcro flaps, ~70×50 mm
- **Inspect cells for puffing or dents before every session; retire on any damage**

### Compute
Any Apple Silicon Mac (uses MPS). A full session processes overnight. No GPU server needed.

---

## Firmware (`firmware/lionimu/`)

Single Arduino sketch, no external cloud, no app. Zero buttons — the boot mode is chosen by location:

1. **Power on** → scan for the home WiFi SSID for 8 s
2. **Found (you're at home):** joins WiFi, announces itself as `lion-chest.local` / `lion-hip.local`, serves HTTP endpoints `/list`, `/download?f=`, `/delete?f=` (POST), `/status` (battery voltage, free flash). *Slow green blink.*
3. **Not found (you're at the dojo):** 10 s amber countdown, then logs at 200 Hz until power-off or low battery. *Blue flash on impacts (visible liveness check), dim green heartbeat otherwise.*

**LED codes:** green blink = download mode · amber countdown = about to log · blue flash = impact recorded · purple blink = MPU-6050 not responding (check wiring) · red = flash/battery fault (log was closed cleanly).

**Binary log format `LJIM` v1** (must stay in sync with `pipeline/imu_ingest.py`):

```
Header (32 B): 'LJIM' | u8 version=1 | u8 unit_id (0=chest,1=hip) | u16 rate_hz
               | f32 accel_scale (LSB/g) | f32 gyro_scale (LSB/°/s) | 16 B reserved
Record (16 B): u32 millis | i16 ax,ay,az | i16 gx,gy,gz     (little-endian)
→ 3.2 KB/s = 11.5 MB per hour at 200 Hz
```

**Build & flash** (per unit — set `UNIT_ID`/`UNIT_NAME` and WiFi credentials in `config.h` first):

```bash
arduino-cli compile --fqbn esp32:esp32:esp32s3:FlashSize=16M,PartitionScheme=app3M_fat9M_16MB firmware/lionimu
arduino-cli upload -p /dev/cu.usbmodem* --fqbn esp32:esp32:esp32s3:FlashSize=16M,PartitionScheme=app3M_fat9M_16MB firmware/lionimu
```

(The 9.9 MB data partition of that scheme is labeled `ffat`; the sketch mounts it as LittleFS explicitly.)

---

## Software Pipeline (`pipeline/`)

| Module | What it does |
|---|---|
| `audio_sync.py` | Decodes both cameras' audio via ffmpeg, builds onset envelopes (robust to different mics/AGC), cross-correlates → sub-frame offset + confidence ratio. Also extracts clap/transient candidates for IMU alignment. |
| `imu_ingest.py` | Parses `LJIM` logs into numpy; spike detection with per-event peak dedup; 3-spike ritual detector; spacing-fingerprint matching of ritual↔audio transients; least-squares clock fit (`t_master = a·t_imu + b`); `measure_throw_power` → peak g, duration, max rotation, power index. |
| `throw_segmenter.py` | Merges chest+hip spikes (mapped to master time) into throw events, drops the ritual events, emits clip windows `[peak−4 s, peak+3 s]`, merges overlaps. A hip-only or chest-only spike still counts — a throw missed by one sensor isn't lost. |
| `face_blur.py` | Two blur layers: (1) YOLO11-pose head keypoints → Gaussian ellipse over every head; (2) YuNet face detector as safety net for anyone pose missed (spectators, partial bodies). Athlete exemption via ByteTrack: one keypress on the clip's first frame; if his track is lost mid-clip his face gets blurred too for those frames (recorded in the report) — the failure mode is always over-blur, never expose. |
| `clip_extractor.py` | Frame-accurate ffmpeg re-encode cuts (stream-copy would snap to the Sony's sparse long-GOP keyframes). Optional downscale to 1080p for faster inference on 4K sources. |
| `run_session.py` | The orchestrator. Reuses the existing analyzers unchanged: `VisualJudoAnalyzer` (skeleton overlay + pose JSON), `classify_technique` (rule-based recognition), `MovementAnalyzer` (movement phases). Writes `session_report.md` + `.json`. |

**Learning components:**

| Module | What it does |
|---|---|
| `pipeline/pose_features.py` | Clip → normalized pose-sequence features: picks the thrower (most hip-vertical travel), hip-centers and torso-scales every frame, resamples to 32 timesteps, adds joint angles + velocities. Cached as `.npz` beside each clip. |
| `pipeline/train_classifier.py` | Trains a HistGradientBoosting classifier on the `dataset/` corpus with stratified k-fold CV; prints per-class precision/recall + confusion matrix and names the weakest class (= what to film next). Saves `models/technique_clf.joblib`. |
| `tools/build_corpus.py` | Corpus builder: `--auto` sorts clips by technique slug in the filename, `--label` is a keypress UI for the rest, `--split-reps` cuts multi-rep drill videos into per-rep samples, `--from-session` feeds confirmed throws from processed sessions back into the dataset. Prints per-class counts every run. |

**Corpus guidance:** own footage beats web scraping (same dojo, same angles, kid-sized dynamics). Layout: `dataset/<technique-slug>/clip.mp4` — folder = label. Count repetitions, not videos. **20 reps/class minimum, 50 solid, ~100 diminishing returns**; 5–8 techniques ≈ 150–400 samples. `run_session.py` uses the trained model automatically (`method: learned`) and falls back to the rules when none exists.

**Tools:**
- `tools/fetch_imu.py` — pulls logs from both units over WiFi, size-verifies, optional `--wipe`
- `tools/imu_plot.py` — g-force/rotation plot with detected spikes and ritual marked; the threshold-tuning instrument
- `tools/make_synthetic_log.py` — generates fake sensor logs so the whole pipeline can be tested with zero hardware
- `tools/build_corpus.py` — dataset builder (see Learning components)

**Dependencies:** the existing venv (ultralytics, opencv, scipy, numpy, torch) plus `lap` (ByteTrack) and the ffmpeg CLI. The YuNet model (230 KB) is committed in `models/`.

---

## Usage

### One-time setup
```bash
uv pip install lap                  # tracking dependency
brew install ffmpeg arduino-cli     # if not present
# flash both units (see Firmware above), sew the pockets
```

### Every session
```bash
# 1. At the dojo: sensors on, cameras on, 3 jumps, train, 3 claps.

# 2. At home: plug sensors into USB, then
python tools/fetch_imu.py --out sessions/$(date +%F)/imu/ --wipe

# 3. Run the pipeline (overnight for long sessions)
python -m pipeline.run_session \
    --sony  /path/to/C0012.MP4 \
    --iphone /path/to/IMG_4411.MOV \
    --imu   sessions/2026-07-05/imu/ \
    --out   sessions/2026-07-05/

# 4. The only interactive step comes early: one window per throw,
#    press the number over your athlete (0 = not in this clip).
#    Then it runs unattended.

# 5. Morning: open sessions/2026-07-05/session_report.md
```

### Output layout
```
sessions/2026-07-05/
├── imu/                      chest_001.bin, hip_001.bin
├── throws/throw_01/
│   ├── sony_raw.mp4              exact cut, unprocessed
│   ├── sony_blurred.mp4          all faces blurred except your athlete
│   ├── sony_blurred_cam0_annotated.mp4   + skeleton & measurements
│   ├── sony_blurred_analysis.json        per-frame 17-keypoint poses
│   ├── iphone_*                  same set for the second angle
│   └── ...
├── session_report.json
└── session_report.md         per throw: technique + confidence + reasoning,
                              peak g / rotation / duration / power index
                              per sensor, links to clips, blur coverage notes
```

### Tuning knobs
| Flag | Default | When to change |
|---|---|---|
| `--threshold-g` | 3.0 | Throw detection threshold. Tuned for adults; a light 8-year-old's throws may peak at 2–4 g — try 2.0–2.5. Re-running segmentation is cheap (no re-inference). Inspect first with `tools/imu_plot.py <log> --threshold-g 2.5`. |
| `--blur-all` | off | Skip the picker; blur everyone including your athlete. Fastest, safest to share. |
| `--no-blur` | off | Skip blurring entirely. **Private review only — never share this output.** |
| `--scale-height` | 1080 | `0` keeps full 4K (slower inference). |
| `--device` | mps | `cpu` on non-Apple hardware. |

---

## Verification Status

Every module was verified before commit:

| Test | Result |
|---|---|
| Audio sync on a synthetically shifted copy of real footage | Recovered 3.700 s offset **exactly**, both directions, confidence 5.2 |
| IMU parse + ritual detection + clock fit on synthetic logs with simulated drift | Drift and offset recovered, residuals **< 4 ms** |
| Segmentation on two-unit logs incl. a hip-only spike and rituals | All 4 throws found, rituals excluded, units merged |
| Face blur on real multi-person footage | All faces blurred; selected track clear 247/247 frames; duplicate appearances of the same person on other tracks stayed blurred (fail-safe confirmed) |
| Clip extraction | 7.000 s requested → 7.000 s delivered |
| Full end-to-end dry run (2 cameras + 2 sensor logs → report) | 2/2 throws detected, clips cut, blurred, annotated, classified, report written |
| Firmware | Compiles for ESP32-S3 16 MB: 32% flash, 16% RAM |

Remaining (hardware-gated): bench test of a real unit, mattress-throw test, dress rehearsal at a real training. Checklist in [PROTOTYPE.md](PROTOTYPE.md).

---

## Privacy

This system records children. The rules are built into the code, not the workflow:

- Every detected face is blurred **by default**; only one explicitly confirmed, continuously tracked athlete is exempted
- Two independent detection layers (pose heads + YuNet faces) so people the pose model misses still get blurred
- Track loss blurs the exempted athlete rather than risking anyone else
- Blur coverage gaps are listed in the report for review before sharing anything
- No cloud upload of any footage; everything stays on the local machine

---

## Repository Layout

```
pipeline/           the post-training processing pipeline (see table above)
firmware/lionimu/   ESP32-S3 wearable logger (Arduino)
tools/              fetch_imu.py · imu_plot.py · make_synthetic_log.py
models/             YuNet face detection ONNX (230 KB)
PROTOTYPE.md        build guide: wiring, flashing, session workflow, bench checklist
examples/, data/    test footage
analysis/           outputs of the earlier single-video analysis scripts

# Earlier-generation scripts (still used by the pipeline or standalone):
phase0_visual_analysis.py    VisualJudoAnalyzer — skeleton overlay (reused by run_session)
phase0_judo_analysis.py      rule-based technique classifier (reused by run_session)
movement_analysis.py         movement phase extraction (reused by run_session)
judo_hybrid_recognition.py   optional YOLO + Vision-LLM hybrid (off by default, $)
compare_techniques.py        training vs performance comparison
batch_process.py             multi-video batch runner
```

### Legacy design documents
The original plan targeted a fixed 3× Raspberry Pi camera rig ($431+). Those documents remain for reference — [PROJECT_PLAN.md](PROJECT_PLAN.md), [HARDWARE_SETUP.md](HARDWARE_SETUP.md), [ACCELEROMETER_SYSTEM.md](ACCELEROMETER_SYSTEM.md) (the wired-sensor concept the wearable units evolved from), [PHASE_0_TESTING.md](PHASE_0_TESTING.md), [LORA_FINETUNING.md](LORA_FINETUNING.md), [YOLO_POSE_GUIDE.md](YOLO_POSE_GUIDE.md). The current build in this README supersedes them: **two cameras you already own + $41 of sensors** instead of $431 of dedicated hardware.

---

## Cost

| | This prototype | Original 3×Pi plan | Commercial systems |
|---|---|---|---|
| Hardware | **≈ $41** (sensors; cameras already owned) | $431–462 | $10,000–50,000 |
| Monthly | **$0** (fully offline) | ~$21 (cloud) | $500+ |
| Per session | **$0** | $0.005–0.02 | — |

The optional Vision-LLM classification upgrade (`judo_hybrid_recognition.py` via OpenRouter) would add ~$0.01–0.10 per session when enabled.

---

## Roadmap

- [x] Pose pipeline, technique rules, movement analysis (earlier phases)
- [x] Two-camera audio sync, IMU ingest, throw segmentation, face blur, orchestrator
- [x] ESP32-S3 firmware (compiled; awaiting hardware)
- [x] Learned technique classifier: corpus tools, pose-sequence features, trainer with CV reporting, auto-integration with rules fallback
- [ ] Bench test + mattress test (parts on order)
- [ ] Dress rehearsal at a real training; tune `--threshold-g` for a child's throw forces
- [ ] Grow the corpus to ≥20 reps per technique and retrain (per-class table shows progress)
- [ ] Technique classification upgrade: per-technique IMU signatures; optional Vision LLM
- [ ] Progress tracking across sessions (same athlete, same technique, power trend)
- [ ] Multi-athlete support (more sensor sets)

---

## License

MIT — see [LICENSE](LICENSE).

**Built with love for judo and kids who deserve better training tools.**
