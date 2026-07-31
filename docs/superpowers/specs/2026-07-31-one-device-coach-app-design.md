# One-Device Live Pilot and Coach App Design

Date: 2026-07-31
Status: Approved design

## Objective

Deliver a field-testable LionOfJudo workflow using the single assembled
ESP32-S3 + GY-521 wearable, one Sony FDR-X3000, and one iPhone 15. The wearable
records IMU data locally during training. After training, the Mac downloads the
log over an iPhone hotspot, aligns it with both videos, detects movement events,
and presents synchronized video, charts, and trainer annotation controls in a
local browser application.

The Raspberry Pi collector, a second wearable, pose inference, and automatic
technique recognition remain compatible future additions but do not participate
in this pilot.

## System Boundary

### Wearable

The ESP32 firmware is the only software installed on the wearable. It samples
the MPU-6050 at 200 Hz, records the existing LJIM v1 binary format to LittleFS,
and exposes the stored logs through the existing HTTP endpoints when the
configured iPhone hotspot is available.

The firmware has two mutually exclusive boot modes:

1. Hotspot absent: wait through the amber countdown and start local logging.
2. Hotspot present: join WiFi and enter download mode without recording.

The hotspot must therefore remain off during recording and be enabled only
after the session has ended and the wearable has been powered off.

### Mac

The Mac owns all video processing, synchronization, chart rendering, and coach
annotations. The browser application is local-only and listens on loopback by
default. No session data is uploaded to a cloud service.

## Field Workflow

1. Place the wearable in the sternum pocket with its long edge vertical, rocker
   switch toward the athlete's head, padded flat side toward the body, and lid
   outward. The case and IMU must not move inside the pocket.
2. Keep the iPhone hotspot off. Start both cameras with audio enabled at
   1080p60.
3. Power on the wearable. Wait for the WiFi scan and amber countdown to finish.
   Confirm the logging heartbeat, then keep the device still for five seconds.
4. In view of both cameras, perform three firm physical taps on the wearable
   with deliberately unequal gaps. The taps must create both audible video
   transients and IMU acceleration spikes.
5. Perform only controlled drills during the first live test.
6. At the end, repeat the same three-tap ritual. Keep the wearable still for ten
   seconds so buffered records are flushed, then power it off.
7. Stop both cameras. Enable the iPhone hotspot and join it from the Mac.
8. Power the wearable on again. A slow green indication means download mode.
9. Download and size-verify logs without deleting the originals during initial
   tests.
10. Copy the Sony and iPhone files to the session input directory and run the
    pilot processor. Open the resulting local coach application.

## Camera Layout

The Sony is the master timeline. It is landscape, 4-6 m from the drill, about
1.4-1.6 m high, and 30-45 degrees diagonal to the athlete. It must show the
athlete's entire body and landing area.

The iPhone is landscape on the opposite side, 3-5 m from the drill, at a
70-90-degree side angle. It uses the 1x lens, records audio, and keeps the
hotspot disabled while recording. Both camera views must include the start and
end tap rituals.

## Pilot Processor

A lightweight command prepares one session without loading YOLO or other heavy
models. It reuses the existing audio synchronization, LJIM ingestion, ritual
alignment, movement segmentation, frame-aligned biomechanics, and clip
extraction modules.

Inputs:

- Sony master video
- optional iPhone video
- directory containing one `chest*.bin` LJIM log
- output session directory
- configurable movement threshold, initially 3.0 g

Outputs:

- session-level synchronization and quality metadata
- one Sony clip and optional iPhone clip per detected event
- one frame-aligned IMU CSV per event
- one review manifest consumed by the coach application
- editable trainer review data in JSON, CSV, and Markdown

The processor fails clearly when the IMU log is malformed, no three-tap ritual
can be matched, camera audio synchronization confidence is inadequate, or no
events exceed the configured threshold. A lower threshold can be used for a
lighter athlete without rerunning expensive inference.

## Coach Application

The coach application uses a Python local server and a dependency-light browser
frontend. It opens at `http://localhost:8765` and is launched with a session
directory argument.

### Review Screen

- Event list showing event number, proposed Sony timestamp, annotation status,
  and quality score.
- Sony and iPhone video views with synchronized play, pause, seek, and restart.
- Acceleration magnitude and rotation magnitude charts directly below video.
- A moving chart cursor tied to video playback time.
- Chart click-to-seek for frame-level inspection around the event.
- Compact metrics for peak acceleration, maximum rotation speed, event
  duration, and the repeatable power index.
- Trainer fields for confirmed technique, quality score from 1 to 5, and free
  notes.
- Previous, Save, and Next commands suitable for repeated annotation.

The application does not describe peak acceleration as force or the power index
as watts. They are sensor-local inertial measurements and comparative indices.

### Persistence

The server validates submitted values and writes annotation data atomically.
JSON is the canonical editable representation. CSV and Markdown are regenerated
after every save so the trainer can inspect or share the results without the
application.

The review table contains:

- `event_id`
- `proposed_timestamp_s`
- `sony_clip`
- `iphone_clip`
- `peak_acceleration_g`
- `max_rotation_dps`
- `duration_ms`
- `power_index`
- `confirmed_technique`
- `quality_score`
- `trainer_notes`
- `reviewed_at`

## Local API

The local server exposes only the active session:

- `GET /api/session`: session metadata and event summaries
- `GET /api/events/{id}`: event metadata and chart series
- `PUT /api/events/{id}/annotation`: validate and save coach fields
- `GET /media/...`: session video files with path containment checks

Static application assets are served from the repository. User-provided paths
cannot escape the selected session directory. The default bind address is
`127.0.0.1`, so another device cannot access the private footage unless the
operator explicitly changes the server configuration in a future version.

## Hardware and Safety Constraints

The GY-521 must remain rigidly fixed relative to the case; soft mounting the IMU
would distort the measured dynamics. Silicone is used to restrain the battery,
remove rattle, spread impact on the body-facing exterior, and cushion case
edges. The LiPo pouch must not be bent, compressed by the lid, contacted by pin
headers, or charged while worn. The external TP4056 charger stays outside the
case and the battery is disconnected from the wearable before charging.

The first test is limited to standing movement, controlled entries, and
controlled throws onto an appropriate mat. The device must pass visual battery
inspection, no-rattle inspection, logging bench test, and padded mattress test
before being worn.

## Verification

Automated tests cover:

- pilot processing from a synthetic one-device LJIM log
- event manifest generation and numerical metrics
- annotation validation and atomic JSON/CSV/Markdown persistence
- local API success, malformed input, missing event, and path traversal cases
- frontend loading and event selection

Manual verification covers:

- firmware compile and upload for the exact connected ESP32-S3 board
- logging mode with hotspot absent
- download mode with hotspot present
- log download and preflight without deletion
- browser video playback, synchronized seek, moving chart cursor, and save
- a short bench recording followed by an end-to-end session import

## Deferred Scope

- Raspberry Pi collection and display
- second hip wearable and two-sensor coupling measures
- automatic technique classification
- pose estimation and skeleton overlays
- face blurring and public sharing workflow
- live WiFi streaming during training
- cloud hosting and multi-user accounts
