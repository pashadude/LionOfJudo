# Video-Only Coach Review Pilot Design

Date: 2026-07-31  
Status: Approved design

## Objective

Produce a reviewable judo-training session from one Sony FDR-X3000 recording
and one iPhone recording when wearable IMU data is absent or unusable. The
system creates a synchronized side-by-side session video and a local coach
review application in Serbian Latin. It derives comparative motion measures
from video, proposes spoken technique labels, and keeps the coach as the final
authority for labels and quality.

This is a separate, video-only path. It must not manufacture IMU, power,
force, acceleration, or biomechanical claims from the recordings.

## Source Session Rules

### Synchronization anchors

The athlete in blue performs two visually and audibly observable sequences of
three taps on the wearable location:

1. The first sequence follows the spoken start cue and starts the usable
   session.
2. The second sequence occurs near the end of the usable session, roughly two
   throws before the reported injury event.

The processor detects transient clusters as candidate anchors. The review UI
shows both cameras around each candidate and requires the operator to confirm
or adjust them at frame resolution. A linear time map from iPhone time to Sony
time is then fitted from the two confirmed anchors, so a small camera clock
drift is handled rather than assuming one constant offset.

Automatic full-recording audio correlation is only a preliminary suggestion.
Repeated tatami impacts can create stronger but incorrect matches, so it
cannot be accepted as the final synchronization source.

### Injury cutoff

The usable session ends at the first frame where the athlete in white remains
on their knees after the reported injury event. The processor stores the
preceding throwing sequence as a distinct `prijavljen_povredni_dogadjaj`, but
excludes it from normal performance statistics, automatic technique proposals,
and any future training dataset.

The app displays the cutoff candidate and requires confirmation. The actual
source files are never deleted or overwritten.

## Processing Architecture

### Inputs

- Sony video, used as the master timeline and primary audio source.
- iPhone video, used as the second visual angle.
- Optional configuration for the target athlete color (`plavo`) and initial
  model path (`yolo11x-pose.pt`).

### Synchronization and exports

The import command creates a session directory containing:

- a sync manifest with the two confirmed anchors, the linear time-map
  coefficients, and the injury cutoff;
- source-relative clips for every detected event from both cameras;
- one side-by-side MP4 of the usable session; and
- previews around the anchors and cutoff for later audit.

The side-by-side video is a convenience deliverable. The browser review view
uses separate video elements because independent, synchronized seeking is more
useful for coaching than a pre-rendered composite.

### Pose and event detection

YOLO11 pose supplies the standard 17 COCO body keypoints. The UI asks the
operator to click the athlete in blue once in an early clear frame. Tracking
then combines the chosen pose track with blue-gi appearance cues; either signal
alone is insufficient during grips and occlusions.

Movement energy from the selected blue track creates suggested attempt windows.
The coach can adjust bounds, split, merge, delete, or create an event. No
automatic technique classification is considered confirmed.

For a non-injury event the video measures are:

- normalized entry speed, based on hip-center travel per torso length;
- 2D torso/shoulder rotation-rate proxy;
- normalized hip-level change;
- stance/base change; and
- recovery-to-stable-stance duration.

`intenzitet_pokreta` is a 0-100 score normalized within the same session. It
is not power, force, real-world velocity, or a clinical biomechanics result.

### Voice-derived technique proposals

The Sony audio is transcribed in Serbian. The processor searches the timed
transcript for an editable judo vocabulary and associates nearby spoken terms
with suggested events. The raw transcript, matched phrase, confidence, and
suggested technique are retained. Low-confidence or missing speech remains
blank rather than being invented.

The coach always confirms or overwrites `predlog_tehnike` before the value is
exported as `potvrdena_tehnika`.

## Coach Application

The local-only server binds to `127.0.0.1` by default and opens a browser
session review page. All visible application text uses Serbian Latin. Free-text
notes accept Latin, Cyrillic, Russian, and other Unicode input.

### Review workflow

1. Confirm the blue athlete and the two triple-tap anchors.
2. Confirm the injury cutoff.
3. Review suggested movement windows and voice-derived technique proposals.
4. Select an event and compare the synchronized Sony and iPhone views.
5. Inspect charts, choose a confirmed technique, quality score 1-5, and note.
6. Save and export the review.

### Primary interface

- A compact event list, with timestamp, proposal, review state, and injury
  flag.
- Two synchronized video panels with play, pause, frame step, restart, and
  seek.
- Charts under the video: `Brzina ulaska`, `Rotacija trupa (2D)`, `Visina
  kukova`, `Stabilnost`, and `Intenzitet pokreta`.
- A shared playback cursor and click-to-seek chart interaction.
- Fields labelled `Predlog tehnike`, `Potvrđena tehnika`, `Ocena`, and
  `Napomena`.
- Explicit controls: `Sačuvaj`, `Podeli`, `Spoji`, `Obriši`, and `Izvezi
  izveštaj`.

## Data Contract and Outputs

`review.json` is the editable canonical output. It contains session metadata,
sync anchors, the injury cutoff, tracks, events, voice suggestions, and coach
annotations. Each normal event includes:

- `event_id`
- `sony_start_s`, `sony_end_s`
- `iphone_start_s`, `iphone_end_s`
- `predlog_tehnike`, `potvrdena_tehnika`
- `glasovna_fraza`, `pouzdanost_glasa`
- `brzina_ulaska_norm`
- `rotacija_trupa_2d_dps`
- `promena_visine_kukova_norm`
- `vreme_oporavka_s`
- `intenzitet_pokreta_0_100`
- `ocena`
- `napomena`
- `iskljuceno_iz_statistike`

Every save atomically regenerates a Serbian-Latin CSV and Markdown report.
The injury event remains in the record for traceability but has
`iskljuceno_iz_statistike: true`.

## Error Handling

- If no reliable anchor candidate is found, require manual Sony/iPhone marker
  placement; do not generate a composite video with unverified timing.
- If the blue athlete cannot be tracked through an event, mark the event as
  `nedovoljno_vidljivo` and omit calculated pose metrics.
- If transcription is unavailable or uncertain, leave the suggestion empty and
  preserve the audio clip for manual review.
- If a source video lacks audio, allow visual-only anchor confirmation.
- Never delete source footage or silently include the injury-marked event in
  normal aggregates.

## Verification

Automated checks cover:

- two-anchor affine sync mapping and frame-time conversion;
- rejection of false full-recording audio-correlation matches;
- event-window editing and injury-event exclusion;
- JSON/CSV/Markdown validation and atomic annotation persistence;
- Serbian-Latin API labels and UI rendering; and
- local path containment for video serving.

Manual verification covers:

- frame-level visual confirmation of both tap sequences;
- alignment of one visible throw in both views after the fitted mapping;
- injury cutoff confirmation;
- blue-athlete selection and tracking across grip/occlusion; and
- playback, chart seek, save, and side-by-side MP4 review in a browser.

## Deferred Scope

- IMU ingestion, power/impact measures, and wearable-to-video alignment;
- Raspberry Pi collection;
- a second hip sensor and two-sensor biomechanics;
- autonomous technique recognition or medical conclusions;
- cloud hosting, sharing, and multi-user accounts.
