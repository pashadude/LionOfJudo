# Trainer Dataset Export Design

## Goal

Extend the local LionOfJudo coach review so each locked assessment records the
trainer and wrestler identities and produces a structured JSON dataset for a
future multimodal LoRA pipeline. The first assessment must remain independent
of the AI challenge, while later trainer corrections remain available for
audit and research without silently becoming clean training labels.

This change covers software and exported data only. Physical GI hardware
optimization remains a separate follow-up.

## User Interface

Add a compact `Podaci sesije` section above the event assessment controls:

- `Ime trenera`: required text field, maximum 120 characters;
- `Ime sportiste`: required text field, maximum 120 characters, initially
  `Dusan` for the current prototype session;
- `Sacuvaj podatke`: explicit save command;
- `Preuzmi skup (JSON)`: downloads the active generation's trainer dataset.

The application must show the saved values after reload. A normal event cannot
lock a trainer assessment until both names are saved. Injury events remain
read-only and never create training examples.

Changing session participant fields affects only future assessment records.
Every locked assessment snapshots both names so historical provenance cannot
be rewritten by a later UI edit.

## Revision Semantics

The current internal `trener_procene[].revizija` is a session-wide unique
counter. It must remain unchanged as `source_trainer_revision` in exports.

The dataset additionally derives `assessment_revision` within one analysis
round, identified by:

- `event_id`;
- `event_revision`;
- `analysis_fingerprint`.

For the exact three-part key, assessments are sorted by the global internal
`revizija`; the one-based position is the exported `assessment_revision`.
Within that round:

- `assessment_revision: 1` has `assessment_phase: "pre_ai"`;
- every `assessment_revision > 1` has
  `assessment_phase: "post_ai_correction"`.

Changing event bounds creates a new `event_revision` and analysis fingerprint,
so the first assessment in the new round starts again at revision 1 and is
`pre_ai`. `assessment_phase` is stored explicitly and validated; consumers do
not infer it only from the number. A `post_ai_correction` additionally requires
an AI evaluation with the same event revision and fingerprint, a non-null
`ai_otkriven_u`, and `locked_at >= ai_otkriven_u`.

The canonical validator also enforces session-wide uniqueness of every
internal `trener_procene[].revizija`, including across different events.

## Storage Model

Session metadata is stored in the canonical review payload:

```json
{
  "participants": {
    "trainer_name": "Ime trenera",
    "wrestler_name": "Dusan",
    "updated_at": "2026-08-01T20:00:00+02:00"
  }
}
```

Every new internal trainer assessment snapshots:

```json
{
  "trainer_name": "Ime trenera",
  "wrestler_name": "Dusan"
}
```

Legacy assessments without snapshots remain readable but are permanently
ineligible for clean training export. A future explicit migration may attach an
immutable identity snapshot only when it also records migration provenance and
time. Merely saving current session participants never rewrites legacy authorship.

## JSON Dataset

Each new immutable generation includes two separate files:

- `trener_dataset.json`: clean, LoRA-eligible pre-AI examples only;
- `trener_assessment_audit.json`: complete trainer assessment history, including
  post-AI corrections and explicit ineligibility reasons.

Keeping history outside the training document makes accidental target leakage
materially harder. Both top-level formats are human-readable JSON rather than
JSONL so the current prototype can inspect and download them directly. A future
training job may deterministically convert `training_examples` to JSONL without
changing this source contract.

```json
{
  "schema_version": 1,
  "session_id": "trainer-ai-session",
  "generation_id": "0123456789abcdef0123456789abcdef",
  "generated_at": "2026-08-01T20:05:00+02:00",
  "participants": {
    "trainer_name": "Ime trenera",
    "wrestler_name": "Dusan"
  },
  "training_examples": [
    {
      "example_id": "trainer-ai-session:e-001:1:sha256-...:1",
      "trainer_name": "Ime trenera",
      "wrestler_name": "Dusan",
      "throw_name": "Tai-otoshi",
      "score_1_5": 4,
      "reasoning": "Obrazlozenje trenera sa citiranom Sony sekundom.",
      "assessment_revision": 1,
      "assessment_phase": "pre_ai",
      "source_trainer_revision": 1,
      "event_id": "e-001",
      "event_revision": 1,
      "analysis_fingerprint": "sha256:...",
      "evidence": {
        "cited_sony_seconds": [130.42],
        "sony_bounds_s": [128.5, 131.2],
        "iphone_bounds_s": [133.1, 138.56],
        "sony_clip": {
          "bundle_relative_path": "events/e-001/sony.mp4",
          "review_url": "/media/events/e-001/sony.mp4",
          "sha256": "..."
        },
        "iphone_clip": {
          "bundle_relative_path": "events/e-001/iphone.mp4",
          "review_url": "/media/events/e-001/iphone.mp4",
          "sha256": "..."
        }
      },
      "video_metrics": {
        "entry_speed_norm": 7.8,
        "torso_rotation_2d_dps": 1068.6,
        "hip_height_change_norm": null,
        "stance_width_norm": null,
        "movement_intensity_0_100": 75.1
      },
      "locked_at": "2026-08-01T20:04:00+02:00",
      "training_eligible": true
    }
  ]
}
```

Only the active analysis round of each event can produce a training example.
If event bounds change, the old assessment stays in the audit file but is no
longer paired with a newly generated clip at the same event path. This avoids
combining an old label with different video content. Historical training media
can be supported later only by versioning clips per event revision or fingerprint.

The separate audit file contains complete assessment snapshots:

```json
{
  "schema_version": 1,
  "session_id": "trainer-ai-session",
  "generation_id": "0123456789abcdef0123456789abcdef",
  "generated_at": "2026-08-01T20:05:00+02:00",
  "assessments": [
    {
      "event_id": "e-001",
      "event_revision": 1,
      "analysis_fingerprint": "sha256:...",
      "assessment_revision": 2,
      "assessment_phase": "post_ai_correction",
      "source_trainer_revision": 3,
      "trainer_name": "Ime trenera",
      "wrestler_name": "Dusan",
      "visibility_status": "dovoljno_vidljivo",
      "throw_name": "Tai-otoshi",
      "score_1_5": 5,
      "reasoning": "Naknadna korekcija trenera.",
      "cited_sony_seconds": [130.7],
      "locked_at": "2026-08-01T20:06:00+02:00",
      "training_eligible": false,
      "ineligibility_reasons": ["post_ai_correction"]
    }
  ]
}
```

The audit contains every normal-event trainer revision, including insufficiently
visible and pre-AI records, in deterministic event/round/revision order. It is
an audit and research artifact, never a direct LoRA input.

## Training Eligibility

A record appears in `training_examples` only when all conditions hold:

- it belongs to a normal, non-injury event;
- visibility is `dovoljno_vidljivo`;
- `assessment_revision` is 1 and phase is `pre_ai`;
- trainer and wrestler names were snapshotted;
- technique, score, reasoning, and at least one in-range Sony citation exist;
- both event clips are listed in the privacy-verified derived-media manifest;
- the assessment matches the event's currently active revision and fingerprint.

Post-AI corrections appear only in `trener_assessment_audit.json` with
`training_eligible: false` and `post_ai_correction` in
`ineligibility_reasons`. AI scores, AI reasons, and trainer reactions to AI are
excluded from `trener_dataset.json` to prevent target leakage and model
anchoring. They remain in canonical review data for separate research.

## Media And Privacy

The dataset references only derived event clips; it never references original
Sony or iPhone files. Every referenced clip must have a matching manifest row
with `privacy_verified: true` and no failure reason. Missing or failed private
media makes that assessment ineligible for multimodal training but does not
delete its audit history.

Every clip object distinguishes its path inside the immutable generation bundle
from its browser URL. `bundle_relative_path` is resolved relative to the folder
containing the dataset and must be a relative POSIX path without parent
traversal. `review_url` works only while that generation is active and must be a
root-relative URL beginning with `/media/`; it rejects schemes, protocol-relative
`//` values, backslashes, query/fragment components, and traversal.
`generation_id` and the clip SHA-256 bind exported labels to exact media bytes.
The JSON contains references, not embedded video bytes.

## Atomic Publication

`trener_dataset.json` and `trener_assessment_audit.json` are generated from the
same in-memory review snapshot as `review.json`,
`analysis/event_metrics.json`, `izvestaj.csv`, and `izvestaj.md`.
`GenerationStore` assigns the generation ID, stages and fsyncs all six artifacts,
then switches the generation pointer. Every post-feature generation requires
both JSON exports. Readers must never observe names, assessments, media, and
exports from different generations.

Legacy generations containing only the original four artifacts remain readable.
The JSON export endpoints return 404 for such a generation. Its next successful
mutation publishes a complete six-artifact generation.

The server exposes:

- `PUT /api/session/participants` for validated participant metadata;
- `GET /trener_dataset.json` for the active generation's clean export;
- `GET /trener_assessment_audit.json` for its separate audit export.

The raw canonical `review.json` remains inaccessible through the browser.

## Validation And Errors

- Names are trimmed, must be non-empty, and may not exceed 120 characters.
- Payloads reject unknown fields and non-string names.
- An assessment lock without saved identities returns a Serbian-Latin 400
  error and leaves the active generation untouched.
- Duplicate or inconsistent phase/revision combinations fail validation.
- Global internal trainer revision duplicates across events fail validation.
- Every post-AI correction must be locked at or after the matching AI reveal.
- An event with any locked trainer assessment cannot be deleted. Its bounds may
  still be revised, preserving old assessments in the audit while invalidating
  them for clean training export.
- JSON is UTF-8, deterministic, finite-number-only, and ends with a newline.
- Dataset generation failure aborts the complete generation activation.

## Compatibility

Existing sessions without `participants` still load and can play videos. The
UI asks for both names before the next assessment. Existing reports remain
available; they may add session identity columns, but their current assessment
and AI-redaction behavior does not change.

Current internal Serbian field names remain stable. The new dataset uses
English ASCII keys because it is a machine-learning interchange contract.

## Verification

Automated tests must cover:

- participant API validation, persistence, reload, and unknown-field rejection;
- requirement to save both identities before locking a new assessment;
- identity snapshot immutability after session metadata changes;
- deletion rejection for every event containing trainer assessment history;
- local `assessment_revision` numbering per analysis round;
- revision 1/pre-AI and later/post-AI phase validation;
- event reanalysis resetting the local assessment revision to 1;
- event reanalysis excluding old labels from clean export rather than pairing
  them with newly generated clips;
- exclusion of injury, insufficient-visibility, incomplete, and unverified-media
  records from `training_examples`;
- inclusion of all revisions and ineligibility reasons in the separate audit;
- absence of AI values from clean training examples;
- validation of AI reveal time before every post-AI correction;
- global uniqueness of source trainer revisions;
- generation ID and SHA-256 binding for both derived clips;
- atomic activation and download of both JSON exports;
- legacy-generation 404 behavior for absent JSON exports;
- UI entry, reload, required-state, and JSON download behavior;
- complete Python test suite and browser smoke test on desktop and mobile widths.

An independent Terra review checks the implementation and generated example
for schema consistency, privacy boundaries, leakage, and revision correctness.
