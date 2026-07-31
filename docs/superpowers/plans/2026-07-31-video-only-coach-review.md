# Video-Only Coach Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Serbian-Latin coach-review workflow that synchronizes the Sony and iPhone videos with two confirmed tap anchors, extracts comparative video-motion metrics for the blue judoka, proposes voice labels, and exports reviewable clips and reports.

**Architecture:** Keep processing in focused Python pipeline modules and make `review.json` the single persisted session contract. A standard-library local HTTP server serves the selected session and a dependency-free browser UI; ffmpeg creates media clips and the side-by-side MP4. The app treats auto-detected anchors, pose tracks, voice labels, and movement events as suggestions until confirmed in the review UI.

**Tech Stack:** Python 3.11+, standard `unittest`, NumPy, SciPy, OpenCV, Ultralytics YOLO11 pose, ffmpeg/ffprobe, OpenAI Whisper CLI when installed, HTML/CSS/vanilla JavaScript, `http.server`.

## Global Constraints

- Use Sony as the master timeline; iPhone time maps as `sony_s = slope * iphone_s + intercept`.
- Require two user-confirmed triple-tap anchors before producing a side-by-side video.
- Use 17 standard COCO pose points; do not present a 16-point model.
- All visible UI and generated CSV/Markdown headings use Serbian Latin.
- Video-only metrics are normalized comparison measures, never force, power, physical acceleration, watts, or medical conclusions.
- The reported-injury event is retained for traceability with `iskljuceno_iz_statistike: true` and excluded from normal metrics/labels.
- Bind the server to `127.0.0.1`; do not upload videos or annotations.
- Never overwrite or delete source videos.
- Preserve unrelated current worktree changes, including hardware firmware edits and `reference_assets/`.

---

### Task 1: Create the Review Contract and Two-Anchor Time Mapping

**Files:**
- Create: `pipeline/video_review_contract.py`
- Create: `pipeline/video_sync.py`
- Create: `tests/__init__.py`
- Create: `tests/test_video_sync.py`

**Interfaces:**
- Consumes: raw Sony/iPhone timestamps and manually confirmed anchor pairs.
- Produces: `AnchorPair`, `TimeMap`, `ReviewEvent`, `ReviewSession`, `fit_time_map()`, `map_iphone_to_sony()`, `validate_review_session()`.

- [ ] **Step 1: Write failing contract and time-map tests**

```python
# tests/test_video_sync.py
import unittest
from pipeline.video_review_contract import AnchorPair, ReviewEvent, ReviewSession
from pipeline.video_sync import fit_time_map, map_iphone_to_sony


class VideoSyncTests(unittest.TestCase):
    def test_two_anchors_create_affine_iphone_to_sony_mapping(self):
        anchors = [
            AnchorPair("pocetak", sony_s=10.0, iphone_s=30.0),
            AnchorPair("kontrola", sony_s=110.4, iphone_s=130.0),
        ]
        time_map = fit_time_map(anchors)
        self.assertAlmostEqual(map_iphone_to_sony(80.0, time_map), 60.2, places=6)

    def test_rejects_duplicate_or_reversed_anchor_times(self):
        with self.assertRaises(ValueError):
            fit_time_map([
                AnchorPair("pocetak", 10.0, 30.0),
                AnchorPair("kontrola", 10.0, 130.0),
            ])

    def test_injury_event_is_excluded_from_normal_statistics(self):
        session = ReviewSession(
            session_id="demo",
            sony_video="sony.mp4",
            iphone_video="iphone.mov",
            anchors=[AnchorPair("pocetak", 10.0, 30.0), AnchorPair("kontrola", 110.0, 130.0)],
            injury_cutoff_s=126.0,
            events=[ReviewEvent("e-1", 100.0, 105.0), ReviewEvent("e-2", 124.0, 126.0, prijavljen_povredni_dogadjaj=True)],
        )
        self.assertEqual([event.event_id for event in session.normal_events()], ["e-1"])
```

- [ ] **Step 2: Run the failing tests**

Run: `python -m unittest tests.test_video_sync -v`  
Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement typed, JSON-serializable review entities**

```python
# pipeline/video_review_contract.py
@dataclass(frozen=True)
class AnchorPair:
    name: str
    sony_s: float
    iphone_s: float

@dataclass
class ReviewEvent:
    event_id: str
    sony_start_s: float
    sony_end_s: float
    prijavljen_povredni_dogadjaj: bool = False
    iskljuceno_iz_statistike: bool = False

    def __post_init__(self) -> None:
        if self.sony_end_s <= self.sony_start_s:
            raise ValueError("event end must be after start")
        if self.prijavljen_povredni_dogadjaj:
            self.iskljuceno_iz_statistike = True

@dataclass
class ReviewSession:
    session_id: str
    sony_video: str
    iphone_video: str
    anchors: list[AnchorPair]
    injury_cutoff_s: float
    events: list[ReviewEvent]

    def normal_events(self) -> list[ReviewEvent]:
        return [event for event in self.events if not event.iskljuceno_iz_statistike]
```

```python
# pipeline/video_sync.py
@dataclass(frozen=True)
class TimeMap:
    slope: float
    intercept: float

def fit_time_map(anchors: Sequence[AnchorPair]) -> TimeMap:
    if len(anchors) != 2:
        raise ValueError("exactly two confirmed anchors are required")
    first, second = sorted(anchors, key=lambda item: item.iphone_s)
    iphone_delta = second.iphone_s - first.iphone_s
    sony_delta = second.sony_s - first.sony_s
    if iphone_delta <= 0 or sony_delta <= 0:
        raise ValueError("anchors must advance in both videos")
    slope = sony_delta / iphone_delta
    return TimeMap(slope=slope, intercept=first.sony_s - slope * first.iphone_s)

def map_iphone_to_sony(iphone_s: float, time_map: TimeMap) -> float:
    return time_map.slope * iphone_s + time_map.intercept
```

- [ ] **Step 4: Add canonical JSON conversion and validation**

Implement `to_dict()`/`from_dict()` only with JSON scalar/list/dict values.
`validate_review_session()` must reject anchors outside source durations, an
injury cutoff before the first anchor, normal events crossing the cutoff, and
an injury event missing `iskljuceno_iz_statistike`.

- [ ] **Step 5: Run tests and static syntax checks**

Run: `python -m unittest tests.test_video_sync -v`  
Expected: PASS.

Run: `python -m py_compile pipeline/video_review_contract.py pipeline/video_sync.py`  
Expected: exit code 0.

- [ ] **Step 6: Commit the contract layer**

```bash
git add pipeline/video_review_contract.py pipeline/video_sync.py tests/__init__.py tests/test_video_sync.py
git commit -m "feat: add video review session contract"
```

### Task 2: Detect Sync Candidates and Create Voice Label Suggestions

**Files:**
- Create: `pipeline/video_review_audio.py`
- Create: `pipeline/voice_labels.py`
- Create: `tests/test_video_review_audio.py`
- Create: `tests/test_voice_labels.py`

**Interfaces:**
- Consumes: video paths, tap-like audio peaks, optional Whisper JSON transcript, normalized event windows.
- Produces: `find_tap_triplet_candidates()`, `TranscriptWord`, `TechniqueSuggestion`, `suggest_techniques()`.

- [ ] **Step 1: Write failing tap-cluster and transcript-association tests**

```python
# tests/test_video_review_audio.py
from pipeline.video_review_audio import find_tap_triplet_candidates

def test_finds_three_transients_with_unequal_but_short_gaps():
    candidates = find_tap_triplet_candidates([10.00, 10.27, 10.73, 18.0, 19.0])
    assert candidates == [(10.00, 10.27, 10.73)]
```

```python
# tests/test_voice_labels.py
from pipeline.voice_labels import TranscriptWord, suggest_techniques

def test_assigns_nearby_spoken_technique_to_event():
    words = [
        TranscriptWord("radimo", 4.0, 4.4),
        TranscriptWord("o-soto-gari", 4.5, 5.1),
    ]
    suggestions = suggest_techniques(words, [("e-1", 5.0, 8.0)])
    assert suggestions["e-1"].predlog_tehnike == "O-soto-gari"

def test_leaves_event_blank_when_no_vocabulary_match_is_nearby():
    assert suggest_techniques([TranscriptWord("pozdrav", 0.0, 0.5)], [("e-1", 8.0, 10.0)])["e-1"].predlog_tehnike is None
```

- [ ] **Step 2: Run the failing tests**

Run: `python -m unittest tests.test_video_review_audio tests.test_voice_labels -v`  
Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement deterministic tap-triplet grouping**

```python
def find_tap_triplet_candidates(
    peak_times_s: Sequence[float],
    min_gap_s: float = 0.12,
    max_gap_s: float = 0.85,
    max_span_s: float = 1.4,
) -> list[tuple[float, float, float]]:
    ordered = sorted(float(value) for value in peak_times_s)
    candidates = []
    for first, second, third in zip(ordered, ordered[1:], ordered[2:]):
        if min_gap_s <= second-first <= max_gap_s and min_gap_s <= third-second <= max_gap_s and third-first <= max_span_s:
            candidates.append((first, second, third))
    return candidates
```

Use the existing `pipeline.audio_sync.extract_audio_envelope()` and
`scipy.signal.find_peaks()` to obtain peaks from a user-selected time range.
Store candidate confidence separately from confirmation state; no candidate
may become a final anchor automatically.

- [ ] **Step 4: Implement transcript parsing and suggestion matching**

Parse Whisper JSON `segments[].words[]`; fall back to segment timestamps when
word timestamps are absent. Normalize Serbian/English spelling variants using
an editable vocabulary mapping such as `"osoto gari" -> "O-soto-gari"` and
`"seoi nage" -> "Seoi-nage"`. Match only terms with a midpoint inside
`[event_start_s - 8, event_end_s + 3]`; choose the closest term and retain its
source phrase and a confidence value. `shutil.which("whisper")` decides whether
the CLI is available. On absence, return an empty transcript and a warning.

- [ ] **Step 5: Run focused tests**

Run: `python -m unittest tests.test_video_review_audio tests.test_voice_labels -v`  
Expected: PASS.

- [ ] **Step 6: Commit audio suggestion modules**

```bash
git add pipeline/video_review_audio.py pipeline/voice_labels.py tests/test_video_review_audio.py tests/test_voice_labels.py
git commit -m "feat: add tap anchors and voice label suggestions"
```

### Task 3: Extract Blue-Athlete Pose Metrics and Suggested Movement Windows

**Files:**
- Create: `pipeline/video_pose_metrics.py`
- Create: `pipeline/video_event_detection.py`
- Create: `tests/test_video_pose_metrics.py`
- Create: `tests/test_video_event_detection.py`

**Interfaces:**
- Consumes: tracked 17-point COCO keypoints for a coach-confirmed blue athlete, `fps`, and Sony-relative timestamps.
- Produces: `FrameMetric`, `EventMetrics`, `compute_pose_metrics()`, `suggest_event_windows()`.

- [ ] **Step 1: Write failing metric tests using synthetic 17-point poses**

```python
# tests/test_video_pose_metrics.py
import numpy as np
from pipeline.video_pose_metrics import compute_pose_metrics

def pose(hip_x, hip_y, left_shoulder, right_shoulder):
    keypoints = np.zeros((17, 3), dtype=float)
    keypoints[:, 2] = 1.0
    keypoints[11, :2] = [hip_x - 5, hip_y]
    keypoints[12, :2] = [hip_x + 5, hip_y]
    keypoints[5, :2] = left_shoulder
    keypoints[6, :2] = right_shoulder
    return keypoints

def test_entry_speed_is_torso_normalized():
    frames = [pose(0, 100, (-5, 80), (5, 80)), pose(10, 100, (5, 80), (15, 80))]
    metrics = compute_pose_metrics(frames, fps=10.0)
    assert metrics[1].brzina_ulaska_norm_s == 5.0
```

```python
# tests/test_video_event_detection.py
from pipeline.video_event_detection import suggest_event_windows

def test_merges_adjacent_motion_samples_into_one_window():
    windows = suggest_event_windows([0.0, 0.1, 0.9, 1.1, 0.0], fps=10.0, threshold=0.5)
    assert windows == [(0.1, 0.4)]
```

- [ ] **Step 2: Run the failing tests**

Run: `python -m unittest tests.test_video_pose_metrics tests.test_video_event_detection -v`  
Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement transparent video-only pose measures**

For every frame, calculate hip midpoint, shoulder midpoint, torso length,
shoulder-line angle, stance width, and a visibility flag. Normalize all
distances by torso length. Derive finite differences using actual `fps`:

```python
entry_speed = np.linalg.norm(hip_midpoint[t] - hip_midpoint[t - 1]) / torso_length[t] * fps
rotation_2d_dps = np.degrees(wrap_angle(shoulder_angle[t] - shoulder_angle[t - 1])) * fps
hip_level_norm = (hip_midpoint[t][1] - hip_midpoint[0][1]) / torso_length[t]
```

If either shoulder or hip pair has confidence below `0.3`, emit `None` for
that frame instead of inventing a value. Use interpolation only inside gaps of
at most five frames, then flag the metric as interpolated.

- [ ] **Step 4: Implement blue-athlete selection and event suggestions**

Use a user-provided seed bounding box on one clear Sony frame. Select the
matching YOLO person by largest IoU, then retain the compatible ByteTrack ID.
When the ID changes after occlusion, choose the nearest compatible pose with a
blue-dominant HSV torso patch; otherwise mark the gap `nedovoljno_vidljivo`.

Combine normalized entry speed and absolute rotation into a smoothed motion
energy. Threshold it into windows, expand each window by 1.0 second on both
sides, merge windows with less than 1.5 seconds between them, and clip at the
confirmed injury cutoff. The injury event is created separately and excluded
from these normal windows.

- [ ] **Step 5: Run focused tests**

Run: `python -m unittest tests.test_video_pose_metrics tests.test_video_event_detection -v`  
Expected: PASS.

- [ ] **Step 6: Commit pose metrics and segmentation**

```bash
git add pipeline/video_pose_metrics.py pipeline/video_event_detection.py tests/test_video_pose_metrics.py tests/test_video_event_detection.py
git commit -m "feat: add video pose metrics and event suggestions"
```

### Task 4: Build the Session Importer and Media Exports

**Files:**
- Modify: `pipeline/clip_extractor.py`
- Create: `pipeline/video_review_import.py`
- Create: `tools/video_review.py`
- Create: `tests/test_video_review_import.py`

**Interfaces:**
- Consumes: Sony/iPhone paths, two confirmed anchors, injury cutoff, selected blue-athlete seed, optional transcript JSON.
- Produces: `import_session(sony, iphone, output_dir, anchors, injury_cutoff_s, blue_seed, transcript_path) -> Path`, `review.json`, per-event Sony/iPhone clips, anchor previews, `session_side_by_side.mp4`, and an import summary.

- [ ] **Step 1: Write failing importer tests with mocked ffmpeg calls**

```python
# tests/test_video_review_import.py
from pathlib import Path
from unittest.mock import patch
from pipeline.video_review_import import make_side_by_side, write_review_json

@patch("pipeline.video_review_import.subprocess.run")
def test_side_by_side_maps_iphone_with_affine_setpts(mock_run, tmp_path):
    make_side_by_side(Path("sony.mp4"), Path("iphone.mov"), slope=1.002, intercept=-19.4, end_s=126.0, output=tmp_path / "out.mp4")
    command = mock_run.call_args.args[0]
    assert "setpts=1.002*PTS-19.4/TB" in " ".join(command)

def test_review_json_is_written_atomically(tmp_path):
    output = write_review_json(tmp_path, {"session_id": "demo", "events": []})
    assert output.name == "review.json"
    assert output.read_text(encoding="utf-8").startswith("{")
```

- [ ] **Step 2: Run the failing importer tests**

Run: `python -m unittest tests.test_video_review_import -v`  
Expected: FAIL because the importer does not exist.

- [ ] **Step 3: Add a generic dual-input side-by-side exporter**

Add `make_side_by_side()` rather than embedding the command in the CLI. Build
an ffmpeg `filter_complex` with Sony trimmed to `[0, end_s]`; transform iPhone
timestamps using the confirmed affine map, scale each input to equal height,
and `hstack`. Keep Sony audio only. Fail before invoking ffmpeg when `end_s <=
0` or the time-map slope is nonpositive.

- [ ] **Step 4: Implement importer CLI and idempotent output layout**

```bash
python tools/video_review.py import \
  --sony /absolute/C0007.MP4 \
  --iphone /absolute/IMG_3852.mov \
  --session-dir sessions/2026-07-31-demo \
  --anchors-json anchors.json \
  --injury-cutoff-sony-s 126.0 \
  --blue-seed-sony 1280,720,480,900
```

The command must create `media/`, `events/`, `previews/`, `analysis/`, and
`review.json`. It must refuse a new import when `review.json` already contains
trainer annotations unless `--force-reimport` is supplied. Reuse
`cut_clip()` for frame-accurate source-relative event clips and call the pose
modules only for the usable range.

- [ ] **Step 5: Add size/duration verification**

After every ffmpeg export, use `probe_duration()` and reject a zero-byte file
or a result whose duration differs by more than 0.75 seconds from the expected
window. Record absolute source paths and SHA-256 hashes in `review.json`.

- [ ] **Step 6: Run importer tests and command help**

Run: `python -m unittest tests.test_video_review_import -v`  
Expected: PASS.

Run: `python tools/video_review.py import --help`  
Expected: usage text with the required source, session, anchors, cutoff, and seed arguments.

- [ ] **Step 7: Commit import/export workflow**

```bash
git add pipeline/clip_extractor.py pipeline/video_review_import.py tools/video_review.py tests/test_video_review_import.py
git commit -m "feat: add video review session importer"
```

### Task 5: Implement the Serbian-Latin Coach Review Server and Browser UI

**Files:**
- Create: `coach_app/__init__.py`
- Create: `coach_app/server.py`
- Create: `coach_app/static/index.html`
- Create: `coach_app/static/app.js`
- Create: `coach_app/static/styles.css`
- Create: `tests/test_coach_server.py`

**Interfaces:**
- Consumes: one imported session directory containing `review.json` and media beneath it.
- Produces: `create_server(session_dir, port) -> ReviewServer`, `save_annotation(review_path, event_id, payload) -> dict`, loopback HTTP review app; `GET /api/session`, `GET /api/events/<event_id>`, `PUT /api/events/<event_id>/annotation`, `POST /api/session/sync`, and contained `/media/` files.

- [ ] **Step 1: Write failing API and persistence tests**

```python
# tests/test_coach_server.py
import json
import tempfile
import unittest
from pathlib import Path
from urllib.request import Request, urlopen
from coach_app.server import create_server

class CoachServerTests(unittest.TestCase):
    def test_annotation_persists_and_regenerates_csv_and_markdown(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "review.json").write_text(json.dumps({"session_id":"demo", "events":[{"event_id":"e-1", "sony_start_s":1, "sony_end_s":2}]}), encoding="utf-8")
            server = create_server(root, port=0)
            thread = server.start_in_thread()
            try:
                body = json.dumps({"potvrdena_tehnika":"O-soto-gari", "ocena":4, "napomena":"Stabilan ulaz."}).encode()
                request = Request(server.base_url + "/api/events/e-1/annotation", data=body, method="PUT", headers={"Content-Type":"application/json"})
                self.assertEqual(urlopen(request).status, 200)
            finally:
                server.shutdown()
                thread.join(timeout=2)
            saved = json.loads((root / "review.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["events"][0]["potvrdena_tehnika"], "O-soto-gari")
            self.assertTrue((root / "izvestaj.csv").exists())
            self.assertTrue((root / "izvestaj.md").exists())

    def test_media_path_cannot_escape_session_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "review.json").write_text(json.dumps({"session_id":"demo", "events":[]}), encoding="utf-8")
            server = create_server(root, port=0)
            thread = server.start_in_thread()
            try:
                with self.assertRaises(Exception) as raised:
                    urlopen(server.base_url + "/media/%2e%2e/%2e%2e/etc/passwd")
                self.assertEqual(raised.exception.code, 404)
            finally:
                server.shutdown()
                thread.join(timeout=2)
```

- [ ] **Step 2: Run failing server tests**

Run: `python -m unittest tests.test_coach_server -v`  
Expected: FAIL because `coach_app.server` does not exist.

- [ ] **Step 3: Implement local-only server and atomic annotation updates**

Use `ThreadingHTTPServer(("127.0.0.1", port), Handler)`. `create_server()`
returns a `ReviewServer` with `base_url`, `start_in_thread()`, and `shutdown()`.
Resolve every media
path with `Path.resolve()` and reject it unless it remains under the selected
session directory. Validate annotation requests with this exact shape:

```json
{
  "potvrdena_tehnika": "O-soto-gari",
  "ocena": 4,
  "napomena": "Ulaz je stabilan, kukovi mogu niže."
}
```

Write `review.json.tmp`, `fsync`, then `replace()` it. Regenerate `izvestaj.csv`
and `izvestaj.md` after every successful save. `POST /api/session/sync` accepts
two confirmed anchors and an injury cutoff, validates them through Task 1, and
updates the time map without permitting a cutoff after an injury-marked event.

- [ ] **Step 4: Build the Serbian-Latin review interface**

Use a dense desktop work surface: event list on the left, two equal video
views at the top right, metric charts below, and annotation fields beneath the
charts. Implement synchronized play/pause/seek by translating current Sony
time with the inverse `TimeMap`. Provide frame-step buttons that move by the
actual source FPS from `review.json`.

Use these literal labels: `Događaji`, `Sinhronizacija`, `Predlog tehnike`,
`Potvrđena tehnika`, `Ocena`, `Napomena`, `Sačuvaj`, `Podeli`, `Spoji`,
`Obriši`, `Izvezi izveštaj`, `Prijavljen povredni događaj`, and
`Nedovoljno vidljivo`. Render the injury event in a restrained warning state
and disable normal score/technique controls for it.

Draw five responsive Canvas charts from the event metric arrays. Clicking a
chart seeks the shared Sony time cursor; all series and cursor remain within
the canvas bounds after resize.

- [ ] **Step 5: Run server tests and a manual smoke check**

Run: `python -m unittest tests.test_coach_server -v`  
Expected: PASS.

Run: `python tools/video_review.py serve --session-dir sessions/2026-07-31-demo --port 8765`  
Expected: `http://127.0.0.1:8765` starts and serves the session page.

- [ ] **Step 6: Commit the review app**

```bash
git add coach_app tests/test_coach_server.py tools/video_review.py
git commit -m "feat: add Serbian coach review app"
```

### Task 6: Verify the Actual Session and Document Operator Workflow

**Files:**
- Create: `docs/VIDEO_ONLY_REVIEW_WORKFLOW.md`
- Modify: `README.md`
- Create: `tests/test_video_review_e2e.py`

**Interfaces:**
- Consumes: a compact synthetic session fixture and the supplied Sony/iPhone source paths supplied at runtime.
- Produces: repeatable end-to-end verification and operator instructions for the real session.

- [ ] **Step 1: Write a synthetic end-to-end test**

```python
# tests/test_video_review_e2e.py
import json
from pathlib import Path
from unittest.mock import patch
from pipeline.video_review_import import import_session
from coach_app.server import save_annotation

@patch("pipeline.video_review_import.run_pose_analysis")
@patch("pipeline.video_review_import.make_side_by_side")
@patch("pipeline.video_review_import.cut_clip")
def test_import_then_annotation_creates_all_review_outputs(mock_cut, mock_composite, mock_pose, tmp_path):
    mock_pose.return_value = [{"event_id":"e-1", "sony_start_s":11.0, "sony_end_s":14.0, "metrics":{}}]
    mock_cut.side_effect = lambda video, start, end, output, **kwargs: Path(output).touch() or Path(output)
    mock_composite.side_effect = lambda *args, output, **kwargs: Path(output).touch() or Path(output)
    review_path = import_session(
        sony=Path("sony.mp4"), iphone=Path("iphone.mov"), output_dir=tmp_path,
        anchors=[{"name":"pocetak","sony_s":10.0,"iphone_s":30.0},{"name":"kontrola","sony_s":110.0,"iphone_s":130.0}],
        injury_cutoff_s=126.0, blue_seed=(1, 2, 3, 4), transcript_path=None,
    )
    save_annotation(review_path, "e-1", {"potvrdena_tehnika":"O-soto-gari", "ocena":4, "napomena":"Dobar ulaz."})
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert review["events"][0]["potvrdena_tehnika"] == "O-soto-gari"
    assert (tmp_path / "izvestaj.csv").exists()
    assert (tmp_path / "izvestaj.md").exists()
    assert (tmp_path / "media" / "session_side_by_side.mp4").exists()
```

- [ ] **Step 2: Run the failing end-to-end test**

Run: `python -m unittest tests.test_video_review_e2e -v`  
Expected: FAIL until Tasks 1-5 are connected through the importer and server.

- [ ] **Step 3: Complete the test with explicit boundary mocks**

Mock only subprocess media execution, YOLO inference, and Whisper invocation;
exercise the real time mapping, contract validation, event exclusion,
annotation persistence, report generation, and HTTP request handling.

- [ ] **Step 4: Run all automated tests**

Run: `python -m unittest discover -s tests -p "test_*.py" -v`  
Expected: PASS for all Python tests. Compile the C++ battery test separately
only if its user-owned firmware changes remain compatible.

- [ ] **Step 5: Process the supplied videos with a controlled review checkpoint**

1. Copy or reference `/Volumes/Untitled/PRIVATE/M4ROOT/CLIP/C0007.MP4` and `/Users/pauldudko/Downloads/IMG_3852.mov` without modifying them.
2. Generate previews around both detected triplet candidates and the white-athlete kneeling cutoff.
3. Confirm the blue athlete, both anchors, and cutoff in the browser before long YOLO processing.
4. Run import, review the side-by-side MP4, and confirm one visible throw is aligned in both views.
5. Launch the server on the first free local port and verify Serbian-Latin labels, chart seek, save, CSV, and Markdown export.

- [ ] **Step 6: Write concise operator instructions**

Document the actual commands, required manual confirmations, meaning and
limits of every video metric, how to repair a failed sync candidate, and why
the injury-marked event remains excluded. Link this workflow from `README.md`.

- [ ] **Step 7: Commit verification and documentation**

```bash
git add docs/VIDEO_ONLY_REVIEW_WORKFLOW.md README.md tests/test_video_review_e2e.py
git commit -m "docs: add video-only review workflow"
```

## Plan Self-Review

### Spec coverage

- Two confirmed anchors and affine sync: Task 1, Task 2, Task 4, Task 5.
- Injury cutoff and exclusion: Task 1, Task 4, Task 5, Task 6.
- 17-point blue-athlete tracking and comparative metrics: Task 3.
- Voice-derived Serbian technique suggestions: Task 2 and Task 4.
- Side-by-side MP4 plus independent browser players: Task 4 and Task 5.
- Serbian Latin UI/reports and trainer editing: Task 5 and Task 6.
- Local-only, source-preserving safety and errors: Tasks 1, 4, and 5.
- Automated and manual verification: every task, with end-to-end coverage in Task 6.

### Consistency check

All later tasks consume `ReviewSession`, `ReviewEvent`, `AnchorPair`, and
`TimeMap` from Task 1. iPhone-to-Sony mapping uses the same affine direction
in exports and browser playback. Injury events have one contract flag and are
excluded through that flag everywhere. No task calls a video-derived value
force, power, acceleration, or a medical result.
