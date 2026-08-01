# Trainer Dataset Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add immutable trainer/wrestler identity capture and publish a privacy-bound clean LoRA JSON dataset plus a separate complete trainer assessment audit.

**Architecture:** The canonical v3 review remains the source of truth. `TrainerAiService` snapshots participant identities into each locked assessment, a new pure export module derives local assessment revisions and eligibility, and `GenerationStore` writes both JSON exports into the same immutable generation as review data and media before switching the pointer. The browser edits session participants through one strict API and downloads only generated exports, never raw `review.json`.

**Tech Stack:** Python 3 standard library, `unittest`, existing `ThreadingHTTPServer`, vanilla HTML/CSS/JavaScript, Playwright browser QA, immutable generation bundles.

## Global Constraints

- `assessment_revision` is one-based within `(event_id, event_revision, analysis_fingerprint)`, sorted by global internal `revizija`.
- `assessment_revision == 1` requires internal `faza == "pre_ai"`; every later local revision requires `faza == "post_ai_korekcija"` and exports as `post_ai_correction`.
- Every post-AI correction requires a matching revealed AI evaluation and `zakljucano_u >= ai_otkriven_u`.
- Only the active event analysis round can enter `trener_dataset.json`; historical and post-AI rows belong only to `trener_assessment_audit.json`.
- Clean training examples contain no AI score, reason, evidence, feedback, or trainer reaction to AI.
- Trainer and wrestler names are required, trimmed, at most 120 characters, and snapshotted into every new assessment; the current session defaults the wrestler UI to `Dusan` only when no value is saved.
- Legacy assessments without identity snapshots remain permanently ineligible unless a future explicit provenance migration is implemented.
- Only privacy-verified derived event clips are referenced. Each reference carries bundle-relative path, `/media/...` review URL, SHA-256, and active generation ID.
- Events with trainer assessment history cannot be deleted.
- Legacy four-artifact generations remain readable; missing dataset/audit downloads return 404. Every new generation atomically publishes all six artifacts.
- UI copy is Serbian Latin. Dataset keys are English ASCII. Existing AI-redaction behavior must remain unchanged.
- Every task uses red-green-refactor, commits independently, and receives a fresh Terra review before the next task.

---

### Task 1: Participant Identity And Assessment Invariants

**Files:**
- Modify: `pipeline/trainer_ai_state.py`
- Modify: `pipeline/video_review_contract.py`
- Modify: `coach_app/trainer_ai_service.py`
- Modify: `coach_app/event_editor.py`
- Test: `tests/test_trainer_ai_state.py`
- Test: `tests/test_coach_trainer_ai.py`
- Test: `tests/test_coach_event_editor.py`

**Interfaces:**
- Produces: `validate_participants(value: object, *, required: bool = False) -> dict[str, str] | None`
- Produces: `TrainerAiService.save_participants(payload: object) -> dict[str, Any]`
- Produces: identity snapshots in `trener_procene[]` as `trainer_name` and `wrestler_name`.
- Preserves: session-wide internal `revizija` while validating global uniqueness in `validate_review_payload`.

- [ ] **Step 1: Write failing state-validation tests**

Add tests proving exact-round local ordering, post-AI timing, participant validation, and cross-event global revision uniqueness:

```python
def test_validation_groups_assessment_phase_by_revision_and_fingerprint(self):
    review = migrate_trainer_ai_payload(legacy_review_fixture())
    event = review["events"][0]
    first = valid_locked_event()["trener_procene"][0]
    event["trener_procene"] = [first]
    validate_review_payload(review)

def test_validation_rejects_post_ai_assessment_before_reveal(self):
    event = valid_locked_event()
    event["trener_procene"].append({
        **event["trener_procene"][0],
        "revizija": 2,
        "faza": "post_ai_korekcija",
        "zakljucano_u": "2026-08-01T11:59:59+00:00",
    })
    with self.assertRaisesRegex(ValueError, "otkriv"):
        validate_trainer_ai_event(event)

def test_review_rejects_duplicate_global_trainer_revision_across_events(self):
    review = migrate_trainer_ai_payload(legacy_review_fixture())
    # Insert revision 1 into two normal events, then validate the full payload.
    with self.assertRaisesRegex(ValueError, "globalno jedinstvena"):
        validate_review_payload(review)
```

- [ ] **Step 2: Run state tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_trainer_ai_state -v
```

Expected: new tests fail because participant validation, exact-round grouping, reveal-time enforcement, and global uniqueness are absent.

- [ ] **Step 3: Implement participant and revision validation**

Add strict reusable validation:

```python
PARTICIPANT_FIELDS = {"trainer_name", "wrestler_name", "updated_at"}

def validate_participants(value: object, *, required: bool = False) -> dict[str, str] | None:
    if value is None and not required:
        return None
    if not isinstance(value, Mapping) or set(value) != PARTICIPANT_FIELDS:
        raise ValueError("podaci učesnika nemaju tačna obavezna polja")
    # Trim both names, reject empty or >120, validate updated_at via _iso_time.
```

In `_validate_trainer_assessments`, group rows by `(event_revision, fingerprint)`, sort by `revizija`, validate phase by local position, and require reveal/time ordering for every post-AI row. In `validate_review_payload`, collect all trainer `revizija` values across normal events and reject duplicates.

- [ ] **Step 4: Write failing service identity tests**

```python
def test_participants_are_required_and_snapshotted_per_assessment(self):
    with self.assertRaisesRegex(ValueError, "ime trenera"):
        self.service.lock_assessment("e-001", self.visible_assessment())
    saved = self.service.save_participants({
        "trainer_name": "  Marko Markovic  ",
        "wrestler_name": " Dusan ",
    })
    locked = self.service.lock_assessment("e-001", self.visible_assessment())
    self.assertEqual(saved["participants"]["trainer_name"], "Marko Markovic")
    self.assertEqual(locked["assessment"]["wrestler_name"], "Dusan")

def test_later_participant_edit_does_not_rewrite_locked_identity(self):
    self.service.save_participants({"trainer_name": "Marko", "wrestler_name": "Dusan"})
    first = self.service.lock_assessment("e-001", self.visible_assessment())["assessment"]
    self.service.save_participants({"trainer_name": "Jovan", "wrestler_name": "Dusan"})
    self.assertEqual(first["trainer_name"], "Marko")
```

- [ ] **Step 5: Run service tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_coach_trainer_ai.CoachTrainerAiTests.test_participants_are_required_and_snapshotted_per_assessment tests.test_coach_trainer_ai.CoachTrainerAiTests.test_later_participant_edit_does_not_rewrite_locked_identity -v
```

Expected: FAIL because `save_participants` and snapshots do not exist.

- [ ] **Step 6: Implement participant persistence and identity snapshots**

Implement `save_participants` under `mutation_lock`, accepting exactly `trainer_name` and `wrestler_name`, adding `updated_at = self._now_iso()`, validating, activating one new generation, and returning the saved object. Make `lock_assessment` require saved participants and copy both names into the immutable assessment before validation/activation.

- [ ] **Step 7: Write and pass the deletion-guard test**

```python
def test_delete_rejects_event_with_trainer_history(self):
    self.upgrade_to_v3()
    review = self.server.trainer_ai_service.load_review()
    review["events"][0]["trener_procene"] = [{"revizija": 1}]
    self.server.trainer_ai_service.activate_review(review)
    error = self.assert_http_error(409, self.server, "/api/events/e-1", method="DELETE")
    self.assertIn("zaključan", error["error"])
```

Add an `EventConflictError` before deletion when `selected.get("trener_procene")` is non-empty. Run `tests.test_coach_event_editor` and confirm the existing unassessed deletion test still passes.

- [ ] **Step 8: Run focused suites and commit**

```bash
python3 -m unittest tests.test_trainer_ai_state tests.test_coach_trainer_ai tests.test_coach_event_editor -v
git add pipeline/trainer_ai_state.py pipeline/video_review_contract.py coach_app/trainer_ai_service.py coach_app/event_editor.py tests/test_trainer_ai_state.py tests/test_coach_trainer_ai.py tests/test_coach_event_editor.py
git commit -m "feat: capture trainer session identities"
```

---

### Task 2: Clean Dataset And Separate Audit Exporter

**Files:**
- Create: `pipeline/trainer_dataset_export.py`
- Create: `tests/test_trainer_dataset_export.py`

**Interfaces:**
- Consumes: validated canonical review with assessment identity snapshots.
- Produces: `build_trainer_exports(review: Mapping[str, Any], *, generation_id: str, bundle_root: Path, generated_at: str) -> tuple[dict[str, Any], dict[str, Any]]`
- Produces: `render_trainer_exports(...) -> tuple[str, str]`, returning deterministic newline-terminated dataset and audit JSON.

- [ ] **Step 1: Write failing clean-export tests**

Create a fixture with active pre-AI, active post-AI, old-round pre-AI, injury, insufficient visibility, and unverified-media cases. Assert:

```python
dataset, audit = build_trainer_exports(
    review,
    generation_id="a" * 32,
    bundle_root=self.root,
    generated_at="2026-08-02T10:00:00+02:00",
)
self.assertEqual([row["assessment_phase"] for row in dataset["training_examples"]], ["pre_ai"])
self.assertNotIn("ai_score", json.dumps(dataset))
self.assertEqual(
    [row["assessment_revision"] for row in audit["assessments"] if row["event_id"] == "e-001"],
    [1, 2, 1],
)
self.assertIn("post_ai_correction", audit["assessments"][1]["ineligibility_reasons"])
self.assertIn("inactive_analysis_round", audit["assessments"][2]["ineligibility_reasons"])
```

- [ ] **Step 2: Run exporter tests and verify RED**

```bash
python3 -m unittest tests.test_trainer_dataset_export -v
```

Expected: import failure because the export module does not exist.

- [ ] **Step 3: Implement deterministic assessment projection**

Implement helpers that group by `(event_id, event_revision, analysis_fingerprint)`, sort each group by `revizija`, and project:

```python
def _assessment_phase(row: Mapping[str, Any], local_revision: int) -> str:
    expected = "pre_ai" if local_revision == 1 else "post_ai_korekcija"
    if row.get("faza") != expected:
        raise ValueError("faza procene ne odgovara lokalnoj reviziji")
    return "pre_ai" if local_revision == 1 else "post_ai_correction"
```

The audit row contains identity, visibility, throw, score, reasoning, citations, lock time, source/global revision, local revision, event revision/fingerprint, eligibility boolean, and deterministic reason codes.

- [ ] **Step 4: Implement active-round eligibility and leakage boundary**

Only add a row to `training_examples` when it is active round, local revision 1/pre-AI, sufficiently visible, complete, has snapshotted identity, and both event clip manifest rows are unique/privacy verified. Never read `ai_procene`, `imu_eksperimentalno`, `procene_ai_predloga`, or `aktivni_duel` when constructing the clean row.

- [ ] **Step 5: Implement exact media binding**

For each camera, resolve `events/<event_id>/<camera>.mp4` under `bundle_root`, reject traversal/missing files, calculate SHA-256 by streaming 1 MiB chunks, and emit:

```python
{
    "bundle_relative_path": relative,
    "review_url": f"/media/{relative}",
    "sha256": digest,
}
```

Assert the URL starts with exactly `/media/`, has no scheme/query/fragment/backslash/traversal, and the manifest row has `media_type == "event_clip"`, `privacy_verified is True`, and `failure_reason is None`.

- [ ] **Step 6: Implement deterministic rendering and pass tests**

Use `json.dumps(..., ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"`. Run:

```bash
python3 -m unittest tests.test_trainer_dataset_export -v
```

Expected: PASS for clean/audit separation, phase mapping, active-round exclusion, identity rules, privacy, checksums, and finite JSON.

- [ ] **Step 7: Commit**

```bash
git add pipeline/trainer_dataset_export.py tests/test_trainer_dataset_export.py
git commit -m "feat: build clean trainer dataset exports"
```

---

### Task 3: Atomic Generation Publication And HTTP API

**Files:**
- Modify: `coach_app/review_bundle.py`
- Modify: `coach_app/trainer_ai_service.py`
- Modify: `coach_app/server.py`
- Test: `tests/test_coach_trainer_ai.py`

**Interfaces:**
- Extends: `GenerationSnapshot.dataset_path` and `GenerationSnapshot.audit_path`.
- Extends: `GenerationStore.stage_and_activate(..., generated_at: str) -> GenerationSnapshot`.
- Adds: `PUT /api/session/participants`.
- Adds: `GET /trener_dataset.json` and `GET /trener_assessment_audit.json`.

- [ ] **Step 1: Write failing atomic-generation tests**

Assert that a successful new generation contains all six files, both JSON documents share the pointer generation ID, and an exporter failure leaves the old pointer unchanged:

```python
snapshot = store.stage_and_activate(
    review, review["events"], "csv", "markdown",
    generated_at="2026-08-02T10:00:00+02:00",
)
self.assertTrue(snapshot.dataset_path.is_file())
self.assertTrue(snapshot.audit_path.is_file())
self.assertEqual(json.loads(snapshot.dataset_path.read_text())["generation_id"], snapshot.generation_id)
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
python3 -m unittest tests.test_coach_trainer_ai -v
```

Expected: new assertions fail because snapshot paths and staged JSON files do not exist.

- [ ] **Step 3: Publish both exports before pointer switch**

After media link/copy and canonical review write, call `render_trainer_exports` with the assigned UUID, target bundle root, and service-supplied ISO time. Write and fsync both export texts before directory fsync and pointer replacement. Keep `resolve_current` compatible by requiring only the original four files; new endpoints explicitly test export file existence.

- [ ] **Step 4: Make every service activation produce six artifacts**

Pass `generated_at=self._now_iso()` from `TrainerAiService.activate_review`. Ensure participant save, assessment lock, AI reveal, feedback, event edits, and legacy draft annotation all continue through this single activation path.

- [ ] **Step 5: Write failing HTTP tests**

```python
status, body = self.read_json(base + "/api/session/participants", method="PUT", payload={
    "trainer_name": "Marko",
    "wrestler_name": "Dusan",
})
self.assertEqual(status, 200)
with urlopen(base + "/trener_dataset.json") as response:
    self.assertEqual(response.status, 200)
    dataset = json.loads(response.read())
self.assertEqual(dataset["participants"]["wrestler_name"], "Dusan")
```

Also create a legacy four-file generation and assert both JSON endpoints return 404.

- [ ] **Step 6: Implement strict participant and download routes**

Handle `/api/session/participants` before the `/api/events/` prefix in `do_PUT`, calling `save_participants`. In `do_GET`, serve only `snapshot.dataset_path` or `snapshot.audit_path` from `snapshot.root`, and raise `FileNotFoundError` when absent. Keep raw `review.json` blocked.

- [ ] **Step 7: Run focused and regression tests**

```bash
python3 -m unittest tests.test_coach_trainer_ai tests.test_coach_event_editor tests.test_video_review_reports -v
```

Expected: PASS with no old/new generation mixing and unchanged AI redaction.

- [ ] **Step 8: Commit**

```bash
git add coach_app/review_bundle.py coach_app/trainer_ai_service.py coach_app/server.py tests/test_coach_trainer_ai.py
git commit -m "feat: publish trainer exports atomically"
```

---

### Task 4: Coach UI For Participants And JSON Downloads

**Files:**
- Modify: `coach_app/static/index.html`
- Modify: `coach_app/static/app.js`
- Modify: `coach_app/static/styles.css`
- Test: `tests/test_coach_app.py`

**Interfaces:**
- Consumes: public `review.participants`.
- Calls: `PUT /api/session/participants` with exactly two name fields.
- Links: `/trener_dataset.json` and `/trener_assessment_audit.json`.

- [ ] **Step 1: Write failing static-contract tests**

Assert unique IDs and Serbian-Latin labels:

```python
self.assertIn('id="trainer-name"', html)
self.assertIn('id="wrestler-name"', html)
self.assertIn('id="save-participants-button"', html)
self.assertIn('href="/trener_dataset.json"', html)
self.assertIn('href="/trener_assessment_audit.json"', html)
self.assertIn('/api/session/participants', javascript)
```

- [ ] **Step 2: Run static tests and verify RED**

```bash
python3 -m unittest tests.test_coach_app -v
```

Expected: FAIL because controls and routes are absent.

- [ ] **Step 3: Add compact session identity controls**

Add one un-nested panel band above the synchronized video panel with two text inputs, `Sacuvaj podatke`, and two download buttons: `Preuzmi skup (JSON)` and `Preuzmi audit (JSON)`. Set `maxlength="120"`, preserve stable responsive dimensions, and do not introduce cards inside cards.

- [ ] **Step 4: Add client state and save flow**

On `/api/session` load, populate saved values; use `Dusan` only when `wrestler_name` is absent. On save, trim inputs, reject empty values locally, PUT exact JSON, update `state.review.participants`, and show `Podaci sesije su sačuvani`. Disable `Zaključaj procenu` until saved participant metadata exists, without breaking existing injury/visibility/reveal state logic.

- [ ] **Step 5: Add responsive styling and pass static tests**

Use a restrained two-column form on desktop and one column below 620 px. Ensure labels and buttons wrap without overflow. Run:

```bash
python3 -m unittest tests.test_coach_app -v
```

- [ ] **Step 6: Commit**

```bash
git add coach_app/static/index.html coach_app/static/app.js coach_app/static/styles.css tests/test_coach_app.py
git commit -m "feat: add trainer dataset controls"
```

---

### Task 5: End-To-End Session QA And Final Review

**Files:**
- Create: `tests/coach_trainer_dataset_qa.cjs`
- Modify only if a test exposes a defect: files owned by Tasks 1-4, always with a new failing regression test first.

**Interfaces:**
- Runs against: `/private/tmp/lionjudo-video-review-session/trainer-ai-session` and `http://127.0.0.1:8765/`.
- Verifies: participant persistence, assessment snapshots, JSON downloads, media playback, privacy-bound references, and responsive layout.

- [ ] **Step 1: Write the Playwright QA script**

The script must:

1. load the app and verify both videos have `readyState >= 2`, non-zero duration, and no media error;
2. save a test trainer name and `Dusan`, reload, and confirm persistence;
3. verify assessment lock becomes available only after identity save and valid assessment fields;
4. download/parse both JSON exports and assert matching `generation_id`;
5. verify every clean example is `pre_ai`, local revision 1, has no serialized AI keys, and both SHA-256 values match files inside the active generation;
6. verify every audit local revision greater than 1 is `post_ai_correction` and ineligible;
7. check desktop 1440x900 and mobile 390x844 for horizontal overflow and overlapping controls;
8. leave the active review tab open for the user.

- [ ] **Step 2: Run the new QA script and fix only observed defects via TDD**

```bash
node tests/coach_trainer_dataset_qa.cjs http://127.0.0.1:8765 /private/tmp/lionjudo-video-review-session/trainer-ai-session
```

Expected: PASS with a concise JSON summary of identities, export counts, media readiness, and viewport checks.

- [ ] **Step 3: Run the complete automated suite**

```bash
python3 -m unittest discover -s tests -v
```

Expected: all Python tests pass with no warnings or tracebacks.

- [ ] **Step 4: Commit QA coverage**

```bash
git add tests/coach_trainer_dataset_qa.cjs
git commit -m "test: cover trainer dataset workflow"
```

- [ ] **Step 5: Run fresh Terra whole-branch review**

Review the complete branch diff from the pre-feature base for spec compliance, privacy, target leakage, revision semantics, atomicity, compatibility, UI behavior, and test quality. Fix every Critical/Important finding through one reviewed fix wave, then rerun focused tests and the full suite.

- [ ] **Step 6: Push and present the final browser build**

```bash
git diff --check
git status --short --branch
git push
```

Open the cache-busted local URL, verify the trainer fields and both videos, and hand the live tab to the user.
