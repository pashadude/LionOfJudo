# Trener vs AI Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Izgraditi lokalni `Trener vs AI` pregled u kome trener prvo zaključava preciznu procenu bacanja, zatim vidi determinističku AI procenu sa dokazima i eksperimentalnim IMU pokazateljima, uz zamagljena lica i sledljive podatke za kasniji LoRA model.

**Architecture:** Kanonske video-pose serije i evaluator ostaju čiste funkcije u `pipeline/`; verzionisani duel i atomske tranzicije žive u posebnom domenskom servisu. Svi promenljivi session artefakti pripadaju nepromenljivoj generaciji, a jedan atomski zamenjen `current-generation.json` bira aktivnu generaciju; HTTP zahtev razrešava taj pokazivač tačno jednom. `coach_app/server.py` samo validira HTTP granicu i rediguje neotkrivene AI podatke. Event editor povećava verziju događaja kada promeni video granice. UI koristi postojeći lokalni HTTP server i ne dobija AI podatke pre zaključavanja. Privatnost je završni obavezni korak svakog izvoza klipa, sa `fail closed` verifikacijom.

**Tech Stack:** Python 3.13, standardni `unittest`, NumPy, OpenCV, Ultralytics YOLO11-pose, YuNet, lokalni `ThreadingHTTPServer`, vanilla HTML/CSS/JavaScript, Playwright QA skripta.

## Global Constraints

- Sav korisnički tekst mora biti na srpskoj latinici.
- Evaluator je `deterministicki-v1`; nema LLM poziva, mrežnih poziva ni tehnika koje model sam predlaže.
- Vidljivi naslov mora biti `IMU merenje (eksperimentalno)`, a podnaslov `Prototip v1. Moguća velika greška.`
- Vidljivo 3D polje mora glasiti `Biće kalibrisano u sledećoj verziji.`
- Trenerova ocena i AI ocena ostaju `null` dok nisu stvarno dobijene; nikakav kod ne upisuje podrazumevanu ocenu `3` ili `0`.
- AI procena, IMU proxy vrednosti i AI dokazi ne smeju izaći kroz trenerski API ili izveštaj pre zaključavanja pre-AI procene.
- Originalni Sony i iPhone fajlovi su nepromenljivi i ne serviraju se direktno.
- Svaki izvedeni video koji server može da prikaže mora imati `privacy_verified=true`; nema nezamagljene zamene kada privatna obrada zakaže.
- Aktivna sesija ima redom `Tai-otoshi`, `Morote-seoi-nage`, zatim povredni događaj van statistike.
- Svaki zadatak koristi RED-GREEN-REFACTOR, zatim svež Terra pregled pre commit-a. Terra ne menja isti write-set kao implementacioni agent.
- Obavezni redosled izvršenja je Task 1-5, zatim privatnost, regeneracija aktivne sesije, browser UI i tek onda završna verifikacija; browser nikada ne otvara neproveren izvedeni video.
- Worktree već sadrži nezaključane TDD izmene u `coach_app/server.py` i `tests/test_coach_server.py` za `ocena=null`; ne odbacivati ih. Task 3 ih proverava i integriše.
- Python komande u worktree-u koriste `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python` jer worktree nema sopstveni `.venv`.

## File Map

- `pipeline/video_pose_metrics.py`: kanonske COCO geometrijske serije iz stvarnih vremenskih razlika.
- `pipeline/trainer_ai_evaluator.py`: deterministički proxy pokazatelji, kvalitet, ocena, razlog i fingerprint.
- `pipeline/trainer_ai_state.py`: migracija verzionisanog event/duel stanja i selektori aktivnih zapisa.
- `pipeline/video_review_contract.py`: stroga validacija novih kolekcija i veza revizija.
- `coach_app/review_bundle.py`: nepromenljive session generacije, jedan atomski current pointer i razrešavanje snapshot-a po zahtevu.
- `coach_app/trainer_ai_service.py`: atomsko zaključavanje, otkrivanje i odgovor trenera.
- `coach_app/event_editor.py`: novi krug procene kada se granice događaja promene.
- `coach_app/server.py`: API rute, redakcija i kompatibilan draft zapis.
- `pipeline/video_review_reports.py`: redigovani i otkriveni CSV/Markdown izvoz.
- `pipeline/face_blur.py`: blur-all obrada, drugi prolaz i privatnosni izveštaj.
- `pipeline/video_review_import.py`: evaluator i privatnost tokom uvoza.
- `coach_app/static/index.html`, `app.js`, `styles.css`: trener-first tok i duel.
- `tools/video_review.py`: CLI opcije za privatni import/migraciju i server.
- `tests/test_trainer_ai_evaluator.py`: formule, pragovi i reproduktivnost.
- `tests/test_trainer_ai_state.py`: migracija, fingerprint i validacija ugovora.
- `tests/test_coach_trainer_ai.py`: atomske tranzicije, API redakcija i greške.
- `tests/test_face_blur.py`: obrada svih kandidata i `fail closed` ponašanje.
- `tests/trainer_vs_ai_qa.cjs`: stvarni tok pregledača na desktop i mobilnom viewport-u.

---

### Task 1: Kanonske 2D serije i deterministički evaluator

**Files:**
- Modify: `pipeline/video_pose_metrics.py`
- Create: `pipeline/trainer_ai_evaluator.py`
- Create: `tests/test_trainer_ai_evaluator.py`
- Modify: `tests/test_video_pose_metrics.py`

**Interfaces:**
- Consumes: COCO-17 keypoints ili kanonske `frame_metrics`, Sony granice događaja i `effective_analysis_fps`.
- Produces: `compute_pose_metrics(frames, fps, timestamps=None)`, `evaluate_event(event, frame_metrics, *, effective_analysis_fps, analysis_fingerprint) -> dict`, `compute_analysis_fingerprint(review, event, evaluator_id) -> str`, konstante `POSE_METRICS_ID` i `EVALUATOR_ID`.

- [ ] **Step 1: Dodati failing test da izvod koristi stvarni `dt`**

```python
def test_entry_speed_uses_actual_timestamp_delta(self):
    frames = [pose(0, 100, (-5, 80), (5, 80)),
              pose(10, 100, (5, 80), (15, 80))]
    metrics = compute_pose_metrics(frames, fps=6.0, timestamps=[10.0, 10.25])
    self.assertEqual(metrics[1].brzina_ulaska_norm_s, 2.0)
```

- [ ] **Step 2: Pokrenuti test i potvrditi RED**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_video_pose_metrics.PoseMetricsTests.test_entry_speed_uses_actual_timestamp_delta -v`

Expected: FAIL jer postojeći kod množi sa `fps` umesto da deli stvarnim `dt`.

- [ ] **Step 3: Implementirati kanonsku geometriju v1**

U `compute_pose_metrics` računati brzinu i rotaciju sa `times[index] - times[index - 1]`; ne računati izvod kada je `dt <= 0` ili kada postoji neinterpolirana praznina. Sačuvati postojeće COCO indekse, prag `0.30`, wrap ugla i maksimalnu interpolaciju pet uzoraka.

- [ ] **Step 4: Dodati failing evaluator test sa ručno izvedenim rezultatom**

```python
def test_available_evaluation_is_deterministic_and_cites_sony_times(self):
    event = {"event_id": "e-1", "sony_start_s": 10.0, "sony_end_s": 12.0}
    frames = available_frame_fixture()
    fingerprint = "sha256:" + "0" * 64
    result = evaluate_event(
        event,
        frames,
        effective_analysis_fps=6.0,
        analysis_fingerprint=fingerprint,
    )
    self.assertEqual(result["status"], "dostupno")
    self.assertEqual(result["evaluator_id"], "deterministicki-v1")
    self.assertIn(result["predlozena_ocena"], range(1, 6))
    self.assertGreaterEqual(len(result["dokazi"]), 2)
    self.assertTrue(all(10.0 <= d["sony_s"] <= 12.0 for d in result["dokazi"]))
    repeated = evaluate_event(
        event,
        frames,
        effective_analysis_fps=6.0,
        analysis_fingerprint=fingerprint,
    )
    self.assertEqual(result, repeated)
```

Fixture mora sadržati literalne vrednosti sa ručno izračunatim nearest-rank percentilima; helper ne sme da poziva evaluatorove privatne funkcije.

- [ ] **Step 5: Pokrenuti evaluator test i potvrditi RED**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_trainer_ai_evaluator -v`

Expected: ERROR zbog nepostojećeg `pipeline.trainer_ai_evaluator`.

- [ ] **Step 6: Implementirati evaluator kao čiste funkcije**

```python
POSE_METRICS_ID = "video-pose-metrics-v1"
EVALUATOR_ID = "deterministicki-v1"

def evaluate_event(event, frame_metrics, *, effective_analysis_fps,
                   analysis_fingerprint):
    samples = _event_samples(event, frame_metrics)
    quality = _quality(samples, event, effective_analysis_fps)
    return _build_evaluation(samples, quality, analysis_fingerprint)

def compute_analysis_fingerprint(review, event, evaluator_id=EVALUATOR_ID):
    canonical = _canonical_fingerprint_payload(review, event, evaluator_id)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
```

Implementacija mora imati fiksne cap vrednosti `4.0`, `540.0`, `24.0`, `12.0`, težine `0.20/0.25/0.20/0.15/0.20`, pragove kvaliteta iz specifikacije, nearest-rank percentil i zaokruživanje šest/tri decimale. Razlog mora navesti najmanje dva dokaza sa Sony sekundama; `niska_pouzdanost` i `nedovoljno_podataka` vraćaju `predlozena_ocena=None`.

- [ ] **Step 7: Dodati granične testove**

Pokriti tačno 12 validnih uzoraka, 11 uzoraka, coverage `0.70`, coverage `0.35`, prazninu `0.50 s`, veću prazninu, dominantnu levu/desnu rotaciju, duplikat timestamp-a, sve `null` vrednosti i JSON bez NaN/Infinity.

- [ ] **Step 8: Pokrenuti fokusirane i postojeće metričke testove**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_video_pose_metrics tests.test_trainer_ai_evaluator -v`

Expected: svi testovi OK.

- [ ] **Step 9: Terra pregled i commit**

Terra proverava formule prema specifikaciji, ručno preračunava najmanje jedan fixture i potvrđuje da razlog ne tvrdi fizičku silu/snagu. Posle ispravki:

```bash
git add pipeline/video_pose_metrics.py pipeline/trainer_ai_evaluator.py tests/test_video_pose_metrics.py tests/test_trainer_ai_evaluator.py
git commit -m "feat: add deterministic trainer ai evaluator"
```

---

### Task 2: Verzije događaja, fingerprint i migracija ugovora

**Files:**
- Create: `pipeline/trainer_ai_state.py`
- Modify: `pipeline/video_review_contract.py`
- Modify: `pipeline/video_review_migration.py`
- Create: `tests/test_trainer_ai_state.py`
- Modify: `tests/test_video_review_migration.py`

**Interfaces:**
- Consumes: review v2 payload i evaluator iz Task 1.
- Produces: `migrate_trainer_ai_payload(payload) -> dict`, `active_ai_evaluation(event)`, `active_trainer_assessment(event)`, `validate_trainer_ai_event(event)`; review verzija raste na `3`.

- [ ] **Step 1: Napisati failing test idempotentne migracije**

```python
def test_migration_adds_versioned_state_without_inventing_scores(self):
    migrated = migrate_trainer_ai_payload(legacy_review_fixture())
    normal = migrated["events"][0]
    self.assertEqual(migrated["version"], 3)
    self.assertEqual(normal["event_revision"], 1)
    self.assertTrue(normal["analysis_fingerprint"].startswith("sha256:"))
    self.assertEqual(normal["trener_procene"], [])
    self.assertIsNone(normal["ocena"])
    self.assertEqual(migrated, migrate_trainer_ai_payload(migrated))
```

- [ ] **Step 2: Pokrenuti test i potvrditi RED**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_trainer_ai_state -v`

Expected: ERROR zbog nepostojećeg modula.

- [ ] **Step 3: Implementirati migraciju bez gubitka legacy podataka**

Legacy `potvrdena_tehnika`, `ocena` i `napomena` kopirati u `legacy_annotations` sa oznakom `nije_pre_ai`, ali ne praviti `trener_procene`. Top-level `potvrdena_tehnika` ostaje draft za prikaz; top-level `ocena` se postavlja na `None` ako nema dokazano zaključane nove revizije. Kreirati `ai_procene`, `trener_procene`, `procene_ai_predloga`, `event_revision`, `analysis_fingerprint`, `aktivna_trener_revizija` i `aktivni_duel`.

- [ ] **Step 4: Dodati stroge validator testove**

Testirati odbijanje: duplikata revizija, pogrešnog SHA-256 oblika, `pre_ai` revizije sa pogrešnim fingerprintom, ocene van `1..5`, razloga bez citirane Sony sekunde kada je vidljivost dovoljna, post-AI odgovora bez timestamp-a i veze na nepostojeći evaluator/trener revision.

- [ ] **Step 5: Implementirati validaciju u postojećem ugovoru**

`validate_review_payload` poziva `validate_trainer_ai_event` za svaki normalni događaj. Povredni događaj mora odbiti AI/trener kolekcije. ISO vremena validirati sa `datetime.fromisoformat`, a `null` dozvoliti samo u stanjima definisanim specifikacijom.

- [ ] **Step 6: Integrisati migraciju sesije i izvedene JSON fajlove**

`migrate_review_payload` prvo kanonizuje frame metrics, zatim računa sve event sažetke/evaluacije, pa poziva `migrate_trainer_ai_payload`. `analysis/event_metrics.json` mora dobiti istu verzionisanu listu kao `review.json`.

- [ ] **Step 7: Pokrenuti migracione i ugovorne testove**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_trainer_ai_state tests.test_video_review_migration tests.test_video_sync -v`

Expected: svi testovi OK i migracija dvaput daje isti JSON.

- [ ] **Step 8: Terra pregled i commit**

Terra proverava da migracija ne pretvara staru ocenu u nezavisnu pre-AI etiketu i da aktivna dva bacanja počinju sa `ocena=null`.

```bash
git add pipeline/trainer_ai_state.py pipeline/video_review_contract.py pipeline/video_review_migration.py tests/test_trainer_ai_state.py tests/test_video_review_migration.py
git commit -m "feat: version trainer ai review state"
```

---

### Task 3: Atomsko zaključavanje, otkrivanje i odgovor trenera

**Files:**
- Create: `coach_app/review_bundle.py`
- Create: `coach_app/trainer_ai_service.py`
- Modify: `coach_app/server.py`
- Create: `tests/test_coach_trainer_ai.py`
- Modify: `tests/test_coach_server.py`

**Interfaces:**
- Consumes: versioned state iz Task 2 i atomski storage.
- Produces: `GenerationStore.resolve_current() -> GenerationSnapshot`, `GenerationStore.stage_and_activate(review, event_metrics, csv_text, markdown_text, staged_media=None) -> GenerationSnapshot`, `TrainerAiService.lock_assessment`, `.reveal_ai`, `.save_ai_feedback`, `.public_review`; svaka mutation metoda vraća mapu sa ključevima `event` (ažurirani event dict) i `assessment` (novi ili aktivni assessment dict); HTTP rute `POST /api/events/{id}/trainer-assessments`, `POST /api/events/{id}/ai-reveal`, `PUT /api/events/{id}/ai-feedback`.

- [ ] **Step 1: Sačuvati i proveriti postojeći nullable-score TDD diff**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_coach_server.CoachServerTests.test_annotation_allows_unscored_trainer_confirmation -v`

Expected: OK. Pregledati `git diff -- coach_app/server.py tests/test_coach_server.py`; zadržati podršku za `ocena=None` i status `trener`, ali uklopiti je u novi servis.

- [ ] **Step 2: Napisati failing test za zaključavanje pre AI**

```python
def test_lock_creates_immutable_pre_ai_revision(self):
    result = service.lock_assessment("e-1", {
        "status_vidljivosti": "dovoljno_vidljivo",
        "potvrdena_tehnika": "Tai-otoshi",
        "ocena": 4,
        "razlog": "Na 130.420 s kukovi kasne za rotacijom.",
        "citirani_sony_trenuci_s": [130.420],
    })
    self.assertEqual(result["assessment"]["faza"], "pre_ai")
    self.assertIsNone(active_ai_evaluation(result["event"])["ai_otkriven_u"])
```

Test koristi ubrizgan `clock=lambda: fixed_datetime` da timestamp bude determinističan.

- [ ] **Step 3: Pokrenuti service test i potvrditi RED**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_coach_trainer_ai -v`

Expected: ERROR zbog nepostojećeg servisa.

- [ ] **Step 4: Implementirati atomske state transitions i generacijski store**

`lock_assessment` validira event, vidljivost, naziv, ocenu, razlog i citirane sekunde unutar granica. Prvi zapis za `event_revision` je `pre_ai`; zapis posle otkrivanja je `post_ai_korekcija` sa sledećim brojem i ne menja prvu reviziju. `reveal_ai` odbija događaj bez matching pre-AI revizije i atomski upisuje `ai_otkriven_u`. `save_ai_feedback` zahteva odnos iz tri dozvoljene vrednosti i vezuje odgovor za aktivni duel.

`GenerationStore` čuva svaku verziju pod `.review-generations/{uuid}/` sa `review.json`, `analysis/event_metrics.json`, `izvestaj.csv`, `izvestaj.md` i kompletnim izvedenim medijima/manifesta te generacije. Nepromenjeni veliki mediji prenose se hard-linkom unutar istog filesystem-a, a novi staged mediji se kopiraju; nijedan fajl aktivne generacije se ne menja. Posle validacije i `fsync`-a nove generacije store jednim `os.replace` pozivom menja `current-generation.json` sa relativnim ID-em generacije. Svaki HTTP/report čitalac poziva `resolve_current()` jednom na početku zahteva i sve dalje čita iz tog snapshot direktorijuma. Legacy session bez pointera je read-only generation zero; prva mutation ga bootstrappuje u generaciju bez menjanja legacy fajlova. Neuspela izgradnja briše samo neaktivnu novu generaciju i ne menja current pointer.

- [ ] **Step 5: Dodati failing test svih javnih puteva i API redakcije**

Pre zaključavanja `GET /api/session`, `GET /api/events/{id}`, svi uspešni odgovori mutation ruta i `/izvestaj.csv`/`izvestaj.md` ne smeju sadržati `predlozena_ocena`, AI razlog, dokaze ni IMU proxy vrednosti. Route matrica obuhvata postojeće `PUT annotation`, `PUT bounds`, `POST session/sync`, `POST create`, `POST merge`, `POST split`, `DELETE event` i tri nove trener/AI rute. Direktni zahtevi za `/review.json`, `/analysis/event_metrics.json`, `/analysis/` i njihove percent-encoded varijante moraju vratiti `404`. Posle reveal rute javni API/izveštaji vraćaju otkrivene vrednosti samo za taj događaj; interni JSON i dalje nije direktno serviran.

- [ ] **Step 6: Implementirati rute i javni projection**

`ReviewServer` dobija jedan `TrainerAiService` sa istim `mutation_lock`. Nijedan handler ne serijalizuje interni review ili rezultat editora/servisa direktno. `service.public_review()` je jedini javni model: `/api/session` vraća tu projekciju, a svaki GET ili mutation event odgovor ponovo bira javnu projekciju odgovarajućeg eventa nakon uspešnog commit-a. Eksplicitna deny-lista blokira `review.json`, `analysis/*.json` i directory fallback pre generičkog static/media rukovanja. Stari annotation endpoint ostaje kompatibilan za draft i dozvoljava `ocena=null`; ne kreira zaključanu procenu.

- [ ] **Step 7: Dodati negativne, istorijske i rollback testove**

Pokriti: razlog bez citirane sekunde, sekundu van event granica, reveal pre lock-a, dvostruki reveal, feedback pre reveal-a, boolean ocenu, povredni event i `nedovoljno_vidljivo` sa svim poljima `null`. Zaključati post-AI korekciju i dokazati da prva `pre_ai` revizija ostaje byte-for-byte ista. Ubrizgati grešku renderovanja, kopiranja/hard-linkovanja, validacije, `fsync`-a i zamene pointera; current snapshot mora ostati prethodna generacija. Kontrolisano pauzirati staging pre pointer switch-a i istovremeno čitati sve javne resurse: svaki zahtev mora videti kompletno staru generaciju; posle jednog switch-a kompletno novu, nikada mešavinu.

- [ ] **Step 8: Pokrenuti serverske testove**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_coach_server tests.test_coach_trainer_ai -v`

Expected: svi testovi OK; CSV rezultat za praznu ocenu je prazan string.

- [ ] **Step 9: Terra pregled i commit**

Terra proverava da nijedan API put ne otkriva AI pre lock-a i da pre-AI revizija ne može biti prepisana.

```bash
git add coach_app/review_bundle.py coach_app/trainer_ai_service.py coach_app/server.py tests/test_coach_trainer_ai.py tests/test_coach_server.py
git commit -m "feat: add trainer first ai reveal workflow"
```

---

### Task 4: Event editor i novi krug procene posle promene granica

**Files:**
- Modify: `coach_app/event_editor.py`
- Modify: `pipeline/trainer_ai_state.py`
- Modify: `tests/test_coach_event_editor.py`

**Interfaces:**
- Consumes: `event_revision`, fingerprint, evaluator i `GenerationStore` iz Taska 3.
- Produces: `start_new_event_revision(review, event) -> dict`; sve event-editor operacije vraćaju review sa usklađenim trenutnim duelom.

- [ ] **Step 1: Napisati failing revision/fingerprint testove**

Zaključati `e-1`, otkriti AI, zatim promeniti granice. Assert: `event_revision` prelazi `1 -> 2`, fingerprint se menja, istorijski nizovi ostaju, novi AI zapis ima `ai_otkriven_u=None`, `aktivni_duel=None`, a javni projection ponovo skriva AI. Parametrizovano ponoviti isto za promenu izabranog track ID-a, potpisa izvora, `POSE_METRICS_ID` i `EVALUATOR_ID`; nepromenjen input ne povećava revision.

- [ ] **Step 2: Pokrenuti test i potvrditi RED**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_coach_event_editor.CoachEventEditorTests.test_bounds_change_starts_new_assessment_round -v`

Expected: FAIL jer editor sada samo menja granice i sažetak.

- [ ] **Step 3: Implementirati `start_new_event_revision`**

Posle uspešnog generisanja oba klipa i novih event metrika povećati revision, izračunati novi fingerprint i AI procenu. Isti helper pozivati i kada se promeni track, izvorni potpis, pose-metrics verzija ili evaluator. Ne menjati postojeće `trener_procene`, `ai_procene` ni `procene_ai_predloga`; dodati novi AI zapis za revision 2 i očistiti samo aktivne pointere. Editor predaje novi review, `analysis/event_metrics.json`, oba izveštaja i staged medije istom `GenerationStore.stage_and_activate` pozivu iz Taska 3; nijedan editor put ne radi sopstveni parcijalni upis.

- [ ] **Step 4: Definisati split/merge/create pravila**

Novi event počinje na revision 1 sa praznim trener kolekcijama. Split čuva istoriju samo na levoj strani; desna strana dobija novu nezavisnu istoriju. Merge je dozvoljen samo pre bilo kog zaključanog duela ili zahteva eksplicitno odbacivanje aktivnih procena; za v1 implementirati bezbedniju zabranu merge-a kada bilo koja strana ima `trener_procene`.

- [ ] **Step 5: Proširiti transakcione rollback testove za ceo bundle**

Neuspelo računanje fingerprinta, evaluatora ili medija mora ostaviti current pointer na prethodnoj generaciji. Zasebno ubrizgati kvar pri pisanju svakog od četiri artefakta (`review.json`, `analysis/event_metrics.json`, CSV, Markdown), pri kopiranju staged medija i pri pointer switch-u; posle svakog kvara svi čitaoci i stari klipovi moraju pokazivati byte-for-byte prethodnu generaciju i ne sme ostati neaktivna nepotpuna generacija. Concurrency test pauzira editor neposredno pre pointer switch-a i dokazuje da paralelni GET zahtevi vide celu staru, a posle switch-a celu novu generaciju.

- [ ] **Step 6: Pokrenuti editor suite**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_coach_event_editor -v`

Expected: svi testovi OK.

- [ ] **Step 7: Terra pregled i commit**

Terra poredi tri istorijska slučaja: bounds change, split i merge zabranu.

```bash
git add coach_app/event_editor.py pipeline/trainer_ai_state.py tests/test_coach_event_editor.py
git commit -m "feat: version assessments across event edits"
```

---

### Task 5: Sledljivi i redigovani CSV/Markdown izveštaji

**Files:**
- Modify: `pipeline/video_review_reports.py`
- Modify: `tests/test_video_review_reports.py`
- Modify: `tests/test_coach_trainer_ai.py`

**Interfaces:**
- Consumes: interni review i policy `include_unrevealed=False`.
- Produces: `report_rows(review, *, include_unrevealed=False)`, CSV i Markdown sa AI/trener/IMU kolonama.

- [ ] **Step 1: Napisati failing report test pre i posle reveal-a**

Pre reveal-a red ne sadrži AI score/reason/dokaze/IMU. Posle reveal-a sadrži evaluator ID, AI ocenu/status/pouzdanost/razlog, JSON-kompaktne dokaze, pre-AI reviziju, citirane sekunde, odnos prema AI i timestamp odgovora.

- [ ] **Step 2: Pokrenuti test i potvrditi RED**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_video_review_reports -v`

Expected: FAIL jer postojeći `REPORT_FIELDS` nema nove kolone.

- [ ] **Step 3: Implementirati stabilan flat export**

Dodati kolone sa srpskim nazivima i deterministički JSON za kolekcije (`ensure_ascii=False`, `sort_keys=True`, kompaktni separators). Povredni događaj ostaje read-only red bez AI/trener vrednosti. Sve `None` vrednosti izlaze kao prazna polja, ne `None` tekst.

- [ ] **Step 4: Testirati escaping i revizije**

Razlog sa `|`, novim redom, navodnicima i srpskim slovima mora ostati jedan CSV/Markdown red. Dve trener revizije moraju biti izvezene bez gubitka prve pre-AI procene.

- [ ] **Step 5: Pokrenuti report i API testove**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_video_review_reports tests.test_coach_trainer_ai -v`

Expected: svi testovi OK.

- [ ] **Step 6: Terra pregled i commit**

Terra proverava jedan CSV i Markdown primer i potvrđuje da pre-reveal izvoz ne curi.

```bash
git add pipeline/video_review_reports.py tests/test_video_review_reports.py tests/test_coach_trainer_ai.py
git commit -m "feat: export trainer ai evidence and revisions"
```

---

### Task 6: Blur-all privatnost i `fail closed` izvedeni mediji

**Files:**
- Modify: `pipeline/face_blur.py`
- Modify: `pipeline/video_review_import.py`
- Modify: `coach_app/event_editor.py`
- Modify: `coach_app/server.py`
- Create: `tests/test_face_blur.py`
- Modify: `tests/test_video_review_import.py`
- Modify: `tests/test_coach_event_editor.py`

**Interfaces:**
- Consumes: sirov izvedeni klip i YOLO/YuNet detektore.
- Produces: `blur_all_faces(model, input_path, output_path, device) -> BlurReport`, `verify_blurred_clip(model, input_path, device, score_threshold=0.30) -> BlurReport`; jedinstveni `derived_media_manifest` sa relativnom putanjom, vrstom medija, brojem kadrova, detektorskim prolazima i `privacy_verified` za event klipove, anchor previews i side-by-side video.

- [ ] **Step 1: Napisati failing unit test da se nijedan track ne izuzima**

Koristiti male sintetičke frame matrice i unapred određene head regione; assertovati da su svi regioni promenjeni blur funkcijom, uključujući designated athlete region. Test ne proverava mock pozive, već stvarne izlazne piksele i smanjenje lokalne varijanse.

- [ ] **Step 2: Pokrenuti test i potvrditi RED**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_face_blur -v`

Expected: FAIL jer postojeći modul može da izuzme athlete track i nema `privacy_verified`.

- [ ] **Step 3: Implementirati blur-all i drugi prolaz**

Ukloniti athlete exemption iz produkcionog review puta. Prvi prolaz koristi YOLO head keypoints i YuNet `score_threshold=0.60`; verifikacioni prolaz koristi oba izvora sa pragom `0.30`, dodatno bluruje reziduale i ponavlja proveru. `BlurReport` sadrži `total_frames`, `first_pass_candidates`, `second_pass_candidates`, `privacy_verified` i failure reason.

- [ ] **Step 4: Uvesti atomski privatni exporter**

Cut ide u privremeni fajl, blur u drugi privremeni fajl, verifikacija, pa tek onda `os.replace` na konačni `events/{id}/{camera}.mp4`. Neuspeh briše privremene fajlove i ne dira prethodni verified klip.

- [ ] **Step 5: Integrisati import, event editor i allow-list server**

Import i sve create/bounds/split/merge regeneracije moraju koristiti isti privacy exporter. Isto važi za svaki anchor preview i `media/session_side_by_side.mp4`. Pre generičkog fallback-a router eksplicitno odbija raw `/events/`, `/previews/`, `/analysis/` i interne `/media/` filesystem putanje. Jedini javni put za izvedeni medij je manifest-checked `/media/{manifest_relative_path}`: server normalizuje relativnu putanju, nalazi je u `derived_media_manifest` i servira samo kada matching zapis ima `privacy_verified=true`; nepoznata, neoznačena, traversal ili neuspešno verifikovana putanja vraća `404`, nikada nezamagljen klip.

- [ ] **Step 6: Dodati realni model smoke test na kratkom klipu**

Iz aktivnog Sony i iPhone izvora iseći po 1 sekundu u `/private/tmp`, pokrenuti YOLO + postojeći `models/face_detection_yunet_2023mar.onnx`, potvrditi dekodiranje svih frame-ova i `privacy_verified=true`. Ne commitovati testne medije.

- [ ] **Step 7: Dodati failure i manifest testove, pa pokrenuti suite**

Ubrizgati: nedostupan detector/model, grešku dekodiranja u sredini klipa, kandidata posle prvog prolaza, kandidata posle drugog prolaza, neoznačen event klip, anchor preview i side-by-side putanju. Assertovati da samo konačni prolaz bez kandidata postavlja `privacy_verified=true`, da manifest obuhvata sve tri vrste medija i da server sve ostalo vraća `404`. HTTP testovi moraju probati raw `/events/{id}/sony.mp4`, raw `/previews/anchor_01_sony.mp4`, manifest rutu `/media/session_side_by_side.mp4`, istu rutu posle postavljanja `privacy_verified=false`, percent-encoded traversal i nepostojeći manifest ključ; samo verified manifest ruta sme vratiti `200/206`.

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_face_blur tests.test_video_review_import tests.test_coach_event_editor -v`

Expected: svi testovi OK.

- [ ] **Step 8: Terra vizuelni pregled i commit**

Terra pregleda kontaktne listove početak/sredina/kraj obe kamere i potvrđuje da nema vidljivih lica u serviranim klipovima.

```bash
git add pipeline/face_blur.py pipeline/video_review_import.py coach_app/event_editor.py coach_app/server.py tests/test_face_blur.py tests/test_video_review_import.py tests/test_coach_event_editor.py
git commit -m "feat: require verified face blur for review media"
```

---

### Task 7: Migracija i regeneracija aktivne sesije

**Files:**
- Modify: `tools/video_review.py`
- Modify: `tests/test_video_review_cli.py`
- Runtime output only: `/private/tmp/lionjudo-video-review-session/trainer-ai-session/`

**Interfaces:**
- Consumes: corrected session, original source hashes, model path i device.
- Produces: novu finalnu session kopiju sa version 3 review-om, tačna tri eventa i verified medijima.

- [ ] **Step 1: Dodati failing CLI test za bezbednu output kopiju**

`migrate-ai --session-dir source --output-dir target --model-path path --device cpu` mora odbiti isti source/target, postojeći neprazan target bez `--replace-derived` i izvor čiji hash ne odgovara review-u.

- [ ] **Step 2: Pokrenuti CLI test i potvrditi RED**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_video_review_cli -v`

Expected: FAIL jer podkomanda ne postoji.

- [ ] **Step 3: Implementirati bezbednu migraciju**

Kopirati samo session JSON/analysis strukturu u novi target; ne kopirati originalne source videe. Normalizovati active events na:

```text
Tai-otoshi         Sony 128.5-132.0 / iPhone 131.5-135.0
Morote-seoi-nage   Sony 132.8-135.0 / iPhone 135.8-138.0
povreda            Sony 135.0-136.0 / iPhone 138.0-139.0
```

Oba normalna eventa imaju `status=trener`, top-level `ocena=null`, prazne `trener_procene`, AI procenu iz sveže frame serije i neotkriven AI timestamp.

- [ ] **Step 4: Regenerisati sve izvedene medije sa blur-all**

Regenerisati event klipove, anchor previews i side-by-side video. Svaki zapis mora imati `privacy_verified=true`; ako ijedan ne prođe, komanda izlazi nonzero i target se ne označava spremnim.

Run:

```bash
/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python tools/video_review.py migrate-ai --session-dir /private/tmp/lionjudo-video-review-session/corrected-session --output-dir /private/tmp/lionjudo-video-review-session/trainer-ai-session --model-path /Users/pauldudko/VSProjects/LionOfJudo/yolo11x-pose.pt --device cpu
```

Expected: exit 0 i target review ima `session_ready=true` tek posle verifikacije celog manifesta.

- [ ] **Step 5: Dokazati nepromenljivost izvora**

Pre i posle migracije izračunati SHA-256 i veličine:

```bash
shasum -a 256 /Volumes/Untitled/PRIVATE/M4ROOT/CLIP/C0007.MP4 /Users/pauldudko/Downloads/IMG_3852.mov
```

Expected: potpisi identični onima u review `sources` i vrednostima pre migracije.

- [ ] **Step 6: Validirati session JSON i medije i dokazati idempotence**

Pokrenuti drugi `migrate-ai` iz istog izvora ka posebnom praznom targetu, zatim ponoviti ka tom targetu sa `--replace-derived`; kanonski review/metrics/report izlazi moraju biti identični osim eksplicitno dokumentovanih runtime timestamp-a, dok source ostaje netaknut. Pokrenuti `ffprobe` za sve MP4 fajlove, proveru duration-a prema event granicama i JSON assert da aktivni izveštaj nema `O-soto-gari`, fake score 3 ili AI leak.

```bash
/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python tools/video_review.py migrate-ai --session-dir /private/tmp/lionjudo-video-review-session/corrected-session --output-dir /private/tmp/lionjudo-video-review-session/idempotence-check --model-path /Users/pauldudko/VSProjects/LionOfJudo/yolo11x-pose.pt --device cpu
/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python tools/video_review.py migrate-ai --session-dir /private/tmp/lionjudo-video-review-session/corrected-session --output-dir /private/tmp/lionjudo-video-review-session/idempotence-check --model-path /Users/pauldudko/VSProjects/LionOfJudo/yolo11x-pose.pt --device cpu --replace-derived
```

- [ ] **Step 7: Terra pregled i commit CLI koda**

Terra proverava session summary, sinhronizovane frame parove i privatne kontaktne listove.

```bash
git add tools/video_review.py tests/test_video_review_cli.py
git commit -m "feat: prepare versioned trainer ai sessions"
```

Runtime target ostaje lokalni artifact i ne ulazi u git.

---

### Task 8: Trener-first duel interfejs

**Files:**
- Modify: `coach_app/static/index.html`
- Modify: `coach_app/static/app.js`
- Modify: `coach_app/static/styles.css`
- Create: `tests/trainer_vs_ai_qa.cjs`
- Modify: `tests/test_coach_server.py`

**Interfaces:**
- Consumes: javni API iz Task 3 i isključivo blur-verifikovanu v3 sesiju iz Taska 7.
- Produces: zaključavanje pre-AI procene, reveal, duel, IMU pločice, činjenice sistema i feedback kontrole.

- [ ] **Step 1: Napisati samostalnu browser RED skriptu za skriven AI**

`tests/trainer_vs_ai_qa.cjs` podržava tačno dva režima: `--session-dir PATH` pravi privremenu kopiju sa `fs.mkdtempSync`/`fs.cpSync`, pokreće `tools/video_review.py serve` kao child process na portu `8767`, čeka uspešan `GET /api/session`, a u `finally` gasi server i briše samo svoju privremenu kopiju; `--base-url URL` koristi već pokrenut finalni server i ne menja lifecycle. U oba režima, pre `page.goto` i bilo kog video zahteva, skripta preko `/api/session` zahteva `version == 3`, `session_ready == true`, neprazan `derived_media_manifest` i `privacy_verified == true` za svaki zapis; bilo koji neuspeh prekida QA pre otvaranja browsera. Skripta ne poziva ni `migrate` ni `migrate-ai`. Zatim bira normalni event i assertuje da su `#ai-duel`, `#imu-panel` i `#system-facts` skriveni. Popunjava tehniku, score 4, razlog i trenutnu Sony sekundu, pa klikće `Zaključaj procenu` i `Otkrij AI izazov`.

- [ ] **Step 2: Pokrenuti skriptu i potvrditi RED**

Run: `node tests/trainer_vs_ai_qa.cjs --session-dir /private/tmp/lionjudo-video-review-session/trainer-ai-session`

Expected: FAIL zbog nepostojećih UI kontrola; child server i privremena sesija se ipak uredno gase/uklanjaju.

- [ ] **Step 3: Izgraditi trener-first formu**

Zameniti postojeći `Sačuvaj` tok sa:

- poljem potvrđene tehnike;
- pet stabilnih numeric score izbora;
- `Dovoljno vidljivo / Nedovoljno vidljivo` kontrolom;
- textarea `Razlog trenera`;
- dugmetom `Dodaj trenutnu sekundu` koje upisuje format `[130.420 s]`;
- dugmetom `Zaključaj procenu`.

Povredni event ostaje samo za čitanje. Post-AI korekcija mora jasno prikazati da pravi novu reviziju.

- [ ] **Step 4: Izgraditi psihološki duel posle reveal-a**

Prikazati `TRENER` i `AI` u istoj ravni, razliku score-a i copy `AI odstupa za X poena. Odbrani procenu.` bez lažnog pobednika. `ČINJENICE SISTEMA` prikazuju metric/value/unit/Sony second. `IMU merenje (eksperimentalno)` prikazuje ugaonu brzinu, proxy ubrzanja, proxy impulsa, intenzitet, smer, vršnu sekundu, pouzdanost i 3D status.

- [ ] **Step 5: Dodati feedback kontrole**

Segmentirana kontrola ima `Slažem se`, `Delimično`, `Ne slažem se`; razlog je opcioni. Svaki dokaz dobija meni `Prihvatam / Nepotpun / Osporavam`. Čuvanje zadržava selekciju posle reload-a. Kada evaluator vrati `niska_pouzdanost` ili `nedovoljno_podataka`, nema AI score poređenja, ali ista kontrola ostaje dostupna uz sistemski status `AI nema dovoljno podataka`.

- [ ] **Step 6: Završiti responsive CSS bez nested cards**

Zadržati kartice samo za pojedinačne funkcionalne panele. Stabilizovati video `aspect-ratio`, IMU grid i score kontrole. Na `390x844` sve prelazi u jednu kolonu, bez horizontalnog overflow-a; na `1440x1000` duel ostaje skenabilan i video je u prvom viewport-u.

- [ ] **Step 7: Pokrenuti browser QA**

Skripta proverava: AI skriven pre lock-a, obavezan timestamp, reveal, tačan copy, sve IMU pločice, tri relation izbora, reload persistence, povredni read-only state, `document.documentElement.scrollWidth === innerWidth`, video `readyState >= 2` i najmanje jedan canvas sa različitim pikselima.

Run: `node tests/trainer_vs_ai_qa.cjs --session-dir /private/tmp/lionjudo-video-review-session/trainer-ai-session`

Expected: `trainer-vs-ai QA OK`.

- [ ] **Step 8: Pokrenuti statičke/server testove**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest tests.test_coach_server -v`

Expected: svi testovi OK.

- [ ] **Step 9: Terra vizuelni pregled i commit**

Terra pregleda desktop/mobile screenshots, tekstualnu hijerarhiju i psihološki tok; posebno proverava da AI nije slučajno vidljiv pre lock-a.

```bash
git add coach_app/static/index.html coach_app/static/app.js coach_app/static/styles.css tests/trainer_vs_ai_qa.cjs tests/test_coach_server.py
git commit -m "feat: add trainer versus ai review interface"
```

---

### Task 9: Potpuna verifikacija, lokalni server i handoff

**Files:**
- Modify only if verification finds a defect: files owned by the failing task
- Runtime screenshots: `/private/tmp/lionjudo-trainer-ai-qa/`

**Interfaces:**
- Consumes: kompletan branch i finalnu aktivnu sesiju.
- Produces: prolaznu test matricu, desktop/mobile QA dokaze i server na `127.0.0.1:8765`.

- [ ] **Step 1: Pokrenuti kompletan Python test suite**

Run: `/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m unittest discover -s tests -v`

Expected: svi testovi OK; nema warning-a o NaN, nezatvorenim resursima ili privremenim fajlovima.

- [ ] **Step 2: Pokrenuti statičke provere**

```bash
git diff --check
/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python -m compileall coach_app pipeline tools tests
```

Expected: exit 0.

- [ ] **Step 3: Pokrenuti finalni server na rezervnom portu**

```bash
/Users/pauldudko/VSProjects/LionOfJudo/.venv/bin/python tools/video_review.py serve --session-dir /private/tmp/lionjudo-video-review-session/trainer-ai-session --port 8767
```

Ne završavati zadatak dok ova potrebna session komanda radi bez greške.

- [ ] **Step 4: Pokrenuti kompletan browser QA**

Run: `node tests/trainer_vs_ai_qa.cjs --base-url http://127.0.0.1:8767`

Snimiti desktop `1440x1000` i mobile `390x844`; proveriti nonblank video, canvas pixels, sinhronizovano seekovanje, AI redakciju pre lock-a, duel posle reveal-a, persistenciju i read-only povredu.

- [ ] **Step 5: Proveriti API i izveštaje ručno kontrolisanim pozivima**

Sa novom kopijom session-a: GET pre lock-a, lock event, reveal, feedback, reload, CSV i Markdown. Potvrditi da drugi event ostaje neotkriven dok se zasebno ne zaključa.

- [ ] **Step 6: Proveriti privatnost i originalne izvore**

Otvoriti kontaktne listove svih šest event klipova i izvedenih preview/side-by-side medija. Ponovo proveriti source SHA-256 i da server odbija direktnu source putanju i svaki medij bez `privacy_verified=true`.

- [ ] **Step 7: Finalni Terra code/data/UI pregled**

Terra dobija diff, test output, review summary i screenshots. Svaki nalaz se vraća vlasniku odgovarajućeg taska; posle ispravke ponoviti puni relevantni test i browser QA.

- [ ] **Step 8: Prebaciti proverenu sesiju na port 8765**

Zaustaviti samo stari LionOfJudo proces na `8765`, zatim pokrenuti isti provereni command sa `--port 8765`. Potvrditi `GET /api/session`, HTML/CSS/JS, oba range video zahteva i otvoriti cache-buster URL:

`http://127.0.0.1:8765/?v=trainer-ai-v1`

- [ ] **Step 9: Završni commit samo ako je verifikacija zahtevala popravke**

Za svaku popravku ponoviti `git add` i commit komandu vlasničkog taska, sa
porukom `fix: complete trainer ai end to end verification`. Staging sme da
sadrži samo fajlove navedene u `Files` odeljku tog taska. Ako nema dodatnih
izmena, ne praviti prazan commit.
