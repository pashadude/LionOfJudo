# LionOfJudo
## Professional sports science for judo clubs that could never afford it

*Sponsor & investor brief — Judo Club Niš pilot, July 2026*

---

## The story

A year ago my son walked into Judo Club Niš. Twelve months later he took 4th
at an international tournament, gold at a local one, and was moved to the
senior group at age 8. His coaches say the word "Olympic" out loud.

Elite federations answer that kind of talent with sports-science labs:
multi-camera motion capture, force measurement, technique analytics —
$10,000–50,000 per installation, plus $500+/month to run. A club in Niš
will never have that. So I built it in my garage.

**LionOfJudo turns two ordinary cameras and $40 of sensors into a
professional analysis system** — technique recognition, biomechanical
skeleton tracking, and (sensor units arriving now) direct measurement of
throw power in g-force and rotation speed. It runs offline on a laptop.
It costs nothing per month. And every child's face except the analyzed
athlete's is blurred automatically, by design.

---

## What already works — today, not on slides

| Capability | Status |
|---|---|
| Automatic throw detection from wearable sensor g-spikes | ✅ software verified end-to-end |
| Two-camera sync with no cables (audio cross-correlation, <33 ms) | ✅ verified |
| 17-point skeleton tracking + biomechanics on every clip | ✅ working |
| Technique recognition: 26-waza trained classifier + 134-waza reference catalog, cross-checking each other | ✅ working |
| Automatic face blurring of every child except the analyzed athlete | ✅ working, fail-safe design |
| One-command overnight processing; morning report with per-throw clips, technique, metrics | ✅ working |
| Showcase video output (recognition panel with probabilities) | ✅ **demoed on real competition footage — 2 of 2 throws recognized correctly** (o-soto-gari, uchi-mata) |
| Wearable sensor units (ESP32 + IMU, in-gi, child-safe mounting) | 🔶 firmware compiled & tested in software; hardware parts arrive ~3 weeks |

**Validation:** the demo was reviewed by five Judo Club Niš
trainers, judoka and judges — Marko Radulović, Stevan Vukadinović,
Vladimir Spasić, Uroš Kostadinović, and **Dragan Spasić, former national
trainer of the Serbian team**. Outcome: the club committed **two young
judokas — one of them Serbia's #1 in U14 — as pilot athletes** to build
the training dataset.

---

## Why this is hard to copy

1. **The dataset nobody else has.** Generic AI knows what judo looks like
   on YouTube. Ours is being trained on Serbian competition youth —
   kid-sized bodies, real resistance, real referees — labeled by the
   trainers who coach them. Every training session automatically produces
   new labeled data; the system gets smarter every week the club uses it.
2. **Measured power, not estimated.** In-gi accelerometers give actual
   impact g-force and hip rotation speed per throw. No video-only system
   can produce "6.2 g impact, 420°/s rotation — 15% more than last month."
3. **Privacy as architecture, not policy.** Blur-by-default with a single
   confirmed athlete exempted; the failure mode over-blurs, never exposes.
   This is what lets a club legally and ethically film children's training.
4. **Cost structure that scales to poor clubs.** ~$40 of sensors per
   athlete kit, cameras the club already owns, zero cloud costs. A
   national federation could equip fifty clubs for the price of one
   commercial installation.

---

## The plan

**Phase 1 — Instrumented pilot (weeks 1–6, starts when parts arrive ~3 weeks)**
- Assemble & bench-test 2 wearable sensor units; dress rehearsal at home
  with my son (the garage lab continues)
- First instrumented sessions at Judo Club Niš with the 2 pilot athletes
- First reports with measured throw power delivered to the trainers

**Phase 2 — Dataset & accuracy sprint (weeks 4–12, overlaps)**
- Trainers label competition/training footage with the one-keypress tool
  (no technical skill needed)
- Target: 50+ repetitions per technique for the 10 techniques the pilot
  athletes actually compete with → recognition accuracy on *our* athletes
  goes from baseline to production-grade
- Held-out evaluation on real competition footage after every retrain

**Phase 3 — Trainer product (months 3–5)**
- Per-athlete progress dashboard: technique quality trend, power trend,
  session-to-session comparison
- Serbian-language reports; PDF export for parents
- Second sensor kit set; 5+ athletes instrumented

**Phase 4 — Beyond one club (months 6+)**
- Packaged kit (sensors, mounting, software image) reproducible by any club
- Pilot #2 at a second Serbian club; federation conversation with results
  in hand

---

## What sponsorship buys

| Tier | Amount | What it funds | What the sponsor gets |
|---|---|---|---|
| **Kit sponsor** | €500 | Sensor kits for 5 athletes, spare cameras/tripods, mounting materials | Logo on every showcase video & report the club produces |
| **Pilot sponsor** | €2,500 | Above + dedicated processing laptop for the club + 6 months of my focused development time on the trainer dashboard | Naming on the pilot program, first access to results, demo events |
| **Program partner** | €10,000 | Above + second club rollout + Serbian localization + packaged replication kit | Partner branding, co-announcement with club(s), seat at the federation conversation |

Every euro goes to hardware and building — the software stack is
deliberately $0/month (no cloud, no licenses, open source, MIT).

---

## The honest state of the numbers

We show sponsors the same numbers we use internally:

- Technique recognition today: 26 trained techniques; on the live
  competition demo it went 2/2 with the reference catalog agreeing.
  Across the full corpus the honest cross-validated baseline is ~23%
  top-1 over 26 classes (chance = 4%) — trained mostly on heterogeneous
  internet instructionals. This number is the *floor*: the entire Phase 2
  exists because accuracy follows data, and the pilot generates exactly
  the right data (our athletes, our cameras, trainer-labeled).
- The physics does not need training: skeleton tracking, sync, face
  blurring, and (with sensors) g-force/rotation measurement work today at
  production quality.

---

## Who

**Paul Dudko** — engineer, judo parent, founder. Built the entire current
stack solo in a garage. Contact: paul.dudko@zenpulsar.com

**Judo Club Niš** — pilot club: five collaborating trainers/judges
including a former Serbian national team trainer; two committed pilot
athletes including Serbia's #1 U14.

**The first user** — an 8-year-old who might be an Olympian, whose father
refuses to let talent be limited by a club's budget.

---

*Repository: github.com/pashadude/LionOfJudo — MIT license.
Built with love for judo and kids who deserve better training tools.*
