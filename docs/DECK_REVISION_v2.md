# Deck revision v2 — slide-by-slide instructions

Based on founder feedback on LionOfJudo.pdf (13 slides, July 2026).
Copy blocks are paste-ready. Slide numbers refer to the current PDF.

---

## NEW SLIDE after Slide 2 (The Story) — "The Problem"
*(feedback #2: the problem slide must come right after the warm story)*

**Headline:** `One coach. Twenty kids. Zero instruments.`

Four audience cards (same card style as "hard to copy" slide):

**PARENTS**
> You pay for years of training and see none of the progress. Is your
> child developing? You get an opinion, never a measurement.

**TRAINERS & CLUBS**
> A coach cannot watch every kid at the same moment — most repetitions
> happen unobserved. And eyes can't see inside the gi: who threw with
> real power, who is coasting, who is one bad landing from injury.

**FEDERATIONS & JUDGES**
> Throw judgement is a real, live problem — ippon or waza-ari decides
> careers, and judges rule on what they glimpsed. Talent identification
> runs on anecdotes, not data.

**AND MORE**
> Competition preparation, overtraining detection, remote coaching,
> parent engagement — every one starts with the same missing ingredient:
> measurement.

Footer line: `LionOfJudo gives all four the same thing: objective data
from every session, at a price a club in Niš can pay.`

---

## Slide 6 (Market gap) — one number check
Change "Over 50 million judoka" → **"40M+ judoka (IJF)"** — the IJF's own
public figures range 20–40M; 40M+ is the defensible ceiling. Keep the
rest.

---

## Slide 8 (Business model) — draw the hard line
*(feedback #1: free vs paid was blurry; justify the €1,000)*

Keep 3 cards but make the boundary explicit:

**DIY tier — software free forever (MIT)**
> Software costs nothing, always. A family pays only the **parts at
> cost: ~€40 per sensor capsule** (we publish the shopping list and
> build guide). Your laptop, your cameras. For parents, garage builders,
> poor schools.

**Club kit — €1,000/year** *(this is where the €1,000 from the market
slide is earned — itemize it):*
> - 2 dedicated cameras + tripods (~€400, year one)
> - 4 assembled, tested sensor capsules — no soldering (~€200)
> - Cloud processing & storage server, managed (~€200/yr)
> - Setup afternoon + trainer onboarding + support (~€200)
>
> Line: **"The club buys an outcome, not a project."**

**Federation tier — custom, later**
> Multi-club rollouts, athlete registries, season analytics,
> talent-scouting dashboards. Priced against the $20k+ labs.

Keep the funnel/dataset-engine footer sentence — it's good.

---

## Slide 9 (Five dead ends) — factual rewrite
*(feedback #6: some items were never "on the kid"; reasons must be the
real engineering reasons)*

**Headline:** `Six dead ends, one survivor.`

Replace the list with (strikethrough style kept):

- ✕ **Fixed 3-camera Raspberry Pi rig + wired sensor hub** (the original
  paper design) — $431 of dedicated hardware, an evening of setup per
  session, and wires that can't follow a moving athlete. Never built.
- ✕ **Cloud processing** (Hetzner, in the original plan) — monthly costs
  forever, and children's footage leaving the building. Replaced by a
  laptop that processes overnight, offline.
- ✕ **Live WiFi streaming from the athlete** — packets die between two
  bodies on a mat; 6× the battery drain; and live data helps nobody —
  analysis happens after training anyway.
- ✕ **Commercial BLE sensor pucks (~$30/unit)** — 6–8× our unit cost,
  closed firmware, sealed batteries we can't inspect, and Bluetooth sync
  drift between two units.
- ✕ **Phone in the gi** — proven for skiing (our consultant's app), wrong
  for an impact sport: too big, too heavy, too expensive to sacrifice on
  a tatami.
- ✕ **4 MB flash boards** — 35 minutes of recording; a session is 90+.

Survivor panel: unchanged (capsule, $8 · 15 g · 6 h · zero dojo setup).

---

## Slide 10 (Placement) — hip position correction + what each sensor gives
*(feedback #4: the belt rotates during randori — the hip unit goes UNDER
the belt, in a pocket sewn to the judogi itself)*

**Chest — on the sternum** (unchanged)
> Under the crossed lapels; falls land on the back and side, technique
> itself protects it.
> **Gives:** impact g-force on landing, throw signature & timing, ukemi
> quality.

**Hip — under the belt, sewn to the gi** *(corrected)*
> A silicone-lined cotton pocket sewn to the judogi panel at the hip,
> worn **under the belt** — front or back side of the gi, close to the
> body. Not attached to the belt: **the belt rotates and shifts in
> randori; the gi panel stays with the body.** The belt above it still
> acts as armor.
> **Gives:** hip rotation speed (°/s) — the engine of every throw; hip
> drive and kuzushi timing; entry mechanics that video can't see through
> bodies.

Keep the red ✕ "Never on the spine or lower back."
**Illustration change:** move the second gold marker from the belt knot
to the gi at hip level, drawn *beneath* the belt line.

---

## Slide 11 (Team) — one attribution fix
*(feedback #3)*
In Milan's card, the skiing app sentence must credit the idea to him:
> Built the phone-based skiing accelerometer app — **his idea**, the
> same IMU-motion approach, already proven on the slopes.

(The rest of the card — full professor, h-index, ZENPULSAR — stays.)

---

## Slide 12 (Roadmap) — compress
*(feedback #5: timing was too conservative. Software is finished and
verified; the founder ships fast; trainers are committed. New timeline
starts the day parts arrive.)*

**Headline:** `One raise, nine months, five stages.`

- **Weeks 1–2 — Units live.** Assemble, bench-test, garage dress
  rehearsal with Lev.
- **Weeks 2–6 — Instrumented pilot.** Sessions with the 2 pilot athletes
  at JC Niš; first measured-power reports to the trainers.
- **Weeks 4–10 — Dataset sprint.** Trainers label with the one-keypress
  tool; recognition on our athletes reaches production grade.
- **Months 3–5 — Trainer product.** Dashboards, quality scoring, Serbian
  reports, parent exports; kits for 10+ athletes.
- **Months 5–9 — Ne-waza, next-gen model, second club.** Ground-work
  recognition, temporal neural network, replication kit, federation
  conversation, published results.

---

## Slide 13 (The Ask) — single ask, no tiers
*(feedback #7, #8, #9: remove tier cards; add the spend table; state the
founder's sunk investment; one number with the pre-seed sweetener)*

**Kicker:** `THE ASK — ONE NUMBER, ONE TIME`

**Headline:** `€30,000 for the full nine-month program.`

**Left column — skin in the game:**
> I have already funded discovery myself: **over $1,000 in parts,
> equipment and spares — and 300+ hours of my professional time.** At my
> market rate as an AI product executive that is **€50,000+ already
> invested.** The exploration risk is spent and paid for. What I'm
> raising now buys **execution of a proven system**, not experiments.

**Right column — where every euro goes:**

| Line | € |
|---|---|
| Sensor & camera hardware (10 athlete kits + spares, 2 cameras) | 2,500 |
| Club processing station + cloud storage | 2,000 |
| Development time, 9 months focused (dashboard, quality scoring, ne-waza, neural model) | 15,000 |
| Trainer collaboration: labeling & filming sessions, competition data trips | 3,500 |
| Second-club rollout + Serbian localization | 2,000 |
| Replication-kit productization (capsule mold, assembly docs, safety testing) | 3,000 |
| Contingency | 2,000 |
| **Total** | **30,000** |

**The backer's terms (bottom, gold):**
> Backing the program now guarantees a **25% discount at the pre-seed
> round when LionOfJudo enters its commercial stage.** You are not
> donating — you are first in line.

Footer: contacts unchanged. Keep "Back the pilot →".

---

## Consistency checklist after edits
- Slide 6 "under €1,000/yr" ↔ Slide 8 Club-kit itemization — now match.
- Slide 2 keeps "$10k–50k" labs ↔ Slide 6 table — unchanged, fine.
- Any remaining "12 months" wording → "9 months".
- Any remaining "€25,000" → "€30,000".
