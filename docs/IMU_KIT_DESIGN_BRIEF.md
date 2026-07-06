# LionOfJudo Wearable IMU Kit — Design Brief

*For designers and illustration tools: everything needed to draw the kit,
its internals, and its placement on the judogi accurately.*

---

## The engineering story (context for the visual narrative)

We did not start here. The final kit is the survivor of a long elimination
process — worth telling visually as a "what we rejected → what we kept"
panel:

| We tried | Why we rejected it |
|---|---|
| **Wired sensors** to a Raspberry Pi Pico controller at the mat edge | Wires on a child being thrown are unsafe and unusable — the first design never left paper |
| **Live WiFi streaming** from the athlete to a router (MikroTik at the dojo) | A 2.4 GHz radio pressed between two bodies on a mat drops packets constantly; battery life collapsed from ~6 h to ~1 h; and live data buys nothing — analysis is post-training anyway |
| **Ready-made Bluetooth IMUs** (WitMotion-class, ~$30 each) | 6–8× the cost per unit, sealed batteries we can't inspect, opaque firmware, no flash logging |
| **Phone-in-pocket sensing** (the approach our scientific consultant proved for skiing) | Perfect for skiing; on a tatami a phone is a rock strapped to a child — weight, size, and impact risk are disqualifying |
| **Smaller/cheaper ESP32 boards with 4 MB flash** | Only ~35 minutes of recording — a judo session is 90+ |
| **Unprotected LiPo cells** (cheapest option) | Never on a child. Protected cells with a cutoff circuit, or LiFePO4 chemistry, only |

**What survived: a $8-per-unit, matchbox-sized, wireless, flash-logging
capsule with zero infrastructure at the dojo.** Switch it on, train,
plug it into USB at home — done.

---

## The kit — exact contents and dimensions

One athlete wears **two identical units** (chest + hip). Each unit:

| # | Part | Size (L×W×H) | Visual description |
|---|---|---|---|
| 1 | **ESP32-S3 SuperMini** board | 22 × 18 × 4 mm | Matte-black PCB, silver postage-stamp WiFi antenna area at one end, USB-C port on the edge, tiny onboard RGB LED |
| 2 | **MPU-6050 motion sensor** (GY-521 board) | 21 × 16 × 3 mm | Deep-blue PCB, small square black IC in the center, one row of gold pin holes |
| 3 | **Battery: protected LiPo 502030, 250 mAh** | 30 × 20 × 5 mm | Flat silver foil pouch, thin red+black wire pair, small white JST-PH plug |
| 4 | **TP4056 USB-C charging board** | 26 × 17.5 × 4 mm | Blue PCB, USB-C on the short edge, two tiny LEDs (red = charging, blue/green = full) |
| 5 | **Mini slide switch** | 12 × 6 × 5 mm | Black with silver toggle, mounted flush at the capsule's rim |
| 6 | Silicone wiring (30 AWG) + heat-shrink | — | Short flexible red/black/yellow/green leads, ~30 mm |

**Assembled electronics stack:** boards sandwiched face-to-face with the
battery flat underneath → **~45 × 22 × 12 mm, ≈15 g** (lighter than a AA
battery).

### The capsule (the hero object)

- **Rigid polypropylene pill-capsule**, cylinder with rounded ends:
  **~50 mm long × 26 mm diameter**, twist-open in the middle
- Interior fully lined with **2 mm translucent silicone sheet**; the stack
  is wrapped so nothing rattles (a loose sensor smears the measurements —
  snugness is functional, not cosmetic)
- Two **25 mm silicone discs** cushion the ends
- Exterior: smooth, matte; one **8 mm hole** for the slide switch, one
  **light-pipe dot** where the LED shines through
- Brand mark: small lion-head silhouette + "LionOfJudo" in the brand gold

**Wiring for the exploded view:** battery → JST plug → slide switch →
TP4056 (charging tap) → ESP32 power; four thin leads ESP32↔MPU-6050
(3V3, GND, SDA→GPIO8, SCL→GPIO9); everything folds into the capsule.

---

## Placement on the judogi — the two positions

Draw a young judoka (8–12 y.o.) in a white gi with a colored belt,
front view, with two glowing markers:

1. **CHEST unit — on the sternum**, centered, sitting *under* the crossed
   lapels of the gi jacket. The lapels naturally cover and pad it; from
   outside only a slight rectangular contour is visible. Rationale to
   caption: judo falls land on the back and side — the sternum position
   is protected by technique itself, and the crossed lapels add a double
   layer of thick cotton.
2. **HIP unit — front of the belt, tucked behind the knot**, slightly to
   one side of it. The belt's four+ layers of dense cotton are the armor.
   Caption rationale: the hip is the engine of every throw — this sensor
   measures the rotation that scores ippons.

**Never drawn / explicitly wrong:** anything on the spine or lower back
(direct mat-impact zone), shoulders, or limbs. If an "incorrect placement"
diagram is wanted, cross out a unit on the lower back in red.

### The pocket (cross-section drawing)

A sewn-in pocket on the gi's inner side at each position,
**~70 × 50 mm**, with a velcro flap. Layer stack, inside-out:

```
child's body
  │  3 mm silicone pad (60 × 40 mm) — impact spreader
  │  capsule (lying horizontally)
  │  gi fabric pocket wall
  │  velcro flap closure
outer gi / lapel / belt covers everything
```

---

## LED language (for state illustrations)

| LED | Meaning | Scene to draw |
|---|---|---|
| Slow **green** blink | At home, on WiFi, ready to download | Capsule on a desk beside a laptop, USB-C plugged |
| **Amber** countdown blink | 10 s until recording starts | Athlete's hand switching the unit on in a gym bag |
| Dim green heartbeat + **blue flash on impact** | Recording; blue = a throw just registered | Mid-throw action shot, tiny blue glow at the chest |
| **Purple** blink | Sensor fault (check wiring) | Bench/repair context only |
| **Red** | Battery low / stopping safely | — |

---

## Usage sequence (four-panel storyboard)

1. **Home, evening:** two capsules charging on USB-C, green LEDs, laptop
   in background showing the analysis dashboard.
2. **Dojo:** parent slips capsules into the gi pockets; kid does the
   3-jump sync ritual in front of a small camera tripod. No cables, no
   equipment on the mat.
3. **Training:** normal randori; a faint blue flash at the chest as a
   throw lands. Nobody is holding a phone; nothing distracts.
4. **Home, night:** capsules back on USB, MacBook processing; morning
   report on screen: throw clips, technique names with probability bars
   (gold), g-force numbers.

---

## Visual identity

- **Brand gold:** RGB 255, 220, 60 (the probability-bar gold from the
  recognition panel)
- **Panel dark:** near-black blue-grey, RGB 10, 14, 18
- Skeleton-overlay green (RGB 0, 255, 0) may appear in screen mockups
- Typography in mockups: clean monospace or geometric sans; the product
  is honest engineering, not a toy — avoid cartoon styling for the
  hardware itself, though the judoka may be stylized
- Tone: "garage-built, lab-grade." Show real materials — PCBs, silicone,
  cotton gi weave — not sci-fi gloss.

## Requested illustrations (priority order)

1. **Exploded view** of one unit: capsule halves + silicone lining +
   electronics stack + battery, parts labeled with the dimensions table
2. **Judoka front view** with the two placement markers and captions
3. **Pocket cross-section** (the layer stack above)
4. **Four-panel usage storyboard**
5. **"What we rejected" strip**: five crossed-out alternatives (wires /
   streaming radio / $30 BLE puck / phone / small-flash board) → arrow →
   the final capsule with "$8 · 15 g · 6 h · zero dojo setup"
