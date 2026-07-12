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
| **Fixed 3-camera Raspberry Pi rig with a wired sensor hub** (the original paper design: 3× Pi 4 + Arducams + Pi Pico hub) | $431 of dedicated hardware, an evening of setup per session, and wires that cannot follow a moving athlete — never built |
| **Cloud processing** (Hetzner servers, in the original plan) | Monthly costs forever, and children's footage leaving the building — replaced by a laptop that processes overnight, fully offline |
| **Live WiFi streaming** from the athlete to a router (MikroTik at the dojo) | Packets die between two bodies on a mat; ~6× the battery drain; and live data buys nothing — analysis is post-training anyway |
| **Ready-made Bluetooth IMUs** (WitMotion-class, ~$30 each) | 6–8× the cost per unit, sealed batteries we can't inspect, opaque firmware, Bluetooth sync drift between units |
| **Phone-in-the-gi sensing** (the approach our scientific consultant conceived and proved for skiing) | Perfect for skiing; wrong for an impact sport — a phone is too big, too heavy, and too expensive to sacrifice on a tatami |
| **Smaller/cheaper ESP32 boards with 4 MB flash** | Only ~35 minutes of recording — a judo session is 90+ |
| **Unprotected LiPo cells** (cheapest option) | Never on a child. Protected cells with a cutoff circuit, or LiFePO4 chemistry, only |

**What survived: a $8-per-unit, matchbox-sized, wireless, flash-logging
capsule with zero infrastructure at the dojo.** Switch it on, train, then
pull logs over the local WiFi after training. The LiPo is charged separately
with an external charger — not through the capsule.

---

## The kit — exact contents and dimensions

One athlete wears **two identical units** (chest + hip). Each unit:

| # | Part | Size (L×W×H) | Visual description |
|---|---|---|---|
| 1 | **ESP32-S3 SuperMini** board | 22 × 18 × 4 mm | Matte-black PCB, silver postage-stamp WiFi antenna area at one end, USB-C port on the edge, tiny onboard RGB LED |
| 2 | **MPU-6050 motion sensor** (GY-521 board) | 21 × 16 × 3 mm | Deep-blue PCB, small square black IC in the center, one row of gold pin holes |
| 3 | **Battery: protected LiPo 502030, 250 mAh** | 30 × 20 × 5 mm | Flat silver foil pouch, thin red+black wire pair, small white JST-PH plug |
| 4 | **TP4056 USB-C charging board** | 26 × 17.5 × 4 mm | External charger only; it is not inside the capsule. Its charge current must be reduced from a generic 1 A setting to the cell's rated limit. |
| 5 | **Mini slide switch** | 12 × 6 × 5 mm | Black with silver toggle, mounted flush at the capsule's rim |
| 6 | Silicone wiring (30 AWG) + heat-shrink | — | Short flexible red/black/yellow/green leads, ~30 mm |

**Candidate internal layout:** one protected LiPo, ESP32-S3, and GY-521 in a
dry-fit rectangular enclosure. The IMU PCB is flat and rigidly fixed to the
case floor; the LiPo is electrically isolated from solder joints and lightly
cushioned. Do not draw a soft silicone-suspended IMU or a TP4056 inside.

### The capsule (the hero object)

- **Rigid rectangular polypropylene enclosure**, candidate outer dimensions
  **64 × 44 × 20 mm**. This is a provisional fit: draw it only as a clear
  rectangular box until a real dry-fit confirms the internal clearance.
- The 3 mm tattoo-silicone sheet is a separately visible **body-facing impact
  spreader pad** in the pocket. Inside the box, use small silicone pieces only
  at the battery/cover and loose-component interfaces; the GY-521 must be
  rigidly attached to the case.
- Exterior: smooth; one correctly sized cutout for the slide switch and one
  light-pipe dot where the onboard LED shines through. There is no USB-C
  cutout in v1 because the charger remains external.
- Brand mark: small lion-head silhouette + "LionOfJudo" in the brand gold

**Wiring for the exploded view:** one protected battery → JST plug → slide
switch → ESP32 `5V/VIN/VBUS`; battery - → common GND. The TP4056 is shown
outside the capsule and connects to the unplugged battery only for charging.
Four short leads go ESP32↔MPU-6050 (`3V3`, `GND`, `SDA→GPIO8`,
`SCL→GPIO9`). A 100k/100k divider from battery + to GPIO4 plus a 100 nF
capacitor from GPIO4 to GND is part of the battery-monitor drawing.

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
2. **HIP unit — under the belt, in a pocket sewn to the judogi itself**
   (front or back panel of the gi at hip level, close to the body). Not
   attached to the belt: **the belt rotates and shifts during randori,
   while the gi panel stays with the body** — a belt-mounted sensor would
   measure belt slip, not hip rotation. The belt worn over the pocket
   still acts as armor (four+ layers of dense cotton).
   Caption rationale: the hip is the engine of every throw — this sensor
   measures the rotation speed (°/s), hip drive, and kuzushi timing that
   score ippons.
   **Draw the marker on the gi beneath the belt line, not on the belt.**

**Never drawn / explicitly wrong:** anything on the spine or lower back
(direct mat-impact zone), shoulders, or limbs. If an "incorrect placement"
diagram is wanted, cross out a unit on the lower back in red.

### The pocket (cross-section drawing)

A sewn-in pocket on the gi's inner side at each position,
**~70 × 50 mm**, with a velcro flap. Layer stack, inside-out:

```
child's body
  │  gi fabric pocket wall
  │  3 mm tattoo-silicone pad (about 60 × 40 mm) — impact spreader
  │  rigid capsule (lying horizontally)
  │  velcro flap closure
outer gi / lapel / belt covers everything
```

---

## LED language (for state illustrations)

| LED | Meaning | Scene to draw |
|---|---|---|
| Slow **green** blink | On JudoNet WiFi, ready to download | Capsule on a desk beside the Pi/Mac, battery connected |
| **Amber** countdown blink | 10 s until recording starts | Athlete's hand switching the unit on in a gym bag |
| Dim green heartbeat + **blue flash on impact** | Recording; blue = a throw just registered | Mid-throw action shot, tiny blue glow at the chest |
| **Purple** blink | Sensor fault (check wiring) | Bench/repair context only |
| **Red** | Battery low / stopping safely | — |

---

## Usage sequence (four-panel storyboard)

1. **Home, evening:** two disconnected batteries charging in external
   protective bags with external TP4056 chargers; capsules and laptop are
   nearby for later log download.
2. **Dojo:** parent slips capsules into the gi pockets; after the logging
   countdown and 5 seconds still, athlete does the 3-event physical sync
   ritual in front of both cameras. No cables, no equipment on the mat.
3. **Training:** normal randori; a faint blue flash at the chest as a
   throw lands. Nobody is holding a phone; nothing distracts.
4. **Home, night:** capsules power on beside the Pi while it pulls logs;
   MacBook processes the copied session. Morning report shows throw clips,
   technique names with probability bars (gold), and inertial indices.

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
5. **"What we rejected" strip**: six crossed-out alternatives (fixed
   Pi-camera rig with wired hub / cloud servers / streaming radio /
   $30 BLE puck / phone / small-flash board) → arrow → the final capsule
   with "$8 · 15 g · 6 h · zero dojo setup"
