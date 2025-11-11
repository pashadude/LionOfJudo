# LionOfJudo: AI-Powered Judo Training Analysis System

## Project Overview

**Mission:** Build an affordable, professional-grade 6-camera synchronized video capture system that uses AI to analyze judo training sessions, providing coaches and athletes with actionable insights 4-5 hours after each training session.

**Core Value Proposition:** Transform traditional judo training into data-driven athletic development by automatically recognizing techniques, measuring movement quality and power, tracking athlete focus (pupil dilation), and monitoring health metrics—all delivered through an intuitive dashboard with video and data evidence.

---

## System Architecture

### Phase 1: Hardware & Data Capture (Edge Computing)

#### 1.1 Synchronized 6-Camera System

**Purpose:** Capture multi-angle, frame-perfect synchronized video of the training mat for 3D reconstruction and analysis.

**Hardware Components:**

| Component | Specification | Quantity | Unit Cost | Total | Purpose |
|-----------|--------------|----------|-----------|-------|---------|
| Compute Hubs | Raspberry Pi 4 (4GB) | 6 | $65 | $390 | Local video recording and upload controller |
| Cameras | Arducam Global Shutter with trigger pin | 6 | $40 | $240 | Motion-blur-free capture with hardware sync capability |
| Storage | 64GB High-Speed microSD Card | 6 | $12 | $72 | Local H.265 video buffer before cloud upload |
| Power | Official Raspberry Pi USB-C PSU | 6 | $10 | $60 | Stable power delivery (critical for sync) |
| Networking | 8-Port Gigabit Switch + Cables | 1 | $50 | $50 | Wired network for reliable bulk uploads |
| Sync Generator | Raspberry Pi Pico + GPIO wires | 1 | $20 | $20 | Master clock - sends simultaneous trigger to all 6 cameras |
| **Total** | | | | **$832** | |

**Key Technical Decision: H.265 Codec**
- 30-50% smaller file sizes vs H.264
- ~2 GB per hour per camera at 1080p 30fps
- 6 cameras × 2 hours training = 24 GB per session

**Synchronization Architecture:**
```
Pi Pico (Master Clock)
    ↓ GPIO Trigger Pulse
    ├──→ Camera 1 (Pi #1) ──┐
    ├──→ Camera 2 (Pi #2) ──┤
    ├──→ Camera 3 (Pi #3) ──┤   All start recording at
    ├──→ Camera 4 (Pi #4) ──┤   exactly the same instant
    ├──→ Camera 5 (Pi #5) ──┤   (hardware-level sync)
    └──→ Camera 6 (Pi #6) ──┘
```

**Why Hardware Sync is Critical:**
- Software sync = 1-2 second drift between cameras
- Hardware sync = <1 millisecond accuracy
- Essential for 3D pose reconstruction and fast-action analysis (throws take 0.3-0.8 seconds)

#### 1.2 Audio Capture (Trainer Voice)

**Purpose:** Capture coaching instructions for context correlation with movements.

**Implementation:**
- Bluetooth microphone worn by trainer
- Pi #1 acts as audio hub (USB Bluetooth dongle)
- Audio recording starts simultaneously with video when Pico sends trigger
- Produces time-synced audio track with Camera 1 video

**Sync Quality:** ~100ms accuracy (acceptable for verbal instructions)

#### 1.3 Biometric Data (Athletes)

**Purpose:** Track heart rate and stress levels during training.

**Recommended Hardware:**
- **10× Polar H10 Chest Straps** (~$90 each, ~$900 total)
- Industry-standard Bluetooth HR broadcasting
- Open ANT+ protocol (no reverse-engineering needed)

**Current Status:** ⚠️ **Deferred to Phase 2**
- Bluetooth sync achieves only ~1-2 second accuracy (not frame-perfect)
- Suitable for general context (e.g., "heart rate during randori") but not instant-specific analysis
- Implementation complexity: Requires managing 10 simultaneous BLE connections

**Alternative Considered (Rejected):**
- "Cheap Chinese watches" lack open data protocols
- Would require reverse-engineering proprietary Bluetooth protocols
- Sync quality would be unusable for frame-level analysis

---

### Phase 2: Cloud Processing Pipeline (AI Analysis)

This is the core intelligence layer that transforms raw synchronized video into actionable training insights.

**Architecture Decision: Hetzner + OpenRouter (Cost-Optimized)**

Instead of AWS and self-hosted ML models, we use:
- **Hetzner Storage Box:** Ultra-cheap storage (€3.81/TB/month = ~$4.20/TB/month)
- **Hetzner Dedicated Server:** Single server for processing (starting at €40/month)
- **OpenRouter API:** Vision-capable AI models (GPT-4V, Claude 3.5 Sonnet, Gemini) for recognition

This eliminates:
- ❌ Training custom ML models (months of work)
- ❌ GPU infrastructure costs ($100s/month)
- ❌ Model maintenance and updates
- ❌ Complex MLOps pipelines

#### 2.1 Upload & Storage (Hetzner)

**Workflow:**
1. Training session ends
2. Coach triggers upload script on primary Pi
3. All 6 Pis upload H.265 files via SFTP/rsync to Hetzner Storage Box
4. Upload triggers processing job on Hetzner server

**Monthly Storage Cost (Hetzner Storage Box):**
- Example: 2 hours/day, 3 days/week
- 12 GB/hour × 2 hours × 3 days × 4 weeks = 288 GB/month
- Hetzner: €3.81 per 1TB/month = **~$0.90/month for 288 GB**
- **90% cheaper than AWS S3**

**Storage Box Specs:**
- 1TB Box: €3.81/month (~$4.20/month)
- Access: SFTP, rsync, WebDAV, Samba
- Snapshots included
- Enough for ~40 training sessions before needing to archive

#### 2.2 Judo Movement Recognition (Core AI System)

**Objective:** Automatically detect and classify all judo techniques performed during training.

**Target Technique Library:**
- **80 top techniques (tachi-waza)** - standing throws and takedowns
- **40 bottom techniques (ne-waza)** - ground grappling and pins
- **120 total techniques** requiring classification

**Reference Materials:**
- Top techniques: https://www.youtube.com/watch?v=LMKgaMdm9UY&t=73s
- Bottom techniques: https://www.youtube.com/watch?v=TgwfUOWB7TQ

---

#### 2.3 AI/ML Architecture (OpenRouter-Based)

**Revolutionary Approach: Vision LLMs Replace Custom Models**

Instead of training specialized computer vision models, we leverage state-of-the-art vision-capable LLMs via OpenRouter API. This is **dramatically simpler and cheaper**.

**Why OpenRouter:**
- Access to best vision models: GPT-4V, Claude 3.5 Sonnet, Gemini 1.5 Pro
- No training data collection needed
- No GPU infrastructure
- Models already understand human movement, sports, and can follow complex instructions
- Pay only for what you use (~$0.01-0.03 per image depending on model)

**Cost-Optimized Processing Strategy:**

We don't send every frame to the API (that would be expensive). Instead:

```
For each camera video (2 hours = 216,000 frames at 30fps):
├─ Extract 1 frame per second = 7,200 frames
├─ Run cheap motion detection locally (OpenCV)
├─ Identify "action segments" (movement > threshold)
├─ Send only action frames to OpenRouter (~2,000 frames per 2-hour session)
└─ Cost: 2,000 frames × 6 cameras × $0.015/frame = ~$180/session
```

**Further Cost Optimization:**
- Use motion detection to identify only moments with athlete interaction
- Send multi-angle grid image (6 camera views in one image) = 1/6 the cost
- Use cheaper models (Gemini Flash 2.0 = $0.00001875/image!)
- **Optimized cost: ~$30-50 per session**

---

**Stage 1: Frame Extraction & Motion Detection (Local on Hetzner)**

**Process:**
1. Upload complete: 6 video files on Hetzner Storage Box
2. Hetzner server runs FFmpeg to extract frames:
   ```bash
   ffmpeg -i video.mp4 -vf fps=1 frames/frame_%04d.jpg
   ```
3. OpenCV motion detection identifies "active" periods:
   - Frame differencing between consecutive frames
   - Threshold for significant movement
   - Output: List of frame timestamps with activity
4. Extract high-res frames at key moments (technique attempts)

**Output:** ~500-2000 key frames per camera per session

---

**Stage 2: Multi-View Image Preparation**

**Problem:** Sending 6 separate images per moment = 6× API cost

**Solution:** Stitch camera views into single grid image
```
+----------+----------+----------+
| Camera 1 | Camera 2 | Camera 3 |
+----------+----------+----------+
| Camera 4 | Camera 5 | Camera 6 |
+----------+----------+----------+
```

**Benefit:** Single API call sees all angles simultaneously
- Better context for technique recognition
- 6× cost reduction
- LLM can reason about 3D positioning from multiple views

---

**Stage 3: Technique Recognition via OpenRouter**

**Prompt Engineering Strategy:**

```python
prompt = f"""
You are analyzing a judo training session from 6 synchronized camera angles.

Reference: There are 120 standard judo techniques:
- 80 tachi-waza (standing techniques): seoi-nage, tai-otoshi, uchi-mata, etc.
- 40 ne-waza (ground techniques): kesa-gatame, juji-gatame, etc.

Frame timestamp: {timestamp}
Previous context: {previous_techniques}

Analyze this multi-angle frame and identify:

1. **Technique Identification:**
   - What judo technique is being attempted? (Be specific: e.g., "ippon seoi-nage" not just "seoi-nage")
   - Who is tori (thrower) and who is uke (receiver)?
   - Confidence level (0-100%)

2. **Technique Phase:**
   - Is this: setup/kuzushi, entry/tsukuri, execution/kake, or follow-through?

3. **Movement Quality (1-10 scale):**
   - Posture and balance
   - Timing and rhythm
   - Technical correctness (compare to ideal form)

4. **Key Observations:**
   - What is done well?
   - What needs improvement?
   - Specific body positions (hip angle, foot placement, grip)

Respond in JSON format.
"""
```

**API Call:**
```python
response = openrouter.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",  # or google/gemini-flash-1.5
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": base64_image}}
        ]
    }]
)
```

**Output Example:**
```json
{
  "timestamp": "00:03:47",
  "technique": {
    "name": "Ippon Seoi Nage (One-arm shoulder throw)",
    "category": "tachi-waza",
    "subcategory": "te-waza (hand technique)",
    "confidence": 87
  },
  "athletes": {
    "tori": "Athlete in white gi (left side of camera 1)",
    "uke": "Athlete in blue gi"
  },
  "phase": "kake (execution)",
  "quality_score": {
    "overall": 7,
    "posture": 8,
    "timing": 6,
    "technical_correctness": 7
  },
  "observations": {
    "strengths": [
      "Good hip positioning below uke's center of gravity",
      "Strong kuzushi (off-balancing) in setup"
    ],
    "improvements": [
      "Entry step slightly too wide - reduces rotational power",
      "Right arm pull could be more circular rather than straight back"
    ],
    "body_mechanics": {
      "hip_angle": "~110 degrees (good)",
      "foot_placement": "Left foot between uke's feet (correct)",
      "grip": "Standard ippon grip visible"
    }
  }
}
```

---

**Stage 3B: Optional LoRA Fine-Tuning (Phase 1.5)**

**When Needed:** If base models can't detect subtle technique errors (<70% accuracy)

**The Problem:**
- Base vision LLMs know general judo techniques
- They might miss critical coaching details:
  - "Harai-goshi with hip too high" (looks like harai-goshi, but wrong)
  - "Seoi-nage without proper kuzushi" (technically correct entry, but setup failed)
  - "Uchi-mata with weak sweep" (identified correctly but quality assessment poor)

**The Solution: LoRA (Low-Rank Adaptation)**
- Fine-tune existing vision model on YOUR judo footage
- Teach it to spot specific errors common in beginner/intermediate judoka
- Much cheaper than training from scratch

**Dataset Requirements:**
- **500-1000 labeled images** (not 5 hours of video!)
- 50-100 examples per major technique
- Mix of correct and incorrect execution
- Specific error labels

**Example Training Data:**
```json
{
  "image": "harai_goshi_beginner_003.jpg",
  "technique": "harai-goshi",
  "execution_quality": "poor",
  "specific_errors": [
    {
      "error": "hip_position_too_high",
      "severity": "critical",
      "description": "Tori's hip is at same height as uke's hip. Should be 20cm lower for effective throw."
    },
    {
      "error": "timing_late",
      "severity": "moderate",
      "description": "Leg sweep initiated after uke recovered balance from kuzushi."
    }
  ],
  "correct_aspects": [
    "grip is correct",
    "foot placement acceptable"
  ]
}
```

**LoRA Training Options:**

| Approach | Cost | Effort | Inference Cost | Best For |
|----------|------|--------|----------------|----------|
| **GPT-4V Fine-tune** | $30-50 one-time | Low | $0.006/image | Quick start, works immediately |
| **Gemini Fine-tune** | TBD | Low | ~$0.00005/image | When Google releases it (cheapest) |
| **Open Source (LLaVA/Idefics)** | Free | High | Free (self-hosted) | Long-term if budget is critical |

**Recommendation:**
1. Start with base GPT-4V or Gemini Flash in Phase 0
2. If error detection < 70% → collect 500 labeled images from YouTube + your first sessions
3. Fine-tune GPT-4V with LoRA (~$40 total cost)
4. New model: "gpt-4v-judocoach-v1" specialized for your needs
5. Inference cost increase: $0.00002 → $0.006 per image (300× more expensive)
6. **But**: If this catches errors base model misses → worth it for coaching value

**LoRA Training Process:**
1. Collect 500-1000 images (1-2 weeks with help from coaches)
2. Label using simple web tool (see LORA_FINETUNING.md)
3. Upload to OpenAI fine-tuning API
4. Training takes 2-6 hours
5. Test on validation set
6. Deploy in production pipeline

**Cost-Benefit Analysis:**
- Base model: $5/session, 70% error detection
- LoRA model: $30/session, 90% error detection
- **Decision:** For wealthy clubs → use LoRA. For Serbian schools → use base model + manual coach review of flagged techniques

---

**Stage 4: Movement Quality & Power Measurement**

**4A: Quality Scoring (Technical Correctness)**

**Biomechanical Analysis:**
- Compare executed technique to reference "ideal" template
- Measure deviations in:
  - Joint angles at key phases (entry, kuzushi, tsukuri, kake)
  - Timing of movements (is the hip rotation before/after it should be?)
  - Balance metrics (center of pressure trajectory)

**Scoring Algorithm:**
```python
quality_score = weighted_average([
    joint_angle_similarity,    # 40% weight
    timing_accuracy,           # 30% weight
    balance_stability,         # 20% weight
    smoothness_of_motion      # 10% weight
])
```

**Output:** Score 0-100 with specific improvement suggestions
- Example: "Hip rotation initiated 0.12s too early - causing uke to post hand"

**4B: Power Measurement**

**Challenge:** No force plates or IMUs in this system.

**Proxy Measurements:**
1. **Throwing velocity:** Speed of uke's center of mass during throw
2. **Impact estimation:** Acceleration at landing (from pose change)
3. **Explosive strength index:** Rate of hip/shoulder acceleration during kake phase

**Calculation Example:**
```
Power Index = (throw_velocity × uke_mass_estimate) / execution_time
```

**Limitations:**
- Relative measurements only (cannot compare to other dojos without calibration)
- Mass estimation from body dimensions (not ground truth)
- Useful for tracking individual athlete progress over time

---

**Stage 5: Concentration Analysis (Pupil Dilation)**

**Purpose:** Detect fatigue and cognitive load during training.

**Implementation:**
- Use MediaPipe Face Mesh when athlete's face is visible
- Extract eye landmarks → calculate pupil diameter
- Track changes over training session

**Known Challenges:**
- ⚠️ **Lighting sensitivity:** Pupil size is primarily controlled by ambient light
- ⚠️ **Camera distance:** Accurate measurement requires close-up view (difficult in wide mat coverage)
- ⚠️ **Head orientation:** Only works when looking toward camera

**Recommended Approach:**
- Track relative changes during similar lighting conditions
- Flag periods of rapid constriction (possible fatigue/stress indicator)
- **Consider deferred to future phase** - may require dedicated eye-tracking camera

---

**Stage 6: Health Metrics Integration**

**Current Status:** Placeholder for Phase 2 biometric data

**Planned Integration:**
- Heart rate variability (HRV) from Polar H10 straps
- Aggregate HR zones during session (aerobic vs anaerobic)
- Recovery time between high-intensity bouts
- Correlate HR spikes with technique attempts (intensity indicator)

**Sync Strategy:**
- Accept ~1-2 second accuracy limitation
- Timestamp alignment using video audio track (coach's voice) as reference
- Display as continuous line graphs overlaid on session timeline

---

#### 2.4 Output Generation (Dashboard)

**Delivery Target:** Ready within 4-5 hours of session end

**Dashboard Sections:**

**1. Session Overview**
- Total training time
- Active time vs rest time
- Number of techniques attempted (by category)
- Average heart rate (when available)

**2. Technique Breakdown**
- List of all recognized techniques with timestamps
- Click any technique → see synchronized 6-camera playback
- Quality scores with specific improvement notes
- Power measurements ranked against athlete's previous sessions

**3. Athlete Performance Cards**
- Per-athlete stats:
  - Most practiced techniques
  - Best/worst quality scores
  - Power trend (improving/plateauing)
  - Fatigue indicators (pupil data if available)

**4. Video Evidence Clips**
- Auto-generated highlight reel of best throws
- Side-by-side comparisons (current attempt vs previous best)
- Slow-motion replays with skeleton overlay

**5. Training Recommendations**
- AI-generated suggestions:
  - "Athlete_03's seoi-nage hip rotation consistently early - recommend focused kuzushi drills"
  - "Athlete_07 showing fatigue after 45min (pupil constriction) - consider hydration break"

**Technical Stack:**
- Frontend: React dashboard with video player
- Backend: FastAPI serving analysis results
- Storage: Processed clips in cloud CDN for fast playback

---

## Cost Summary (Revised - Much Cheaper!)

### One-Time Hardware Investment

| Phase | Hardware | Cost |
|-------|----------|------|
| **Phase 0 (Proof of Concept)** | Use YouTube videos - no hardware | **$0** |
| **Phase 1 (MVP for 1 school)** | 3-camera sync system | **$431** |
| **Phase 3 (Scale to 5 schools)** | 5× 3-camera kits | **$2,155** |
| **Optional upgrade to 6 cameras** | +3 cameras per school | +$186/school |

### Recurring Monthly Costs (Per School)

| Service | Cost (3 sessions/week, 2 hrs each) |
|---------|-------------------------------------|
| Hetzner Storage Box (1TB) | €3.81 (~$4.20/month) |
| OpenRouter API (Gemini Flash) | ~$5-10/month |
| **Total per school** | **~$9-14/month** |

**Shared Hetzner Server (All Schools):**
- 1× Dedicated server: ~€40/month (~$44/month) shared across all schools
- Total for 5 schools: ~$44 + (5 × $12) = **~$104/month total**
- **Per school: ~$21/month**

**Phase 0 Testing Cost:**
- Process 20 YouTube videos (~10 hours of footage)
- Estimated: $10-20 total to validate the concept

---

## Implementation Phases & Timeline (Simplified)

### Phase 0: Proof of Concept (Week 1-2) - $0 Hardware Cost
**Goal:** Validate that OpenRouter can recognize judo techniques from video

**Approach:**
- Download 10-20 YouTube videos of judo training/competitions
- Extract key frames (1 per second during action)
- Test OpenRouter API with different prompts and models
- Compare Gemini Flash ($0.00001875/image) vs Claude 3.5 Sonnet ($0.004/image) accuracy

**Deliverables:**
- [ ] Python script to extract frames from video
- [ ] OpenRouter integration with technique recognition prompt
- [ ] Cost analysis: actual $ per technique recognized
- [ ] Accuracy report: can it distinguish seoi-nage from tai-otoshi?

**Success Criteria:**
- LLM correctly identifies at least 7/10 common techniques from single angles
- Cost <$5 to process 2 hours of footage
- If this fails, we pivot before buying any hardware

---

### Phase 1: Minimum Viable System (Weeks 3-6) - $280 Hardware
**Goal:** Build simplest possible working system for one dojo

**Hardware (Start with 3 cameras, not 6):**
| Component | Quantity | Unit Cost | Total |
|-----------|----------|-----------|-------|
| Raspberry Pi 4 (4GB) | 3 | $65 | $195 |
| Arducam Global Shutter | 3 | $40 | $120 |
| MicroSD Cards | 3 | $12 | $36 |
| Pi Pico (sync) | 1 | $20 | $20 |
| Power supplies | 3 | $10 | $30 |
| Basic switch + cables | 1 | $30 | $30 |
| **Total** | | | **$431** |

**Why 3 cameras?**
- Front, side, back = enough to see most techniques
- Still hardware-synced (Pi Pico to 3 cameras)
- 1/2 the cost of 6-camera system
- Can add 3 more later if it works

**Deliverables:**
- [ ] 3-camera sync system working
- [ ] Auto-upload to Hetzner Storage Box (1TB = €3.81/month)
- [ ] Frame extraction + motion detection
- [ ] OpenRouter batch processing script
- [ ] Simple HTML report: list of techniques with timestamps + quality scores

**Success Criteria:**
- Process 1-hour session in under 3 hours
- Cost <$10 per session in OpenRouter fees
- Coaches can identify their athletes' techniques in the report

---

### Phase 2: Dashboard & Multi-Athlete Tracking (Weeks 7-10)
**Goal:** Make it actually usable for daily training

**Deliverables:**
- [ ] Simple React dashboard (replace HTML report)
- [ ] Video player that jumps to technique timestamps
- [ ] Athlete identification (manual tagging at session start)
- [ ] Progress tracking: compare quality scores over time
- [ ] Export reports as PDF for parents

**No new hardware needed** - same 3-camera setup

---

### Phase 3: Scale to Multiple Schools (Months 3-6)
**Goal:** Deploy to 3-5 schools in Serbia

**Considerations:**
- Build 3-5 identical 3-camera kits
- Shared Hetzner server processes all schools' videos
- Each school uploads to their own folder
- Dashboard shows per-school and per-athlete views

**Optional Upgrades (only if budget allows):**
- Add 3 more cameras per school (total 6) for better coverage
- Audio capture (coach microphone)
- Biometrics (Polar H10 straps) - probably Phase 4

---

### Phase 4: Advanced Features (Month 6+)
**Only add if Phase 1-3 proves valuable:**
- Heart rate integration
- Comparative analysis (athlete vs athlete)
- Video highlight reels (auto-generated "best throws of the month")
- Mobile app for parents

---

## Technical Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 3D reconstruction fails with complex occlusions | High | High | Start with 2D analysis only; require minimum 3 visible cameras per athlete |
| Technique classification accuracy <70% | Medium | High | Begin with smaller technique subset; use active learning to improve dataset |
| Processing time >5 hours | Medium | Medium | Pre-compute pose estimation; optimize with quantized models; add more GPU resources |
| Pupil tracking unusable due to lighting | High | Low | Mark as experimental feature; focus on proven metrics (HR, technique count) |
| Upload bandwidth insufficient (rural dojo) | Medium | High | Add local NAS option - upload overnight when bandwidth available |

---

## Open Questions for Next Sprint

1. **Technique Labeling:** How will we create training dataset for 120 techniques?
   - Option A: Manual labeling party (time-intensive)
   - Option B: Hire judo experts on Upwork to label footage
   - Option C: Start with public competition footage + transfer learning

2. **Camera Placement:** What mat coverage pattern for 6 cameras?
   - Option A: Hexagon around mat (even spacing)
   - Option B: 3 pairs at opposite sides (better depth perception)
   - Requires physical testing in real dojo

3. **Athlete Identification:** How to track who is who across sessions?
   - Option A: Manual assignment at session start
   - Option B: Face recognition (privacy concerns?)
   - Option C: Colored belts/uniforms as visual markers

4. **Cloud vs Edge Processing:** Should we do any analysis on the Pis?
   - Current plan: Upload all raw video to cloud
   - Alternative: Run pose estimation locally, upload skeleton data only (10x smaller)
   - Trade-off: Added complexity vs reduced cloud costs

---

## Success Metrics (6 Month Target)

**Technical:**
- 6-camera system operational with <10ms sync
- Recognize 80+ judo techniques with >85% accuracy
- Process 2-hour session in <5 hours
- Dashboard accessible on mobile/desktop

**User:**
- 5 judo clubs actively using system
- Coaches report technique detection saves 2+ hours/week of manual video review
- Athletes can articulate 1 specific improvement from dashboard insights

**Business:**
- System cost <$1000 upfront per installation
- Monthly operating cost <$60/dojo
- Positive ROI for clubs with 20+ students

---

## Repository Structure (Planned)

```
LionOfJudo/
├── hardware/
│   ├── pi_camera_setup/       # Raspberry Pi video capture scripts
│   ├── pico_sync/             # Sync pulse generator code
│   └── calibration/           # Camera intrinsic/extrinsic calibration
├── cloud/
│   ├── upload_pipeline/       # Video upload and storage
│   ├── pose_estimation/       # YOLO-11 + MediaPipe inference
│   ├── reconstruction_3d/     # Multi-view triangulation
│   ├── technique_classifier/  # ML models for judo technique recognition
│   ├── quality_scorer/        # Biomechanical analysis
│   └── api/                   # FastAPI backend for dashboard
├── dashboard/
│   └── react_app/             # Frontend web application
├── training_data/
│   ├── labeled_techniques/    # Video clips with ground truth labels
│   └── models/                # Trained model checkpoints
└── docs/
    ├── PROJECT_PLAN.md        # This file
    ├── HARDWARE_SETUP.md      # Step-by-step camera system build guide
    └── API_REFERENCE.md       # Dashboard API documentation
```

---

## Next Immediate Actions (Phase 0 - Start This Week!)

**No hardware purchase until we validate the concept!**

### Week 1: Validate OpenRouter Can Recognize Judo

1. **Download Test Videos** (~2 hours)
   - Get 10-20 YouTube videos of judo training/competition
   - Mix of tachi-waza and ne-waza
   - Different camera angles and quality levels

2. **Build Frame Extraction Script** (~3 hours)
   ```bash
   # Simple FFmpeg command
   ffmpeg -i video.mp4 -vf fps=1 frames/frame_%04d.jpg
   ```

3. **Test OpenRouter API** (~4 hours)
   - Sign up for OpenRouter account (add $20 credit)
   - Test 3 models: Gemini Flash, Claude 3.5 Sonnet, GPT-4V
   - Send 100 frames with judo technique recognition prompt
   - Measure: accuracy, cost per frame, speed

4. **Create Simple Report** (~2 hours)
   - Which model is most accurate?
   - Which is cheapest?
   - Can it distinguish similar techniques (seoi-nage vs tai-otoshi)?
   - Estimated cost to process 2-hour training session

**Decision Point:** If accuracy >70% and cost <$10/session → proceed to Phase 1

### Week 2: Refine Prompt & Motion Detection

5. **Improve Prompt Engineering** (~4 hours)
   - Add examples of each technique to prompt
   - Test multi-shot prompting (show reference images)
   - Test with/without telling LLM about previous frames (context)

6. **Build Motion Detection** (~4 hours)
   - OpenCV frame differencing to find "active" segments
   - Reduce frames sent to API by 70-80%
   - Retest cost: should drop to $2-5/session

**Decision Point:** If successful → buy Phase 1 hardware ($431 for 3 cameras)

**No commitments, no hardware costs, total testing budget: $20-30**

---

**Document Version:** 1.0
**Last Updated:** 2025-11-11
**Owner:** LionOfJudo Team
**Status:** Planning Phase
