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

#### 2.1 Upload & Storage

**Workflow:**
1. Training session ends
2. Coach triggers upload script on primary Pi
3. All 6 Pis upload H.265 files to cloud storage (AWS S3 / Google Cloud Storage)
4. Processing pipeline automatically triggered on upload completion

**Monthly Storage Cost:**
- Example: 2 hours/day, 3 days/week
- 12 GB/hour × 2 hours × 3 days × 4 weeks = 288 GB/month
- AWS S3: 288 GB × $0.023/GB = **~$6.62/month**
- Cost grows linearly with video retention (archive old sessions to Glacier for $0.004/GB)

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

#### 2.3 AI/ML Architecture

**Stage 1: Person Detection & Pose Estimation**

**YOLO-11 Pose Model**
- **Purpose:** Real-time detection of all people in frame + 17-keypoint skeletal pose
- **Why YOLO-11:**
  - State-of-the-art speed/accuracy (critical for 6 video streams)
  - Native pose estimation (no separate model needed)
  - Handles occlusions and multiple people in frame
- **Output:** Bounding boxes + skeleton coordinates for each athlete per frame

**MediaPipe Holistic (Supplementary)**
- **Purpose:** Enhanced detail for specific frames requiring fine-grained analysis
- **Use Cases:**
  - Hand grip detection (important for certain throws like morote-seoi-nage)
  - Facial landmark tracking (for pupil dilation analysis - see Stage 4)
  - Body angle refinement for scoring movement quality
- **Why Not Primary:** Too computationally expensive to run on all frames; use selectively

**Processing Strategy:**
```
For each frame from 6 cameras:
├─ YOLO-11 (always): Fast person detection + skeleton
└─ MediaPipe (conditionally): Only when:
    - Technique initiation detected (need grip detail)
    - Face visible (for pupil analysis)
    - Quality scoring required (need precise joint angles)
```

---

**Stage 2: 3D Pose Reconstruction**

**Purpose:** Combine 2D poses from 6 camera angles into single 3D skeletal model.

**Method:** Multi-view triangulation
- Use camera calibration data (intrinsic + extrinsic parameters)
- Match skeletons across views using spatial-temporal consistency
- Triangulate corresponding 2D keypoints → 3D joint coordinates
- Apply Kalman filtering to reduce jitter

**Output:** 3D skeleton trajectories for every athlete in the training space

**Critical Dependency:** Hardware-synced cameras (this is why the Pi Pico is non-negotiable)

---

**Stage 3: Technique Classification**

**Challenge:** Judo techniques are highly complex, context-dependent sequences.

**Approach: Hybrid Classification System**

**3A: Temporal Action Detection**
- **Model:** 3D CNN (e.g., X3D or SlowFast) trained on 3D skeleton sequences
- **Input:** 2-3 second windows of 3D joint trajectories
- **Output:** Technique class + temporal boundaries (start/end frames)

**Training Data Requirements:**
- Need labeled video dataset of all 120 techniques
- Minimum ~50 examples per technique from multiple athletes
- **Bootstrapping strategy:**
  - Start with publicly available judo competition footage
  - Augment with your own training sessions (manually labeled initially)
  - Active learning: Model flags uncertain predictions for manual review

**3B: Technique-Specific Classifiers**
- Some techniques are ambiguous (e.g., tai-otoshi vs. seoi-nage variations)
- Train specialized binary classifiers for confusable pairs
- Use biomechanical features:
  - Hip rotation angle
  - Foot placement patterns
  - Center-of-mass trajectory
  - Contact points between athletes

**Output Example:**
```json
{
  "timestamp": "00:03:47.2 - 00:03:48.1",
  "technique": "Ippon Seoi Nage",
  "confidence": 0.92,
  "athletes": {
    "tori": "Athlete_03",  // person executing throw
    "uke": "Athlete_07"     // person being thrown
  }
}
```

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

## Cost Summary

### One-Time Hardware Investment

| Category | Cost |
|----------|------|
| 6-Camera Sync System | $832 |
| Audio Equipment (Bluetooth mic) | ~$50 |
| **Total** | **~$882** |

### Recurring Monthly Costs

| Service | Cost (Example Usage) |
|---------|---------------------|
| Cloud Storage | ~$7/month (288 GB/month new data) |
| Cloud Compute (AI inference) | ~$20-50/month (depends on GPU pricing - estimate 10 hours GPU time/month) |
| **Total** | **~$27-57/month** |

**Note:** AI processing costs scale with training frequency. Above estimate assumes 3 sessions/week × 2 hours each.

---

## Implementation Phases & Timeline

### Phase 1: MVP (Months 1-3)
**Goal:** Prove the core concept works

**Deliverables:**
- [ ] Build and test 6-camera hardware sync (Week 1-2)
- [ ] Implement video capture and upload pipeline (Week 3-4)
- [ ] Train YOLO-11 pose model on judo footage (Week 5-8)
- [ ] Build basic 3D reconstruction pipeline (Week 9-10)
- [ ] Classify 10 most common techniques (Week 11-12)
- [ ] Create simple dashboard showing detected techniques + video clips

**Success Criteria:**
- 6 cameras recording in <10ms sync
- System recognizes 10 techniques with >80% accuracy
- Dashboard ready 6 hours after session

### Phase 2: Production System (Months 4-6)
**Goal:** Full 120-technique recognition + quality scoring

**Deliverables:**
- [ ] Expand technique library to all 120 techniques
- [ ] Implement movement quality scoring algorithm
- [ ] Add power measurement calculations
- [ ] Integrate audio capture + coach voice timestamps
- [ ] Reduce processing time to 4-5 hours

### Phase 3: Advanced Features (Months 7-9)
**Goal:** Biometrics and predictive insights

**Deliverables:**
- [ ] Integrate Polar H10 heart rate data
- [ ] Add pupil dilation tracking (if feasible)
- [ ] Build trend analysis (athlete progress over time)
- [ ] Add predictive insights ("Athlete_X likely to improve uchi-mata based on pattern similarity to Athlete_Y's development")

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

## Next Immediate Actions

1. **Order Hardware:** Purchase 1x full 6-camera setup for prototyping (~$900)
2. **Build Sync Prototype:** Test Pi Pico → 2 camera trigger (validate <10ms sync)
3. **Collect Training Data:** Film 5 hours of judo training with technique labels
4. **Set Up Cloud Pipeline:** AWS account + S3 bucket + EC2 GPU instance
5. **Test YOLO-11:** Run pose estimation on sample footage - measure accuracy/speed

---

**Document Version:** 1.0
**Last Updated:** 2025-11-11
**Owner:** LionOfJudo Team
**Status:** Planning Phase
