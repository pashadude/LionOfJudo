# 🥋 LionOfJudo: AI-Powered Judo Training Analysis

**Mission:** Bring professional sports science analysis to underfunded judo schools in Serbia (and beyond) using affordable hardware and AI.

**Status:** Planning Phase → Ready for Phase 0 Testing

---

## What This System Does

Automatically analyzes judo training sessions to provide:

1. **Technique Recognition** - Identifies all 120 standard judo techniques (80 standing, 40 ground)
2. **Quality Scoring** - Rates execution quality with specific improvement suggestions
3. **Error Detection** - Spots common mistakes (hip height, timing, foot placement, etc.)
4. **Progress Tracking** - Shows athlete improvement over time
5. **Video Evidence** - Clips with synchronized multi-angle playback

**Delivered 4-5 hours after training session via web dashboard**

---

## Cost Breakdown

### One-Time Hardware (Start Small!)

| Phase | Hardware | Cost |
|-------|----------|------|
| **Phase 0: Testing** | Use YouTube videos only | **$0** |
| **Phase 1: MVP (1 school)** | 3-camera sync system | **$431** |
| **Phase 3: Scale (5 schools)** | 5× 3-camera kits | **$2,155** |

### Monthly Operating Costs

| Service | Cost (per school) |
|---------|-------------------|
| Hetzner Storage (1TB) | ~$4/month |
| OpenRouter API | ~$5-10/month |
| Shared Hetzner server | ~$9/month |
| **Total** | **~$21/month** |

**Per Session:** $5-10 using base Gemini Flash, or $30 with fine-tuned model

---

## How It Works

### 1. Hardware: 3-Camera Synchronized Capture

```
┌─────────────────────────────────────────┐
│   Pi Pico (Master Sync Clock)          │
│   Sends pulse 30 times/second          │
│   + Reads IMU sensors (optional)       │
└────────┬─────────┬──────────┬───────────┘
         │         │          │
    ┌────▼───┐ ┌──▼────┐ ┌───▼────┐
    │ Pi #1  │ │ Pi #2 │ │ Pi #3  │
    │ Cam 1  │ │ Cam 2 │ │ Cam 3  │
    │(Front) │ │(Side) │ │(Back)  │
    └────────┘ └───────┘ └────────┘
         │         │          │
         └─────────┴──────────┘
                   │
          Ethernet Switch
                   │
         Upload to Hetzner Storage
```

**Key Feature:** Hardware-synced cameras (<10ms accuracy) for frame-perfect multi-angle analysis

**Optional: Add IMU Sensors (+$31)**
- Attach accelerometers to judogi (chest + hip)
- Direct power measurement (g-force, not estimated!)
- Auto-sync via acceleration spikes (throws = 5-8g impacts)
- See [ACCELEROMETER_SYSTEM.md](ACCELEROMETER_SYSTEM.md)

### 2. Cloud Processing: AI Recognition Pipeline

```
Videos uploaded → Hetzner server
    ↓
Extract frames (1/sec) → ~7,200 frames
    ↓
Motion detection → Filter to ~2,000 action frames
    ↓
Stitch 3 views into grid → Reduce API cost by 67%
    ↓
OpenRouter Vision API → Technique recognition + quality scoring
    ↓
Generate dashboard → Video clips + athlete progress charts
```

**Models Used:**
- **Base:** Gemini Flash 1.5 ($0.00002/image) - General technique recognition
- **Fine-tuned:** Custom GPT-4V ($0.006/image) - Detects subtle errors specific to your coaching

### 3. Output: Coaching Dashboard

**For Each Session:**
- List of all techniques attempted with timestamps
- Quality scores (1-10) with specific improvement notes
- Click any technique → see synchronized 3-camera playback
- Per-athlete progress tracking
- Exportable PDF reports for parents

---

## Quick Start

### Phase 0: Test the Concept (This Week!)

**Goal:** Validate AI can recognize judo techniques BEFORE buying hardware

**Investment:** $0 hardware, $20 API testing

```bash
# 1. Clone the repo
git clone https://github.com/pashadude/LionOfJudo.git
cd LionOfJudo

# 2. Install dependencies
pip install yt-dlp opencv-python openai

# 3. Run Phase 0 testing (see PHASE_0_TESTING.md)
# Download YouTube videos → Test OpenRouter API → Measure accuracy & cost

# 4. Decision: If >70% accuracy → Proceed to Phase 1
```

**See:** [PHASE_0_TESTING.md](PHASE_0_TESTING.md) for detailed guide

---

### Phase 1: Build 3-Camera System (If Phase 0 Succeeds)

**Investment:** $431 hardware, ~$21/month operating

**See:** [HARDWARE_SETUP.md](HARDWARE_SETUP.md) for step-by-step build instructions

**What You Get:**
- 3 Raspberry Pi 4 + Arducam cameras
- Pi Pico sync generator
- Capture scripts
- Upload scripts to Hetzner
- Basic HTML report with technique list

---

### Optional: Fine-Tune for Error Detection

**If base model can't spot subtle errors (hip height, timing, etc.):**

```bash
# BREAKTHROUGH: Automatic labeled dataset creation!
python3 create_training_dataset.py

# This script:
# 1. Downloads YouTube videos with 120 techniques
# 2. Uses OCR to read technique names from video text
# 3. Extracts ~1200 labeled images in 10 minutes
# 4. Creates training file for OpenAI fine-tuning
```

**Cost:** $30-50 one-time training
**Result:** Model that knows YOUR students' common mistakes

**See:** [LORA_FINETUNING.md](LORA_FINETUNING.md) for complete guide

---

### Optional: Add Accelerometer Sensors (+$31)

**Transform from video analysis to professional sports science!**

```bash
# Hardware needed:
# 4× MPU-6050 IMU sensors ($14 total)
# Pi Pico already handles camera sync - just add I2C connection
# Velcro + cables ($17)
```

**What you get:**
- **Direct power measurement:** 6.2g impact (not estimated from video!)
- **Auto-sync:** Acceleration spikes perfectly match video frames
- **Throw signatures:** Each technique has unique g-force profile
- **Coaching feedback:** "Excellent 7.2g impact, 450°/s hip rotation"

**Total system with accelerometers: $462 (only $31 more than video-only!)**

**See:** [ACCELEROMETER_SYSTEM.md](ACCELEROMETER_SYSTEM.md) for complete guide + Pi Pico code

---

## Documentation

| Document | Purpose |
|----------|---------|
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | Complete system architecture & cost analysis |
| [PHASE_0_TESTING.md](PHASE_0_TESTING.md) | Test AI recognition with $0 hardware investment |
| [HARDWARE_SETUP.md](HARDWARE_SETUP.md) | Build 3-camera synchronized capture system |
| [LORA_FINETUNING.md](LORA_FINETUNING.md) | Fine-tune AI for error detection |
| [YOLO_POSE_GUIDE.md](YOLO_POSE_GUIDE.md) | YOLO11 pose-based recognition (3 approaches) |
| [ACCELEROMETER_SYSTEM.md](ACCELEROMETER_SYSTEM.md) | IMU sensors for power measurement & auto-sync |
| `judo_pose_recognition.py` | Pure YOLO11 pose analysis |
| `judo_hybrid_recognition.py` | YOLO + Vision LLM hybrid (best accuracy) |
| `pico_unified_controller.py` | Pi Pico code for camera sync + sensors |
| `create_training_dataset.py` | Auto-generate labeled dataset from YouTube |

---

## Technology Stack

**Hardware:**
- Raspberry Pi 4 (4GB) - Video capture & upload
- Arducam Global Shutter - Motion-blur-free cameras
- Raspberry Pi Pico - Hardware sync generator

**Storage:**
- Hetzner Storage Box (1TB) - €3.81/month
- Hetzner Dedicated Server - ~€40/month shared

**AI/ML:**
- OpenRouter API - Vision model access
- Gemini Flash 1.5 - Base recognition ($0.00002/image)
- GPT-4V Fine-tuned - Error detection ($0.006/image)

**Software:**
- Python + OpenCV - Video processing
- FFmpeg - Frame extraction
- Tesseract - OCR for auto-labeling
- React - Dashboard frontend
- FastAPI - Backend API

---

## Why This Approach?

### Compared to Traditional Systems

**Traditional sports analysis systems:**
- Cost: $10,000-50,000 per installation
- Custom ML models require months of training
- GPU infrastructure: $500+/month
- Limited to wealthy clubs and national teams

**LionOfJudo:**
- Cost: $431 hardware + $21/month
- Use existing vision LLMs (no training needed)
- Optional fine-tuning: $40 one-time
- Accessible to underfunded schools

### Key Innovations

1. **Hardware Sync on Budget:** Pi Pico triggers cameras simultaneously (<$20)
2. **OCR for Auto-Labeling:** Extract labels from YouTube videos (10 min vs weeks)
3. **Vision LLMs:** No custom ML training needed
4. **Hetzner Hosting:** 90% cheaper than AWS
5. **Start Small:** 3 cameras (not 6) for half the cost

---

## Roadmap

- [x] **Phase 0:** Design system architecture (DONE)
- [ ] **Phase 0.1:** Test OpenRouter on YouTube videos (THIS WEEK)
- [ ] **Phase 0.2:** Auto-create labeled dataset with OCR (10 MINUTES)
- [ ] **Phase 1:** Build 3-camera prototype (4 weeks)
- [ ] **Phase 1.5:** Fine-tune model if needed ($40, 1 week)
- [ ] **Phase 2:** Dashboard with video playback (4 weeks)
- [ ] **Phase 3:** Deploy to 3-5 Serbian schools (3 months)
- [ ] **Phase 4:** Add biometrics, audio, advanced features (6+ months)

---

## Contributing

This project is built to help kids in underfunded schools. Contributions welcome!

**Ways to Help:**
- Test Phase 0 scripts on your own judo videos
- Improve OCR accuracy for technique name extraction
- Optimize motion detection algorithm
- Design dashboard UI/UX
- Translate to Serbian (Cyrillic)
- Donate hardware to Serbian schools

---

## Success Stories (Coming Soon!)

_This section will showcase schools using the system and athlete improvements._

---

## Contact & Support

- **GitHub Issues:** [Report bugs or request features](https://github.com/pashadude/LionOfJudo/issues)
- **Discussions:** [Ask questions or share ideas](https://github.com/pashadude/LionOfJudo/discussions)

---

## License

MIT License - See [LICENSE](LICENSE) file

**Built with love for judo and kids who deserve better training tools.**

---

## Acknowledgments

- **Inspiration:** Professional sports science labs that cost $100k+
- **Goal:** Make the same analysis available for $500
- **Motivation:** Every kid should have access to quality coaching feedback, regardless of their school's budget

**"The best time to plant a tree was 20 years ago. The second best time is now."**

Let's give Serbian judo schools the tools they deserve. 🥋
