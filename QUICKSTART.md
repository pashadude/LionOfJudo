# LionOfJudo - Quick Start Guide

## ⚡ 5-Minute Setup

### 1. Install Dependencies
```bash
uv venv
source .venv/bin/activate
uv pip install opencv-python ultralytics numpy
```

### 2. Run Analysis

**Basic Analysis:**
```bash
# Analyze single video
python3 phase0_judo_analysis.py examples/o-soto-gari-train.mov

# View results
cat analysis/results/o-soto-gari-train_report.txt
```

**Visual Analysis (with skeleton overlay):**
```bash
# Generate annotated videos
python3 phase0_visual_analysis.py examples/o-soto-gari-train.mov

# Output: analysis/visual_results/*_cam*_annotated.mp4
```

**Batch Process All Videos:**
```bash
# Process everything in examples/ and data/Lav/
python3 batch_process.py

# View combined report
cat analysis/combined_report.txt
```

---

## 📁 Test Videos

### Examples Folder (Rubber Band Training)
- `o-soto-gari-train.mov` - O-soto-gari with bands (3 cameras)
- `o-goshi-train.mov` - O-goshi hip throw (3 cameras)
- `ippon-seoi-nagi-train.mov` - One-arm shoulder throw (3 cameras)

### Data/Lav Folder (Pupil Performance)
- `o-soto-gari.mov` - O-soto-gari technique (20MB, 2-3 cameras)
- `ippon_seoi_nagi.mov` - Ippon seoi-nagi (48MB, 3 cameras)
- `o-soto-gari-x-uki-goshi.mov` - Two techniques (41MB, 2 cameras)

---

## 📊 What You Get

### Text Report
```
============================================================
JUDO VISUAL ANALYSIS REPORT
============================================================

Video: o-soto-gari-train
Duration: 3.05s
Layout: 3h (3 cameras)
Poses detected: 183

============================================================
MOVEMENT DETECTION
============================================================

✓ Detected 2 movement(s)

1. Movement at 0.50s - 1.83s
   Duration: 1.33s
   Camera: 1 | Person: 0

   Biomechanics:
     • Hip drop: 87.3 pixels
     • Avg hip height: 423.5px
     • Avg torso angle: 28.7°
     • Avg knee angle: 145.3°
```

### JSON Data
- Full pose keypoints for all frames
- Biomechanical measurements
- Movement timing and features
- Per-camera, per-person tracking

### Annotated Videos
- Skeleton overlay on each frame
- Color-coded body parts:
  - **Blue** = Head
  - **Yellow** = Arms
  - **Magenta** = Hips
  - **Green** = Legs
- Real-time measurement display
- Frame numbers and camera labels

---

## 🎯 Key Features

✅ **Multi-Camera Support** - Auto-splits 2-cam and 3-cam layouts
✅ **YOLO11 Pose Estimation** - 17 keypoints per person
✅ **Biomechanical Measurements** - Hip height, angles, movement
✅ **Visual Feedback** - Skeleton overlays with measurements
✅ **Movement Detection** - Automatic technique attempt detection
✅ **Batch Processing** - Process entire folders at once

---

## 🔧 Customization

### Adjust Movement Detection Sensitivity

Edit `phase0_visual_analysis.py`, line 229:
```python
if hip_drop > 30:  # Lower = more sensitive (e.g., 20)
```

### Change YOLO Confidence

Edit line 183:
```python
conf=0.3,  # Lower = more detections (0.2-0.5 recommended)
```

---

## 🐛 Common Issues

**"No movements detected"**
→ Lower threshold (30 → 20px) or check if person is visible

**"Module not found"**
→ Make sure virtual environment is activated: `source .venv/bin/activate`

**"Video codec error"**
→ Convert video: `ffmpeg -i input.mov -c:v libx264 output.mp4`

---

## 📖 Full Documentation

See `IMPLEMENTATION_GUIDE.md` for:
- Complete feature list
- Technical details
- Performance metrics
- Troubleshooting guide
- Future roadmap

---

## 🚀 What's Next?

### Current Status: Phase 0/1/2 Complete ✅

### Phase 1.5: Add AI Recognition
- Integrate Vision LLM (OpenRouter API)
- Automatic technique classification
- Error detection with coaching feedback

### Phase 2: Build Dashboard
- Web interface (React)
- Athlete progress tracking
- Session comparisons
- PDF reports

See full roadmap in `IMPLEMENTATION_GUIDE.md`

---

**Built for Serbian judo schools. Making professional sports science accessible. 🥋**
