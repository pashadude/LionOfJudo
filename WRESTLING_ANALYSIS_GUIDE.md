# Wrestling/Judo Analysis - Complete Guide

## 🎯 Overview

This guide covers the complete analysis pipeline for wrestling and judo technique videos, including:
- **Labeling** - Automatic technique identification
- **Analysis** - Biomechanical measurements
- **Recognition** - Movement pattern detection
- **Body Movements** - Detailed descriptions of all body parts
- **Comparison** - Training vs performance evaluation

---

## 📹 Video Collection

### Training Videos (Rubber Band Practice)
Located in `examples/`:
- `uki-goshi.mov` - Floating hip throw
- `o-soto-gari.mov` - Major outer reap
- `o-goshi.mov` - Major hip throw
- `Ippon-seoi-Nagi.mov` - One-arm shoulder throw

### Pupil Performance Videos
Located in `data/Lav/`:
- `o-soto-gari-x-uki-goshi.mov` - Two techniques in one video

---

## 🚀 Complete Analysis Pipeline

### Step 1: Process Videos (Already Running)

```bash
# Process all training videos
python3 phase0_visual_analysis.py examples/uki-goshi.mov
python3 phase0_visual_analysis.py examples/o-soto-gari.mov
python3 phase0_visual_analysis.py examples/o-goshi.mov
python3 phase0_visual_analysis.py examples/Ippon-seoi-Nagi.mov

# Process pupil performance
python3 phase0_visual_analysis.py data/Lav/o-soto-gari-x-uki-goshi.mov
```

**Outputs:**
- Annotated videos with skeleton overlays
- JSON files with pose data
- Basic movement detection reports

---

### Step 2: Generate Detailed Movement Reports

```bash
# Make executable
chmod +x movement_analysis.py

# Generate detailed reports for each video
python3 movement_analysis.py analysis/visual_results/uki-goshi_analysis.json
python3 movement_analysis.py analysis/visual_results/o-soto-gari_analysis.json
python3 movement_analysis.py analysis/visual_results/o-goshi_analysis.json
python3 movement_analysis.py analysis/visual_results/Ippon-seoi-Nagi_analysis.json
python3 movement_analysis.py analysis/visual_results/o-soto-gari-x-uki-goshi_analysis.json
```

**What You Get:**
Detailed analysis of:
- 🧠 **Head movement** - Position, stability, range of motion
- 💪 **Torso movement** - Angles, rotation, forward lean
- 🤚 **Arm movements** - Elbow angles, extension, positioning
- 🔵 **Hip movement** - Vertical drop, horizontal shift, rotation
- 🦵 **Leg movements** - Knee angles, stance width, flexion

---

### Step 3: Create Technique Comparisons

```bash
# Make executable
chmod +x compare_techniques.py

# Compare training vs performance
python3 compare_techniques.py \
  --training analysis/visual_results/o-soto-gari_analysis.json \
  --performance analysis/visual_results/o-soto-gari-x-uki-goshi_analysis.json \
  --create-video \
  --max-duration 15
```

**Outputs:**
- Side-by-side comparison video
- Biomechanical difference report
- Consistency scoring

---

### Step 4: Run Complete Pipeline (Automated)

```bash
# Make executable
chmod +x comprehensive_wrestling_analysis.py

# Run everything
python3 comprehensive_wrestling_analysis.py
```

**This automatically:**
1. Checks which videos are processed
2. Generates detailed movement reports for all
3. Creates technique comparisons where applicable
4. Generates master summary report

---

## 📊 Output Structure

```
analysis/
├── visual_results/                      # From Step 1
│   ├── uki-goshi_cam*.mp4              # Annotated videos
│   ├── uki-goshi_analysis.json         # Pose data
│   ├── uki-goshi_report.txt            # Basic report
│   └── ... (similar for all videos)
│
├── comparisons/                         # From Step 3
│   ├── comparison_*.mp4                # Side-by-side videos
│   └── comparison_*.txt                # Comparison reports
│
└── wrestling_results/                   # From Steps 2 & 4
    ├── detailed_movements/             # Detailed movement reports
    │   ├── uki-goshi_detailed_movement.txt
    │   ├── o-soto-gari_detailed_movement.txt
    │   └── ...
    └── MASTER_ANALYSIS_REPORT.txt      # Complete summary
```

---

## 📖 Understanding the Outputs

### Basic Movement Report (from phase0_visual_analysis.py)

```
============================================================
JUDO VISUAL ANALYSIS REPORT
============================================================

Video: o-soto-gari
Duration: 3.57s
Layout: 3h (3 cameras)
Poses detected: 214

============================================================
MOVEMENT DETECTION
============================================================

✓ Detected 1 movement(s)

1. Movement at 0.50s - 2.83s
   Duration: 2.33s
   Camera: 1 | Person: 0

   Biomechanics:
     • Hip drop: 127.5 pixels
     • Avg hip height: 445.3px
     • Avg torso angle: 18.7°
     • Avg knee angle: 152.3°
```

**What it means:**
- Hip drop = How much the person lowered their center of mass
- Hip height = Position in frame (lower number = higher stance)
- Torso angle = Lean forward from vertical
- Knee angle = 180° = straight, 90° = deep squat

---

### Detailed Movement Report (from movement_analysis.py)

```
================================================================================
DETAILED BODY MOVEMENT ANALYSIS
================================================================================

Video: o-soto-gari
Duration: 3.57s
Total poses detected: 214

================================================================================
BODY PART MOVEMENTS
================================================================================

🧠 HEAD MOVEMENT:
   • Head remains stable throughout movement
   • Horizontal range: 45.2px
   • Vertical range: 67.8px

💪 TORSO MOVEMENT:
   • Torso maintains mostly upright position with moderate rotation (12.5° range)
   • Average angle: 18.7°
   • Angle range: 12.3° to 24.8°

🤚 ARM MOVEMENTS:
   • Left arm moderately bent, Right arm moderately bent
   • Left arm angle range: 118.4° to 156.7°
   • Right arm angle range: 125.2° to 162.3°

🔵 HIP MOVEMENT:
   • Moderate hip drop (127.5px) with moderate lateral movement
   • Total hip drop: 127.5px
   • Horizontal shift: 78.3px

🦵 LEG MOVEMENTS:
   • Left leg moderately bent (avg 152.3°). Right leg moderately bent (avg 148.9°).
     Wide stance maintained (avg 245.7px)
   • Average stance width: 245.7px
```

**What it means:**
- Describes the full body movement pattern
- Identifies key characteristics of the technique
- Provides coaching insights

---

### Comparison Report (from compare_techniques.py)

```
======================================================================
JUDO TECHNIQUE COMPARISON REPORT
======================================================================

Video 1: o-soto-gari (Training)
Video 2: o-soto-gari-x-uki-goshi (Performance)

======================================================================
BIOMECHANICAL COMPARISON
======================================================================

📊 Hip Height:
   Training:    445.3px
   Performance: 512.7px
   Difference:  +67.4px (+15.1%)
   ⚠️  Performance has higher hip (lower in frame)

📐 Torso Angle:
   Training:    18.7°
   Performance: 22.3°
   Difference:  +3.6° (+19.3%)
   ⚠️  More forward lean in performance

🦵 Knee Angle:
   Training:    152.3°
   Performance: 145.8°
   Difference:  -6.5° (-4.3%)
   ⚠️  Deeper knee bend in performance

======================================================================
OVERALL ASSESSMENT
======================================================================

Technique Consistency: 82.3%
✓ Good - Minor differences between training and performance
```

**What it means:**
- Direct comparison of biomechanical features
- Identifies where technique differs
- Helps refine training approach

---

## 🎯 Analysis Workflow

### For a Single Technique

1. **Process video:**
   ```bash
   python3 phase0_visual_analysis.py examples/technique.mov
   ```

2. **View annotated video:**
   ```bash
   open analysis/visual_results/technique_cam1_annotated.mp4
   ```

3. **Generate detailed report:**
   ```bash
   python3 movement_analysis.py analysis/visual_results/technique_analysis.json
   ```

4. **Compare with performance:**
   ```bash
   python3 compare_techniques.py \
     --training analysis/visual_results/technique_training_analysis.json \
     --performance analysis/visual_results/technique_performance_analysis.json \
     --create-video
   ```

---

### For All Techniques (Batch)

1. **Process all videos** (5 videos in parallel - already running)

2. **Run comprehensive pipeline:**
   ```bash
   python3 comprehensive_wrestling_analysis.py
   ```

3. **Review master report:**
   ```bash
   cat analysis/wrestling_results/MASTER_ANALYSIS_REPORT.txt
   ```

---

## 🔍 What to Look For

### In Annotated Videos

**Color-coded skeleton:**
- 🔵 **Blue** = Head
- 🟡 **Yellow** = Arms
- 🟣 **Magenta** = Hips
- 🟢 **Green** = Legs

**On-screen measurements:**
- "Hip Y" = Vertical position (lower number = higher stance)
- "Torso" = Angle from vertical
- "L Knee" = Left knee angle

### In Movement Reports

**Key indicators of good technique:**
- ✅ Smooth, controlled hip drop
- ✅ Consistent torso angle
- ✅ Stable head position
- ✅ Appropriate knee flexion
- ✅ Wide, balanced stance

**Red flags:**
- ⚠️ Erratic head movement (loss of balance)
- ⚠️ Excessive torso angle (leaning too far)
- ⚠️ Straight legs during throw (no power generation)
- ⚠️ Narrow stance (unstable base)

---

## 💡 Coaching Insights

### O-Soto-Gari (Major Outer Reap)

**Expected biomechanics:**
- Upright torso (10-20°)
- Straight legs initially
- Wide stance
- Minimal hip drop until execution

**Common errors:**
- Too much forward lean
- Bent legs during setup
- Narrow stance

### Uki-Goshi (Floating Hip Throw)

**Expected biomechanics:**
- Moderate forward lean (20-30°)
- Hip positioned under opponent
- Significant hip drop
- Knee flexion during entry

**Common errors:**
- Hip too high
- Insufficient rotation
- Straight legs

### O-Goshi (Major Hip Throw)

**Expected biomechanics:**
- Strong forward lean (25-35°)
- Deep hip rotation
- Bent knees during entry
- Large hip drop

**Common errors:**
- Not enough hip rotation
- Legs too straight
- Hip not low enough

### Ippon-Seoi-Nagi (One-Arm Shoulder Throw)

**Expected biomechanics:**
- Significant forward lean (30-45°)
- Deep knee bend
- Large hip drop
- Strong torso rotation

**Common errors:**
- Insufficient drop under opponent
- Not enough rotation
- Poor arm positioning

---

## 📈 Progress Tracking

### Compare Sessions Over Time

1. Save each session's analysis
2. Create comparison videos between sessions
3. Track key metrics:
   - Hip drop amplitude (increasing = better)
   - Torso angle consistency (stable = better)
   - Knee flexion range (appropriate depth)
   - Movement smoothness (less variation = better)

### Example: Tracking O-Soto-Gari Improvement

```
Session 1: Hip drop 87px, Torso 18°, Consistency 75%
Session 2: Hip drop 105px, Torso 16°, Consistency 82%
Session 3: Hip drop 127px, Torso 15°, Consistency 89%

✅ Improvement in power generation (hip drop)
✅ More upright posture (better technique)
✅ More consistent execution
```

---

## 🚧 Current Status

**Processing (in background):**
- ✅ uki-goshi.mov
- ✅ o-soto-gari.mov
- ✅ o-goshi.mov
- ✅ Ippon-seoi-Nagi.mov
- ✅ o-soto-gari-x-uki-goshi.mov

**Once complete, you can:**
1. Run `python3 comprehensive_wrestling_analysis.py`
2. Review all detailed reports
3. View comparison videos
4. Use insights for coaching

---

## 🎓 Next Steps

### Immediate
- Wait for video processing to complete
- Review generated annotated videos
- Read detailed movement reports
- View side-by-side comparisons

### Advanced
- Track progress over multiple sessions
- Compare different athletes
- Identify common error patterns
- Develop targeted training drills

---

**🥋 Built to help coaches and athletes improve technique through data-driven insights!**
