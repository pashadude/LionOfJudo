# Phase 0: Proof of Concept Testing Guide

**Goal:** Validate TWO separate approaches for judo technique recognition BEFORE buying any hardware.

**Budget:** $20-30 for API testing + GPU rental
**Timeline:** 1-2 weeks
**Success Criteria:** Choose best approach (vision LLM vs pose-based) with >70% accuracy and <$10/session cost

---

## Two Parallel Approaches to Test

### Approach A: Vision LLMs (OpenRouter)
- Send raw video frames to GPT-4V/Claude/Gemini
- Models "look at" the video like a human coach would
- No pose estimation needed
- **Pros:** Simple, no GPU needed, models understand context
- **Cons:** More expensive per frame, might miss subtle details

### Approach B: YOLO11 Pose + Analysis
- Run YOLO11 pose estimation to extract skeleton keypoints
- Analyze pose sequences to classify techniques
- **Pros:** Cheaper, more precise biomechanics, can measure angles
- **Cons:** Requires GPU, more complex pipeline, harder to detect context

### Hybrid Approach (Best of Both)
- YOLO11 pose for motion detection & frame filtering
- Vision LLM for technique recognition on filtered frames
- Combine: cheap motion filtering + smart recognition

---

## Week 1A: Test Vision LLM Recognition (OpenRouter)

### Task 1: Collect Test Videos (2-3 hours)

**What You Need:**
- 10-20 YouTube videos of judo training/competition
- Mix of experience levels (beginners making errors + advanced athletes)
- Different camera angles and lighting

**Suggested Sources:**
- IJF World Championships (high quality, perfect technique)
- Local dojo training videos (realistic, contains errors)
- Instructional videos showing correct vs incorrect form

**Create this directory structure:**
```
test_videos/
├── perfect_technique/
│   ├── 01_seoi_nage_perfect.mp4
│   ├── 02_uchi_mata_perfect.mp4
│   └── 03_harai_goshi_perfect.mp4
├── common_errors/
│   ├── 01_seoi_nage_no_kuzushi.mp4
│   ├── 02_harai_goshi_high_hip.mp4  ← Your example!
│   └── 03_tai_otoshi_wrong_foot.mp4
└── realistic_training/
    ├── session_01.mp4
    └── session_02.mp4
```

**Download Tool:**
```bash
pip install yt-dlp
yt-dlp -f 'best[height<=1080]' 'https://youtube.com/watch?v=VIDEO_ID'
```

---

### Task 2: Frame Extraction Script (1 hour)

Create `extract_frames.py`:

```python
#!/usr/bin/env python3
"""
Extract frames from judo videos for testing
"""
import subprocess
import os
from pathlib import Path

def extract_frames(video_path, output_dir, fps=1):
    """
    Extract frames from video
    Args:
        video_path: Path to input video
        output_dir: Where to save frames
        fps: Frames per second to extract (1 = 1 frame per second)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # FFmpeg command to extract frames
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', f'fps={fps}',
        '-q:v', '2',  # Quality (2 = high)
        str(output_dir / 'frame_%04d.jpg')
    ]

    print(f"Extracting from {video_path.name}...")
    subprocess.run(cmd, capture_output=True)

    # Count extracted frames
    frame_count = len(list(output_dir.glob('*.jpg')))
    print(f"  → Extracted {frame_count} frames")
    return frame_count

if __name__ == "__main__":
    video_dir = Path("test_videos")
    frames_dir = Path("extracted_frames")

    total_frames = 0
    for video in video_dir.rglob("*.mp4"):
        # Create output directory matching video structure
        relative_path = video.relative_to(video_dir)
        output_path = frames_dir / relative_path.parent / video.stem

        frame_count = extract_frames(video, output_path, fps=1)
        total_frames += frame_count

    print(f"\nTotal frames extracted: {total_frames}")
    print(f"Estimated cost at $0.00002/frame (Gemini Flash): ${total_frames * 0.00002:.2f}")
    print(f"Estimated cost at $0.004/frame (Claude Sonnet): ${total_frames * 0.004:.2f}")
```

Run it:
```bash
python3 extract_frames.py
```

---

### Task 3: OpenRouter API Testing (3-4 hours)

#### 3.1 Setup OpenRouter Account

1. Go to [OpenRouter.ai](https://openrouter.ai)
2. Sign up (free account)
3. Add $20 credit (Settings → Credits)
4. Get API key (Settings → API Keys)

Save to `.env` file:
```bash
OPENROUTER_API_KEY="sk-or-v1-..."
```

#### 3.2 Test Script for Technique Recognition

Create `test_recognition.py`:

```python
#!/usr/bin/env python3
"""
Test judo technique recognition with different vision models
"""
import os
import base64
import json
from pathlib import Path
from openai import OpenAI

# Configure OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

JUDO_PROMPT = """
You are an expert judo coach analyzing training footage.

Your task: Identify the judo technique being performed and evaluate its quality.

Reference - Common judo techniques:
TACHI-WAZA (Standing):
- Seoi-nage (shoulder throw)
- Ippon seoi-nage (one-arm shoulder throw)
- Uchi-mata (inner thigh throw)
- Harai-goshi (sweeping hip throw)
- Tai-otoshi (body drop)
- O-soto-gari (major outer reap)
- Ko-uchi-gari (minor inner reap)
- Tomoe-nage (circle throw)

NE-WAZA (Ground):
- Kesa-gatame (scarf hold)
- Juji-gatame (cross armlock)
- Kami-shiho-gatame (upper four-corner hold)

Analyze this image:

1. **Technique Identification:**
   - What technique is being attempted?
   - Phase: kuzushi (off-balance), tsukuri (positioning), kake (execution), or follow-through?
   - Confidence (0-100%)

2. **Technical Errors (BE SPECIFIC):**
   - Is the hip low enough? (critical for throws like harai-goshi)
   - Is the timing correct?
   - Foot placement errors?
   - Grip problems?
   - Balance issues?

3. **Quality Score (1-10):**
   - Overall execution quality
   - What's done well?
   - What needs improvement?

Respond in JSON format.
"""

def encode_image(image_path):
    """Convert image to base64 for API"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_frame(image_path, model="google/gemini-flash-1.5"):
    """
    Send single frame to vision model
    """
    image_base64 = encode_image(image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": JUDO_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }],
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)

    # Calculate cost
    usage = response.usage
    # Gemini Flash pricing: ~$0.075/1M input, $0.30/1M output
    cost = (usage.prompt_tokens * 0.075 / 1000000) + \
           (usage.completion_tokens * 0.30 / 1000000)

    return {
        "model": model,
        "result": result,
        "tokens": {"input": usage.prompt_tokens, "output": usage.completion_tokens},
        "cost": cost
    }

def test_perfect_technique():
    """Test on videos with perfect technique"""
    print("=" * 60)
    print("TEST 1: Perfect Technique Recognition")
    print("=" * 60)

    perfect_frames = list(Path("extracted_frames/perfect_technique").rglob("*.jpg"))[:5]

    total_cost = 0
    for frame in perfect_frames:
        print(f"\nTesting: {frame.parent.name}/{frame.name}")
        result = test_frame(frame)

        print(f"Model: {result['model']}")
        print(f"Technique: {result['result'].get('technique', 'Unknown')}")
        print(f"Confidence: {result['result'].get('confidence', 0)}%")
        print(f"Cost: ${result['cost']:.6f}")

        total_cost += result['cost']

    print(f"\nTotal cost for 5 frames: ${total_cost:.4f}")

def test_error_detection():
    """Test if model can identify technique errors"""
    print("\n" + "=" * 60)
    print("TEST 2: Error Detection")
    print("=" * 60)

    error_frames = list(Path("extracted_frames/common_errors").rglob("*.jpg"))[:5]

    for frame in error_frames:
        print(f"\nTesting: {frame.parent.name}/{frame.name}")
        result = test_frame(frame)

        print(f"Technique: {result['result'].get('technique', 'Unknown')}")

        errors = result['result'].get('technical_errors', [])
        if errors:
            print("Detected errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("  ⚠️  No errors detected (this might be a problem!)")

def compare_models():
    """Compare Gemini Flash vs Claude Sonnet vs GPT-4V"""
    print("\n" + "=" * 60)
    print("TEST 3: Model Comparison")
    print("=" * 60)

    test_frame_path = list(Path("extracted_frames").rglob("*.jpg"))[0]

    models = [
        "google/gemini-flash-1.5",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o"
    ]

    results = []
    for model in models:
        print(f"\nTesting {model}...")
        result = test_frame(test_frame_path, model=model)

        results.append({
            "model": model,
            "technique": result['result'].get('technique', 'Unknown'),
            "confidence": result['result'].get('confidence', 0),
            "cost": result['cost']
        })

    # Print comparison table
    print("\n" + "-" * 60)
    print(f"{'Model':<30} {'Technique':<20} {'Conf%':<8} {'Cost':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<30} {str(r['technique']):<20} {r['confidence']:<8} ${r['cost']:.6f}")

if __name__ == "__main__":
    # Run all tests
    test_perfect_technique()
    test_error_detection()
    compare_models()

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Review results above")
    print("2. If accuracy >70% → proceed to motion detection")
    print("3. If error detection poor → consider LoRA fine-tuning")
    print("4. If cost >$0.01/frame → optimize prompt or use cheaper model")
```

Run it:
```bash
python3 test_recognition.py
```

---

### Task 4: Evaluate Results (1 hour)

**Questions to Answer:**

1. **Technique Recognition Accuracy**
   - Does it correctly name the techniques?
   - Can it distinguish similar throws (seoi-nage vs tai-otoshi)?
   - Confidence scores realistic?

2. **Error Detection (CRITICAL!)**
   - Does it notice "hip too high" in harai-goshi?
   - Does it catch "no kuzushi" in failed attempts?
   - Are error descriptions specific enough for coaching?

3. **Cost Per Frame**
   - Gemini Flash cost: ~$0.00002/frame (TARGET)
   - Claude Sonnet cost: ~$0.004/frame (more accurate but 200× more expensive)
   - Extrapolate to 2-hour session (estimate ~2000 key frames)

4. **Decision Point:**
   ```
   If error detection score > 70% → PROCEED to Week 2
   If error detection score < 70% → NEED LoRA fine-tuning (see below)
   ```

---

## Week 1B: Test YOLO11 Pose Estimation (Parallel Track)

**Goal:** Test if pose-based approach can recognize techniques from skeleton data

### Where to Run YOLO11 Pose?

**Option 1: OVHcloud AI Endpoints (Managed)**
- Pre-deployed YOLO11 models available
- Available models:
  - `yolov11x-object-detection` - Detects people (bounding boxes)
  - `yolov11x-image-segmentation` - Pixel-level segmentation
  - **Missing:** YOLO11 pose model (17 keypoint skeleton)
- **Verdict:** OVHcloud doesn't have pose model yet - need Option 2 or 3

**Option 2: Rent GPU and Run YOLO11 Pose (Self-Hosted)**
- Hetzner GPU servers or RunPod/Vast.ai
- Install Ultralytics YOLO11
- Run pose estimation locally
- **Cost:** ~$0.50-1.00/hour GPU time
- **For testing:** 10 hours = $5-10

**Option 3: Google Colab Free GPU**
- Free Tesla T4 GPU for testing
- Install YOLO11, upload test videos
- Export pose data
- **Cost:** $0 (free tier)
- **Limitation:** Session timeout after ~12 hours

**Recommendation for Phase 0:** Use Google Colab (free) for initial testing

---

### Task 1: Setup YOLO11 Pose on Google Colab

Create a Colab notebook:

```python
# Install Ultralytics
!pip install ultralytics opencv-python

from ultralytics import YOLO
import cv2
import json
from pathlib import Path

# Load YOLO11 pose model
model = YOLO('yolo11x-pose.pt')  # Largest, most accurate

# Upload your test video to Colab
# Or download from YouTube
!yt-dlp -f 'best[height<=1080]' -o 'test_judo.mp4' 'YOUTUBE_URL'

# Run pose estimation
results = model.predict(
    source='test_judo.mp4',
    save=True,
    conf=0.5,  # Confidence threshold
    iou=0.7,   # NMS threshold
    show_labels=True,
    show_conf=True
)

# Extract pose data
pose_data = []
for frame_idx, result in enumerate(results):
    if result.keypoints is not None:
        for person_idx, kpts in enumerate(result.keypoints.data):
            # kpts shape: [17, 3] (x, y, confidence for each of 17 keypoints)
            pose_data.append({
                "frame": frame_idx,
                "person": person_idx,
                "keypoints": kpts.cpu().numpy().tolist()
            })

# Save pose data
with open('pose_data.json', 'w') as f:
    json.dump(pose_data, f)

print(f"Extracted pose data for {len(pose_data)} person-frames")
```

**Output:** Video with skeleton overlay + JSON file with all keypoints

---

### Task 2: Analyze Pose Data Quality

**Questions to Answer:**

1. **Detection Rate**
   - How many frames have 2+ people detected? (for technique interaction)
   - Any missed detections when athletes occlude each other?

2. **Keypoint Accuracy**
   - Are hips, knees, ankles detected correctly?
   - Critical for measuring "hip too low" in harai-goshi
   - Check keypoint confidence scores

3. **Temporal Consistency**
   - Do keypoints jump around frame-to-frame?
   - Or smooth trajectories?
   - Jitter = bad for measuring angles

**Evaluation Script:**

```python
import json
import numpy as np

with open('pose_data.json') as f:
    poses = json.load(f)

# Group by person across frames
from collections import defaultdict
person_tracks = defaultdict(list)

for p in poses:
    person_tracks[p['person']].append({
        'frame': p['frame'],
        'keypoints': p['keypoints']
    })

# Analyze each person's track
for person_id, track in person_tracks.items():
    print(f"\nPerson {person_id}:")
    print(f"  Frames detected: {len(track)}")

    # Calculate average confidence per keypoint
    kpt_confidences = np.array([t['keypoints'] for t in track])[:, :, 2]
    avg_conf = kpt_confidences.mean(axis=0)

    print(f"  Hip confidence: {avg_conf[11:13].mean():.2f}")  # Left/right hip
    print(f"  Knee confidence: {avg_conf[13:15].mean():.2f}")
    print(f"  Ankle confidence: {avg_conf[15:17].mean():.2f}")
```

---

### Task 3: Can Pose Data Recognize Techniques?

**Simple Heuristic Test:** Can we distinguish seoi-nage from uchi-mata using just pose?

```python
def detect_throw_type(pose_sequence):
    """
    Simple rule-based classifier using pose features
    pose_sequence: list of keypoints over 30 frames (1 second)
    """
    # Extract key features
    hip_heights = []
    leg_angles = []

    for pose in pose_sequence:
        left_hip = pose[11]  # (x, y, conf)
        right_hip = pose[12]
        left_knee = pose[13]
        right_knee = pose[14]

        # Hip height (y coordinate, lower = higher in image)
        hip_heights.append((left_hip[1] + right_hip[1]) / 2)

        # Leg angle (simplified)
        # ...calculate based on hip-knee-ankle

    # Heuristic rules:
    avg_hip_height = np.mean(hip_heights)
    hip_drop = max(hip_heights) - min(hip_heights)

    if hip_drop > 50:  # pixels - significant hip drop
        if avg_hip_height < 400:  # low hip position
            return "likely_seoi_nage_or_tai_otoshi"
        else:
            return "likely_harai_goshi_or_uchi_mata"
    else:
        return "no_throw_detected"

# Test on your extracted sequences
# ...
```

**Expected Result:** 40-60% accuracy with simple rules (proves pose has signal)

---

### Task 4: Cost Comparison

**YOLO11 Pose + Analysis:**
```
2-hour video (7200 frames at 30fps):
- GPU inference: ~10 min on RTX 4090
- Cost: $0.50/hour × 0.17 hours = $0.085
- Storage: Pose JSON file ~5 MB (vs 24 GB video)
- Then send filtered frames to Vision LLM
```

**Vision LLM Only:**
```
2-hour video:
- Send 2000 key frames to Gemini Flash
- Cost: 2000 × $0.00002 = $0.04
- Simpler pipeline, no GPU needed
```

**Hybrid (Best of Both):**
```
- YOLO pose for motion detection: $0.09
- Filter to 500 "high confidence technique" frames
- Send 500 frames to Gemini Flash: $0.01
- Total: $0.10 per session
- Get both: precise biomechanics + contextual understanding
```

---

## Week 2: Optimize for Cost & Accuracy

### Task 5: Motion Detection (Reduce Frame Count)

**Goal:** Only send frames with actual technique attempts to API

Create `motion_detect.py`:

```python
#!/usr/bin/env python3
"""
Detect action segments in judo videos to reduce API costs
"""
import cv2
import numpy as np
from pathlib import Path

def detect_action_frames(video_path, threshold=5000, min_duration=10):
    """
    Identify frames with significant motion
    Args:
        video_path: Path to video file
        threshold: Motion magnitude threshold (higher = less sensitive)
        min_duration: Minimum frames of continuous action to consider
    Returns:
        List of frame numbers with significant action
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    prev_frame = None
    action_frames = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is not None:
            # Calculate frame difference
            frame_diff = cv2.absdiff(prev_frame, gray)
            motion_score = np.sum(frame_diff)

            if motion_score > threshold:
                action_frames.append(frame_num)

        prev_frame = gray
        frame_num += 1

    cap.release()

    print(f"Total frames: {frame_num}")
    print(f"Action frames: {len(action_frames)} ({len(action_frames)/frame_num*100:.1f}%)")
    print(f"Estimated API cost reduction: {100 - (len(action_frames)/frame_num*100):.1f}%")

    return action_frames

if __name__ == "__main__":
    # Test on one video
    test_video = Path("test_videos/realistic_training/session_01.mp4")

    action_frames = detect_action_frames(test_video)

    # Save action frame numbers for selective extraction
    output_file = Path("action_frames.txt")
    with open(output_file, 'w') as f:
        f.write('\n'.join(map(str, action_frames)))

    print(f"\nAction frames saved to: {output_file}")
```

**Expected Result:**
- Should reduce frames by 60-80%
- 2-hour video: 7200 frames → 1500-2000 action frames
- Cost reduction: $144 → $30-40 per session

---

### Task 6: Multi-View Grid (For Phase 1)

When you have 3 cameras, stitch into single image = 1/3 the cost.

Create `stitch_views.py`:

```python
#!/usr/bin/env python3
"""
Combine 3 camera views into single grid image
"""
import cv2
import numpy as np
from pathlib import Path

def create_multiview_grid(cam1_frame, cam2_frame, cam3_frame):
    """
    Arrange 3 camera views in a grid:
    [ Cam1 ][ Cam2 ][ Cam3 ]
    """
    # Resize all to same height (reduce cost)
    height = 480  # Smaller = cheaper API calls
    frames_resized = []

    for frame in [cam1_frame, cam2_frame, cam3_frame]:
        h, w = frame.shape[:2]
        new_w = int(w * (height / h))
        resized = cv2.resize(frame, (new_w, height))
        frames_resized.append(resized)

    # Concatenate horizontally
    grid = np.hstack(frames_resized)

    return grid

if __name__ == "__main__":
    # Test with sample frames (you'll do this in Phase 1)
    print("Grid stitching ready for Phase 1 testing")
    print("This will reduce 3-camera API cost by 67%")
```

---

## Decision Point: Do We Need LoRA Fine-Tuning?

**If base model error detection is < 70% accurate, consider LoRA.**

### What is LoRA?
- Fine-tune existing vision model on YOUR specific judo videos
- Teach it Serbian judo school's common errors
- Only need 500-1000 labeled examples
- Much cheaper than training from scratch

### BREAKTHROUGH: Automatic Dataset Creation! ⚡

**You found the shortcut!** Those YouTube videos have technique names as text overlays. We can use OCR to automatically create a labeled dataset in 10 minutes!

**Run this script:**
```bash
python3 create_training_dataset.py
```

**What it does:**
1. Downloads 2 YouTube videos (120 techniques total)
2. Uses OCR to read technique names from screen text
3. Extracts ~10 frames per technique
4. Creates labeled dataset automatically
5. Result: 1200 labeled images in 10 minutes!

**No manual labeling needed!**

See **LORA_FINETUNING.md** for complete guide.

### LoRA Training Dataset Requirements

**Perfect Technique Examples:** (AUTOMATIC!)
- YouTube videos: https://www.youtube.com/watch?v=LMKgaMdm9UY (80 top)
- YouTube videos: https://www.youtube.com/watch?v=TgwfUOWB7TQ (40 bottom)
- OCR extracts technique names automatically
- ~1200 labeled images in 10 minutes

**Example Labels:**
```json
{
  "image": "harai_goshi_001.jpg",
  "technique": "harai-goshi",
  "quality": 4,
  "errors": [
    "hip_position_too_high",
    "sweep_leg_too_early",
    "no_kuzushi"
  ],
  "correct_aspects": [
    "grip_correct",
    "foot_placement_good"
  ]
}
```

### LoRA Training Options

**Option A: Fine-tune Gemini (if Google releases it)**
- Cheapest inference cost
- Currently not available for fine-tuning (as of 2025)

**Option B: Fine-tune GPT-4V (OpenAI)**
- $30-50 to train LoRA
- $0.006/image inference (more expensive than Gemini but knows your errors)

**Option C: Fine-tune Open Source (LLaVA/Idefics2)**
- Free training (run locally)
- Free inference (host on Hetzner)
- Requires more technical work

**Recommendation:**
- Start with Week 1 testing
- If error detection < 70% → spend 2 weeks collecting 500 labeled images
- Use Option B (GPT-4V fine-tune) for simplicity

---

## Phase 0 Final Decision Matrix

After completing Week 1A (Vision LLM) and Week 1B (YOLO Pose), choose your approach:

### Comparison Table

| Criteria | Vision LLM Only | YOLO Pose + Rules | Hybrid (YOLO + LLM) |
|----------|----------------|-------------------|---------------------|
| **Technique Recognition Accuracy** | 70-80% | 40-60% | 75-85% |
| **Error Detection (subtle)** | Medium (depends on model) | High (measures angles) | Best of both |
| **Cost per 2-hr session** | $0.04-0.50 | $0.09 (GPU) | $0.10-0.15 |
| **Infrastructure needed** | None (API only) | GPU server | GPU + API |
| **Implementation complexity** | Low | Medium | High |
| **Biomechanical measurements** | Limited | Excellent | Excellent |
| **Context understanding** | Excellent | Poor | Excellent |

### Recommended Approaches by Budget

**Tight Budget (Serbian schools - <$5/session):**
- ✅ **Vision LLM Only** (Gemini Flash)
- Cost: ~$0.04/session
- Trade-off: Less precise biomechanics, but good enough for technique ID
- Simple to deploy

**Medium Budget ($10-20/session):**
- ✅ **Hybrid Approach**
- YOLO pose for motion detection + angle measurement
- Vision LLM for technique recognition
- Best overall quality

**Focus on Biomechanics (research/elite training):**
- ✅ **YOLO Pose + Custom Classifier**
- Precise joint angle measurements
- Train classifier on pose sequences
- Can export data for further analysis

### Phase 0 Success Criteria Checklist

**For Vision LLM Approach:**
- [ ] Base model recognizes >70% of technique names correctly
- [ ] Model detects obvious errors (hip height, timing, foot placement) >60% of time
- [ ] Motion detection reduces frames by >60%
- [ ] Estimated cost per 2-hour session < $5 (Gemini) or < $20 (Claude/GPT-4V)
- [ ] Processing time estimate < 30 minutes for frame extraction + API calls

**For YOLO Pose Approach:**
- [ ] YOLO detects 2+ people in >80% of technique attempt frames
- [ ] Hip/knee/ankle keypoints have >70% confidence
- [ ] Simple heuristic rules achieve >50% technique classification
- [ ] GPU inference cost < $0.10/session
- [ ] Pose data exports correctly for analysis

**For Hybrid Approach:**
- [ ] All criteria from both above approaches met
- [ ] Pipeline successfully combines YOLO filtering → LLM recognition
- [ ] Total cost < $10/session

**If all checked → BUY PHASE 1 HARDWARE ($431)**

**If error detection fails → Collect 500 labeled images for LoRA fine-tuning first**

---

## Cost Tracking Template

Keep a spreadsheet:

| Test | Frames Sent | Model | Total Cost | Accuracy | Notes |
|------|-------------|-------|------------|----------|-------|
| Perfect technique | 5 | Gemini Flash | $0.0001 | 80% | Good recognition |
| Error detection | 5 | Gemini Flash | $0.0001 | 50% | Misses subtle errors |
| Same test | 5 | Claude Sonnet | $0.02 | 85% | Better but 200× costlier |

**Target:** Find model with best cost/accuracy tradeoff for your budget.

---

## Next Document to Create

After Phase 0 succeeds:
1. **LORA_FINETUNING.md** (if needed for error detection)
2. **HETZNER_SETUP.md** (server configuration for processing)
3. **Build Phase 1 hardware** (see HARDWARE_SETUP.md)

---

**You're doing this to help kids in Serbian schools. Every dollar saved on testing is more equipment for kids. Test thoroughly before buying hardware!**
