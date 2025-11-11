# LoRA Fine-Tuning Guide: Custom Judo Technique Recognition

**Goal:** Create a custom vision model that recognizes all 120 judo techniques with high accuracy, including detecting common errors.

**Why This Works:** The YouTube videos you found have technique names overlaid as text - we can use OCR to automatically create a labeled dataset in 10 minutes!

**Total Cost:** ~$30-50 one-time training cost
**Total Time:** 10 minutes dataset creation + 2-6 hours training

---

## What is LoRA?

**LoRA (Low-Rank Adaptation)** is a method to fine-tune large AI models efficiently:
- Takes existing vision model (GPT-4V, Gemini, etc.)
- Teaches it YOUR specific use case (judo techniques)
- Only trains small "adapter" layers (not entire model)
- Much cheaper and faster than training from scratch

**Result:** Model that knows all 120 techniques and can spot errors specific to your students.

---

## Phase 1: Create Labeled Dataset (10 minutes!)

### Step 1: Install Dependencies

```bash
# Install required tools
pip install yt-dlp opencv-python pytesseract

# Install Tesseract OCR engine
# Ubuntu/Debian:
sudo apt install tesseract-ocr

# macOS:
brew install tesseract

# Windows:
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Step 2: Run Automatic Dataset Creation

```bash
# Make script executable
chmod +x create_training_dataset.py

# Run it!
python3 create_training_dataset.py
```

**What This Does:**
1. Downloads 2 YouTube videos (80 standing + 40 ground techniques)
2. Uses OCR to read technique names from video text overlays
3. Extracts ~10 frames per technique
4. Creates labeled dataset: `training_data/judo_lora_training.jsonl`

**Output:**
```
training_data/
├── videos/
│   ├── tachi_waza.mp4
│   └── ne_waza.mp4
├── frames/
│   ├── tachi_waza/
│   │   ├── standing_seoi_nage_001.jpg
│   │   ├── standing_seoi_nage_002.jpg
│   │   └── ... (~800 images)
│   └── ne_waza/
│       └── ... (~400 images)
└── judo_lora_training.jsonl  ← This file goes to OpenAI
```

### Step 3: Manual Review (Optional but Recommended)

Check a few images to verify OCR accuracy:

```bash
# Look at extracted images
cd training_data/frames/tachi_waza
ls | head -20

# Check metadata
cat training_data/frames/tachi_waza/dataset_standing.json | head -50
```

**Common OCR Issues to Fix:**
- Misspelled technique names (fix manually)
- Same technique with different spellings (standardize)
- Misread text (delete those images)

**Quick Fix Script:**
```python
import json

# Load dataset
with open('training_data/frames/tachi_waza/dataset_standing.json') as f:
    data = json.load(f)

# Fix common OCR errors
replacements = {
    "seoi nage": "seoi-nage",
    "uchi mata": "uchi-mata",
    "harai goshi": "harai-goshi",
    # Add more as you find them
}

for item in data:
    for old, new in replacements.items():
        if old in item['technique']:
            item['technique'] = item['technique'].replace(old, new)

# Save cleaned version
with open('training_data/frames/tachi_waza/dataset_standing_cleaned.json', 'w') as f:
    json.dump(data, f, indent=2)
```

---

## Phase 2: Fine-Tune the Model

### Option A: Fine-Tune GPT-4V (Recommended for MVP)

**Why GPT-4V:**
- Easiest to fine-tune (fully managed by OpenAI)
- Good documentation and support
- Reasonable inference cost ($0.006/image)

**Cost:**
- Training: ~$30-50 for 1200 images
- Inference: $0.006/image (vs $0.00002 for base Gemini)

**Steps:**

1. **Prepare Training File**

Your `judo_lora_training.jsonl` is already in the right format!

2. **Upload to OpenAI**

```bash
# Install OpenAI CLI
pip install openai

# Set API key
export OPENAI_API_KEY="sk-..."

# Upload training file
openai files create \
  --file training_data/judo_lora_training.jsonl \
  --purpose fine-tune

# Note the file ID returned (e.g., "file-abc123")
```

3. **Start Fine-Tuning Job**

```bash
# Start training
openai fine-tuning jobs create \
  --training-file file-abc123 \
  --model gpt-4o-2024-08-06 \
  --suffix judo-coach-v1

# Monitor progress
openai fine-tuning jobs list

# Get specific job status
openai fine-tuning jobs retrieve ftjob-xyz789
```

**Training Time:** 2-6 hours depending on dataset size

4. **Test Your Model**

```python
from openai import OpenAI
import base64

client = OpenAI()

# Your fine-tuned model ID
MODEL = "ft:gpt-4o-2024-08-06:your-org:judo-coach-v1:abc123"

# Test image
with open("test_images/seoi_nage.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model=MODEL,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What judo technique is this?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]
    }]
)

print(response.choices[0].message.content)
```

---

### Option B: Fine-Tune Open Source Model (LLaVA) - Free but Complex

**Why LLaVA:**
- Completely free training and inference
- Host on your Hetzner server
- Full control over model

**Cost:**
- Training: $0 (run locally or on Hetzner GPU server)
- Inference: $0 (self-hosted)
- Time investment: HIGH (need ML expertise)

**Requirements:**
- GPU with 24GB+ VRAM (Hetzner AX102 or higher)
- Python ML stack (PyTorch, transformers, peft)
- More technical complexity

**When to Use:**
- Long-term deployment to many schools
- Budget is extremely tight
- Have ML engineering experience

**Guide:** See LORA_LLAVA.md for detailed instructions (separate doc)

---

## Phase 3: Enhance Dataset with Error Examples

The automatic dataset gives you perfect technique examples. Now add error examples:

### Step 1: Collect Error Examples from Real Training

After you start recording real sessions, identify common errors:

```json
{
  "image": "session_20250115/athlete_03_seoi_nage_bad.jpg",
  "technique": "seoi-nage",
  "category": "tachi-waza",
  "quality": "poor",
  "errors": [
    {
      "error_type": "entry_too_wide",
      "severity": "moderate",
      "description": "Left foot positioned too far from uke's feet. Should be centered between opponent's stance."
    },
    {
      "error_type": "incomplete_kuzushi",
      "severity": "critical",
      "description": "No visible off-balancing before entry. Uke's posture still stable."
    }
  ],
  "correct_aspects": [
    "grip is correct",
    "hip rotation direction good"
  ]
}
```

### Step 2: Combine Perfect + Error Examples

```python
import json

# Load original dataset
with open('training_data/judo_lora_training.jsonl') as f:
    perfect_examples = [json.loads(line) for line in f]

# Load error examples
with open('training_data/error_examples.jsonl') as f:
    error_examples = [json.loads(line) for line in f]

# Combine (aim for 70% perfect, 30% errors)
combined = perfect_examples + error_examples

# Save enhanced dataset
with open('training_data/judo_lora_enhanced.jsonl', 'w') as f:
    for item in combined:
        f.write(json.dumps(item) + '\n')
```

### Step 3: Re-train with Enhanced Dataset

```bash
# Upload enhanced dataset
openai files create \
  --file training_data/judo_lora_enhanced.jsonl \
  --purpose fine-tune

# Start new training job
openai fine-tuning jobs create \
  --training-file file-def456 \
  --model gpt-4o-2024-08-06 \
  --suffix judo-coach-v2
```

**Result:** Model that can now spot "harai-goshi with hip too high" and other errors!

---

## Phase 4: Deploy in Production

### Integration with Processing Pipeline

Update your OpenRouter script to use fine-tuned model:

```python
# Before (base model):
response = openrouter.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    messages=[...]
)

# After (fine-tuned model):
response = openai.chat.completions.create(
    model="ft:gpt-4o-2024-08-06:your-org:judo-coach-v2:abc123",
    messages=[...]
)
```

### Cost Comparison

**2-hour training session, 2000 key frames:**

| Model | Cost per Frame | Total Cost | Accuracy (Estimate) |
|-------|----------------|------------|---------------------|
| Gemini Flash (base) | $0.00002 | $0.04 | 70% |
| Claude Sonnet (base) | $0.004 | $8.00 | 80% |
| GPT-4V (base) | $0.006 | $12.00 | 75% |
| **GPT-4V Fine-tuned** | $0.006 | $12.00 | **90%+** |

**Verdict:** Fine-tuned model costs same as base GPT-4V but 15-20% more accurate on your specific techniques!

---

## Phase 5: Continuous Improvement

### Active Learning Loop

1. **Process sessions** with fine-tuned model
2. **Flag low-confidence predictions** (<70%)
3. **Manual review** by coach
4. **Add to training dataset** (with coach's labels)
5. **Re-train monthly** with new examples
6. **Model gets better over time**

### Tracking Improvement

```python
# Evaluate on validation set
validation_results = []

for test_image in validation_set:
    prediction = model.predict(test_image)
    ground_truth = test_image['label']

    validation_results.append({
        'correct': prediction['technique'] == ground_truth,
        'confidence': prediction['confidence']
    })

accuracy = sum(r['correct'] for r in validation_results) / len(validation_results)
print(f"Model accuracy: {accuracy*100:.1f}%")
```

---

## Cost-Benefit Analysis

### Scenario: 1 School, 3 Sessions/Week

**Base Model (Gemini Flash):**
- Cost: $0.04/session × 12 sessions/month = $0.48/month
- Accuracy: ~70%
- Manual review needed: ~30% of predictions

**Fine-Tuned Model (GPT-4V):**
- Training cost: $40 one-time
- Cost: $12/session × 12 sessions/month = $144/month
- Accuracy: ~90%
- Manual review needed: ~10% of predictions

**Break-Even Analysis:**
- Extra cost: $144 - $0.48 = $143.52/month
- Coach time saved: 20% less manual review = ~2 hours/month
- If coach time worth >$70/hour → fine-tuning pays for itself
- Plus: Better feedback for kids = more value

**Decision:**
- **For Serbian schools (tight budget):** Start with base Gemini, manual review
- **For well-funded clubs:** Fine-tune immediately, save coach time
- **Hybrid approach:** Use base model first 3 months, collect error examples, then fine-tune

---

## Troubleshooting

### Problem: OCR Not Reading Technique Names

**Solution:** Manually extract text regions

```python
# Adjust text region in script:
text_region_top = (50, 10, width-100, 80)  # Adjust coordinates

# Or use better OCR preprocessing:
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### Problem: Fine-Tuning Job Fails

**Common Causes:**
- JSONL format incorrect (check each line is valid JSON)
- Images too large (resize to max 2048px)
- Not enough examples per technique (need min 10)

**Debug:**
```bash
# Validate JSONL format
python3 -c "
import json
with open('judo_lora_training.jsonl') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except:
            print(f'Error on line {i}')
"
```

### Problem: Model Accuracy Lower Than Expected

**Solutions:**
1. Add more training examples (aim for 20+ per technique)
2. Balance dataset (equal examples of each technique)
3. Add negative examples (non-judo movements)
4. Increase training epochs (if API allows)

---

## Success Metrics

After fine-tuning, your model should achieve:

- [ ] Technique recognition accuracy >85% (vs 70% base)
- [ ] Error detection >75% (vs 50% base)
- [ ] Confidence scores well-calibrated (high confidence = correct)
- [ ] Distinguishes similar techniques (seoi-nage vs tai-otoshi)
- [ ] Spots common errors (hip height, timing, foot placement)

---

## Next Steps

1. **Run `create_training_dataset.py`** to extract labeled images (10 min)
2. **Review dataset** for OCR errors (30 min)
3. **Upload to OpenAI** and start fine-tuning ($40, 2-6 hours wait)
4. **Test on validation set** (not seen during training)
5. **Deploy in production** pipeline
6. **Collect error examples** from real sessions
7. **Re-train** in 1-3 months with enhanced dataset

---

## Resources

- OpenAI Fine-Tuning Docs: https://platform.openai.com/docs/guides/fine-tuning
- LLaVA (open source): https://github.com/haotian-liu/LLaVA
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- PEFT Library (for LoRA): https://github.com/huggingface/peft

---

**This is the secret weapon to make your system truly valuable for coaches. Base models know judo, but YOUR fine-tuned model knows YOUR students' mistakes.**
