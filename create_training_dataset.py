#!/usr/bin/env python3
"""
Automatically create labeled judo technique dataset from YouTube videos with text overlays
Using OCR to read technique names directly from video
"""

import subprocess
import cv2
import pytesseract
from pathlib import Path
import json
import re

# The two source videos with all 120 techniques labeled
VIDEOS = {
    "tachi_waza": {
        "url": "https://www.youtube.com/watch?v=LMKgaMdm9UY",
        "techniques": 80,
        "category": "standing"
    },
    "ne_waza": {
        "url": "https://www.youtube.com/watch?v=TgwfUOWB7TQ",
        "techniques": 40,
        "category": "ground"
    }
}

def download_video(url, output_path):
    """Download video using yt-dlp"""
    print(f"Downloading {url}...")
    cmd = [
        'yt-dlp',
        '-f', 'best[height<=1080]',
        '-o', str(output_path),
        url
    ]
    subprocess.run(cmd, check=True)
    print(f"  → Downloaded to {output_path}")

def extract_text_from_frame(frame, region=None):
    """
    Extract text from frame using OCR
    Args:
        frame: OpenCV frame
        region: (x, y, w, h) to crop - typically top or bottom of frame where text appears
    """
    if region:
        x, y, w, h = region
        frame = frame[y:y+h, x:x+w]

    # Preprocess for better OCR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
    # Threshold to make text clearer
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # OCR
    text = pytesseract.image_to_string(binary, config='--psm 6')
    return text.strip().lower()

def clean_technique_name(text):
    """
    Clean OCR output to get technique name
    Remove numbers, extra spaces, common OCR errors
    """
    # Remove common prefixes/suffixes
    text = re.sub(r'\d+[\.\)]\s*', '', text)  # Remove "1. " or "1) "
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = text.strip()

    # Common judo technique patterns
    if 'goshi' in text or 'nage' in text or 'gari' in text or 'otoshi' in text:
        return text

    return None

def process_video_with_ocr(video_path, output_dir, category):
    """
    Extract frames and automatically label them using OCR
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing {video_path.name}...")
    print(f"  FPS: {fps}, Total frames: {total_frames}")

    frame_num = 0
    current_technique = None
    technique_frame_count = 0
    dataset = []

    # Text region - adjust based on where technique names appear in video
    # Typically top 10% or bottom 10% of frame
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    text_region_top = (0, 0, width, int(height * 0.15))  # Top 15%
    text_region_bottom = (0, int(height * 0.85), width, int(height * 0.15))  # Bottom 15%

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check for text every 15 frames (0.5 seconds at 30fps)
        if frame_num % 15 == 0:
            # Try both top and bottom regions
            text_top = extract_text_from_frame(frame, text_region_top)
            text_bottom = extract_text_from_frame(frame, text_region_bottom)

            text = text_top if text_top else text_bottom

            technique_name = clean_technique_name(text)

            if technique_name and technique_name != current_technique:
                # New technique detected!
                print(f"  Frame {frame_num}: Detected '{technique_name}'")
                current_technique = technique_name
                technique_frame_count = 0

        # Save frames while current technique is being demonstrated
        if current_technique:
            # Save every 30 frames (1 per second) for current technique
            if frame_num % 30 == 0:
                frame_filename = f"{category}_{current_technique.replace(' ', '_')}_{technique_frame_count:03d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)

                dataset.append({
                    "image": frame_filename,
                    "technique": current_technique,
                    "category": category,
                    "frame_number": frame_num,
                    "timestamp": frame_num / fps
                })

                technique_frame_count += 1

                # Collect ~10 frames per technique, then move on
                if technique_frame_count >= 10:
                    current_technique = None

        frame_num += 1

        # Progress indicator
        if frame_num % 300 == 0:
            print(f"  Progress: {frame_num}/{total_frames} frames ({frame_num/total_frames*100:.1f}%)")

    cap.release()

    # Save dataset metadata
    metadata_file = output_dir / f"dataset_{category}.json"
    with open(metadata_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n  ✓ Extracted {len(dataset)} labeled images")
    print(f"  ✓ Metadata saved to {metadata_file}")

    return dataset

def create_lora_training_file(all_datasets, output_file):
    """
    Create JSONL file for LoRA fine-tuning
    Format expected by OpenAI/OpenRouter fine-tuning API
    """
    training_data = []

    for item in all_datasets:
        training_data.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert judo coach analyzing techniques."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What judo technique is being demonstrated in this image?"},
                        {"type": "image_url", "image_url": {"url": f"file://{item['image']}"}}
                    ]
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "technique": item["technique"],
                        "category": item["category"],
                        "confidence": 100
                    })
                }
            ]
        })

    # Save as JSONL (one JSON object per line)
    with open(output_file, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')

    print(f"\n✓ LoRA training file created: {output_file}")
    print(f"  Total training examples: {len(training_data)}")

def main():
    print("=" * 70)
    print("AUTOMATIC JUDO TECHNIQUE DATASET CREATION")
    print("=" * 70)
    print("\nThis script will:")
    print("1. Download 2 YouTube videos with all 120 judo techniques")
    print("2. Use OCR to read technique names from video overlays")
    print("3. Extract ~10 frames per technique")
    print("4. Create labeled dataset for LoRA fine-tuning")
    print(f"\nEstimated time: 10-15 minutes")
    print("=" * 70)

    input("\nPress Enter to start...")

    # Setup directories
    videos_dir = Path("training_data/videos")
    frames_dir = Path("training_data/frames")
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    all_datasets = []

    # Process each video
    for name, info in VIDEOS.items():
        # Download
        video_path = videos_dir / f"{name}.mp4"
        if not video_path.exists():
            download_video(info["url"], video_path)
        else:
            print(f"Video already downloaded: {video_path}")

        # Extract and label frames
        dataset = process_video_with_ocr(
            video_path,
            frames_dir / name,
            info["category"]
        )
        all_datasets.extend(dataset)

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Total techniques labeled: {len(set(item['technique'] for item in all_datasets))}")
    print(f"Total training images: {len(all_datasets)}")
    print(f"Images per technique (avg): {len(all_datasets) / len(set(item['technique'] for item in all_datasets)):.1f}")

    # Create LoRA training file
    create_lora_training_file(all_datasets, "training_data/judo_lora_training.jsonl")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Review extracted images in: training_data/frames/")
    print("2. Upload training file to OpenAI/OpenRouter for fine-tuning:")
    print("   training_data/judo_lora_training.jsonl")
    print("3. Cost: ~$30-50 for fine-tuning")
    print("4. Training time: 2-6 hours")
    print("5. Result: Custom model that knows all 120 judo techniques!")

    print("\n✓ Dataset creation complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
