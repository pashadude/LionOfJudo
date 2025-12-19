#!/usr/bin/env python3
"""
Batch process all judo videos in examples and data/Lav folders
"""

import subprocess
from pathlib import Path
import sys

def process_video(video_path: Path, output_dir: Path):
    """Process a single video"""
    print(f"\n{'='*70}")
    print(f"Processing: {video_path.name}")
    print(f"{'='*70}")

    cmd = [
        sys.executable,
        'phase0_judo_analysis.py',
        str(video_path),
        '--output-dir', str(output_dir)
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✓ Successfully processed: {video_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error processing {video_path.name}: {e}")
        return False

def main():
    # Find all videos
    examples_dir = Path('examples')
    data_dir = Path('data/Lav')

    example_videos = list(examples_dir.glob('*.mov')) + list(examples_dir.glob('*.mp4'))
    pupil_videos = list(data_dir.glob('*.mov')) + list(data_dir.glob('*.mp4'))

    all_videos = example_videos + pupil_videos

    print(f"Found {len(example_videos)} example videos")
    print(f"Found {len(pupil_videos)} pupil videos")
    print(f"Total: {len(all_videos)} videos to process")

    if not all_videos:
        print("No videos found!")
        return

    # Process each video
    results = {
        'success': [],
        'failed': []
    }

    for video in all_videos:
        # Determine output directory
        if video.parent.name == 'examples':
            output_dir = Path('analysis/examples')
        else:
            output_dir = Path('analysis/pupil_videos')

        output_dir.mkdir(parents=True, exist_ok=True)

        # Process
        success = process_video(video, output_dir)

        if success:
            results['success'].append(video.name)
        else:
            results['failed'].append(video.name)

    # Summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Successfully processed: {len(results['success'])}")
    for name in results['success']:
        print(f"   - {name}")

    if results['failed']:
        print(f"\n✗ Failed: {len(results['failed'])}")
        for name in results['failed']:
            print(f"   - {name}")

    # Generate combined report
    generate_combined_report(results, Path('analysis'))


def generate_combined_report(results: dict, base_dir: Path):
    """Generate a combined summary report"""
    report_path = base_dir / 'combined_report.txt'

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("LIONOFJUDO - COMBINED ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Total videos processed: {len(results['success']) + len(results['failed'])}\n")
        f.write(f"Successful: {len(results['success'])}\n")
        f.write(f"Failed: {len(results['failed'])}\n\n")

        # Collect all individual reports
        f.write("="*70 + "\n")
        f.write("INDIVIDUAL REPORTS\n")
        f.write("="*70 + "\n\n")

        for report_file in base_dir.rglob('*_report.txt'):
            f.write(f"\n{'='*70}\n")
            f.write(f"File: {report_file.name}\n")
            f.write(f"{'='*70}\n\n")

            with open(report_file) as rf:
                f.write(rf.read())
            f.write("\n")

    print(f"\n✓ Combined report saved: {report_path}")


if __name__ == "__main__":
    main()
