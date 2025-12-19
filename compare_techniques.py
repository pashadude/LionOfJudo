#!/usr/bin/env python3
"""
LionOfJudo Technique Comparison Tool
Compare training videos with performance videos to show differences
"""

import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple


class TechniqueComparator:
    """Compare technique execution between videos"""

    def __init__(self, output_dir: Path = Path("analysis/comparisons")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_analysis(self, json_path: Path) -> Dict:
        """Load analysis JSON file"""
        with open(json_path) as f:
            return json.load(f)

    def compare_biomechanics(self, analysis1: Dict, analysis2: Dict) -> Dict:
        """Compare biomechanical features between two analyses"""

        # Extract features from movements
        def extract_features(analysis):
            if not analysis.get('movements'):
                return None

            # Use first significant movement
            movement = analysis['movements'][0]
            return movement.get('features', {})

        features1 = extract_features(analysis1)
        features2 = extract_features(analysis2)

        if not features1 or not features2:
            return {
                'comparison_possible': False,
                'reason': 'Insufficient movement data in one or both videos'
            }

        comparison = {
            'comparison_possible': True,
            'video1': analysis1['video_name'],
            'video2': analysis2['video_name'],
            'features': {}
        }

        # Compare each feature
        for key in set(features1.keys()) & set(features2.keys()):
            val1 = features1[key]
            val2 = features2[key]

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = val2 - val1
                percent_diff = (diff / val1 * 100) if val1 != 0 else 0

                comparison['features'][key] = {
                    'video1_value': val1,
                    'video2_value': val2,
                    'difference': diff,
                    'percent_difference': percent_diff
                }

        return comparison

    def create_side_by_side_video(self, video1_path: Path, video2_path: Path,
                                   output_path: Path, max_duration: int = 10):
        """Create side-by-side comparison video"""

        cap1 = cv2.VideoCapture(str(video1_path))
        cap2 = cv2.VideoCapture(str(video2_path))

        if not cap1.isOpened() or not cap2.isOpened():
            print(f"Error: Cannot open one or both videos")
            return False

        # Get video properties
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)

        # Use lower FPS for output
        output_fps = min(fps1, fps2)

        # Read first frames to get dimensions
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Error: Cannot read first frames")
            return False

        # Reset to beginning
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Resize frames to same height
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]

        target_height = min(h1, h2)
        new_w1 = int(w1 * (target_height / h1))
        new_w2 = int(w2 * (target_height / h2))

        # Create output video
        output_width = new_w1 + new_w2 + 20  # 20px gap
        output_height = target_height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps,
                                 (output_width, output_height))

        print(f"\nCreating side-by-side comparison...")
        print(f"Output: {output_path}")
        print(f"Max duration: {max_duration}s")

        frame_count = 0
        max_frames = int(output_fps * max_duration)

        while frame_count < max_frames:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            # Resize frames
            frame1_resized = cv2.resize(frame1, (new_w1, target_height))
            frame2_resized = cv2.resize(frame2, (new_w2, target_height))

            # Create combined frame with gap
            combined = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255
            combined[:, :new_w1] = frame1_resized
            combined[:, new_w1+20:] = frame2_resized

            # Add labels
            cv2.putText(combined, "Training (Rubber Bands)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "Performance (Live)", (new_w1 + 30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Add frame counter
            cv2.putText(combined, f"Frame {frame_count}", (10, output_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(combined)
            frame_count += 1

            if frame_count % 30 == 0:
                progress = (frame_count / max_frames) * 100
                print(f"  Progress: {progress:.1f}%", end='\r')

        cap1.release()
        cap2.release()
        writer.release()

        print(f"\n✓ Created {frame_count} frames")
        return True

    def generate_comparison_report(self, comparison: Dict, analysis1: Dict,
                                   analysis2: Dict) -> str:
        """Generate comparison report"""

        report = []
        report.append("="*70)
        report.append("JUDO TECHNIQUE COMPARISON REPORT")
        report.append("="*70)

        if not comparison['comparison_possible']:
            report.append(f"\n⚠️  {comparison['reason']}")
            return "\n".join(report)

        report.append(f"\nVideo 1: {comparison['video1']} (Training)")
        report.append(f"Video 2: {comparison['video2']} (Performance)")

        report.append(f"\n{'='*70}")
        report.append("BIOMECHANICAL COMPARISON")
        report.append("="*70)

        features = comparison['features']

        if 'avg_hip_height' in features:
            f = features['avg_hip_height']
            report.append(f"\n📊 Hip Height:")
            report.append(f"   Training:    {f['video1_value']:.1f}px")
            report.append(f"   Performance: {f['video2_value']:.1f}px")
            report.append(f"   Difference:  {f['difference']:+.1f}px ({f['percent_difference']:+.1f}%)")

            if abs(f['percent_difference']) < 5:
                report.append(f"   ✅ Very similar hip height")
            elif f['difference'] > 0:
                report.append(f"   ⚠️  Performance has higher hip (lower in frame)")
            else:
                report.append(f"   ⚠️  Performance has lower hip (higher in frame)")

        if 'avg_torso_angle' in features:
            f = features['avg_torso_angle']
            report.append(f"\n📐 Torso Angle:")
            report.append(f"   Training:    {f['video1_value']:.1f}°")
            report.append(f"   Performance: {f['video2_value']:.1f}°")
            report.append(f"   Difference:  {f['difference']:+.1f}° ({f['percent_difference']:+.1f}%)")

            if abs(f['difference']) < 3:
                report.append(f"   ✅ Consistent torso angle")
            elif f['video2_value'] > f['video1_value']:
                report.append(f"   ⚠️  More forward lean in performance")
            else:
                report.append(f"   ✅ More upright in performance")

        if 'avg_knee_angle' in features:
            f = features['avg_knee_angle']
            report.append(f"\n🦵 Knee Angle:")
            report.append(f"   Training:    {f['video1_value']:.1f}°")
            report.append(f"   Performance: {f['video2_value']:.1f}°")
            report.append(f"   Difference:  {f['difference']:+.1f}° ({f['percent_difference']:+.1f}%)")

            if abs(f['difference']) < 5:
                report.append(f"   ✅ Similar knee bend")
            elif f['video2_value'] < f['video1_value']:
                report.append(f"   ⚠️  Deeper knee bend in performance")
            else:
                report.append(f"   ⚠️  Straighter legs in performance")

        if 'hip_drop' in features:
            f = features['hip_drop']
            report.append(f"\n📉 Hip Drop (Movement Amplitude):")
            report.append(f"   Training:    {f['video1_value']:.1f}px")
            report.append(f"   Performance: {f['video2_value']:.1f}px")
            report.append(f"   Difference:  {f['difference']:+.1f}px ({f['percent_difference']:+.1f}%)")

            if f['video2_value'] > f['video1_value']:
                report.append(f"   ✅ Larger movement in performance (more dynamic)")
            else:
                report.append(f"   ⚠️  Smaller movement in performance")

        # Overall assessment
        report.append(f"\n{'='*70}")
        report.append("OVERALL ASSESSMENT")
        report.append("="*70)

        # Calculate consistency score
        consistency_scores = []
        for key, data in features.items():
            if 'percent_difference' in data:
                consistency = max(0, 100 - abs(data['percent_difference']))
                consistency_scores.append(consistency)

        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            report.append(f"\nTechnique Consistency: {avg_consistency:.1f}%")

            if avg_consistency > 90:
                report.append("✅ Excellent - Training translates well to performance")
            elif avg_consistency > 75:
                report.append("✓ Good - Minor differences between training and performance")
            elif avg_consistency > 60:
                report.append("⚠️  Moderate - Notable differences, may need adjustment")
            else:
                report.append("❌ Low - Significant differences, review technique")

        report.append(f"\n{'='*70}")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Compare judo technique videos')
    parser.add_argument('--training', type=str, required=True,
                       help='Path to training video analysis JSON')
    parser.add_argument('--performance', type=str, required=True,
                       help='Path to performance video analysis JSON')
    parser.add_argument('--create-video', action='store_true',
                       help='Create side-by-side comparison video')
    parser.add_argument('--max-duration', type=int, default=10,
                       help='Maximum duration for comparison video (seconds)')

    args = parser.parse_args()

    comparator = TechniqueComparator()

    # Load analyses
    training_json = Path(args.training)
    performance_json = Path(args.performance)

    if not training_json.exists() or not performance_json.exists():
        print("Error: One or both JSON files not found")
        return

    print(f"Loading analyses...")
    training_analysis = comparator.load_analysis(training_json)
    performance_analysis = comparator.load_analysis(performance_json)

    # Compare biomechanics
    comparison = comparator.compare_biomechanics(training_analysis, performance_analysis)

    # Generate report
    report = comparator.generate_comparison_report(comparison, training_analysis,
                                                   performance_analysis)

    print("\n" + report)

    # Save report
    report_path = comparator.output_dir / f"comparison_{training_analysis['video_name']}_vs_{performance_analysis['video_name']}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n✓ Saved comparison report: {report_path}")

    # Create side-by-side video if requested
    if args.create_video:
        # Find corresponding annotated videos
        training_video = Path(args.training).parent / f"{training_analysis['video_name']}_cam1_annotated.mp4"
        performance_video = Path(args.performance).parent / f"{performance_analysis['video_name']}_cam0_annotated.mp4"

        if not training_video.exists():
            training_video = Path(args.training).parent / f"{training_analysis['video_name']}_cam0_annotated.mp4"

        if training_video.exists() and performance_video.exists():
            output_video = comparator.output_dir / f"comparison_{training_analysis['video_name']}_vs_{performance_analysis['video_name']}.mp4"

            success = comparator.create_side_by_side_video(
                training_video,
                performance_video,
                output_video,
                max_duration=args.max_duration
            )

            if success:
                print(f"\n✓ Created comparison video: {output_video}")
        else:
            print("\n⚠️  Could not find annotated videos for side-by-side comparison")
            print(f"   Looking for: {training_video}")
            print(f"            and: {performance_video}")


if __name__ == "__main__":
    main()
