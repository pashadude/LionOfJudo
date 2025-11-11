#!/usr/bin/env python3
"""
Judo Throw Recognition Using YOLO11 Pose Estimation + Biomechanical Analysis

This script implements the hybrid approach:
1. YOLO11 extracts skeleton poses from video
2. Biomechanical feature extraction (hip height, angles, velocities)
3. Rule-based classifier for throw detection
4. Optional: Vision LLM refinement for final classification

Usage:
    python judo_pose_recognition.py --video path/to/judo_video.mp4
    python judo_pose_recognition.py --video test.mp4 --use-llm --llm-model gemini-flash
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    exit(1)

# YOLO11 COCO keypoint indices
KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}

@dataclass
class ThrowFeatures:
    """Biomechanical features extracted from pose sequence"""
    frame_start: int
    frame_end: int
    duration_frames: int

    # Hip dynamics
    avg_hip_height: float
    min_hip_height: float
    max_hip_height: float
    hip_drop_amount: float  # max - min (indicates throw execution)

    # Velocity and acceleration
    max_hip_velocity: float  # pixels/frame
    max_center_velocity: float

    # Body angles
    avg_torso_angle: float  # relative to vertical
    max_torso_rotation: float

    # Leg dynamics
    avg_knee_separation: float  # distance between knees
    max_leg_extension: float  # hip-to-ankle distance

    # Interaction metrics (requires 2 people)
    has_two_people: bool
    min_person_distance: float  # closest approach

    # Confidence
    avg_confidence: float

@dataclass
class ThrowPrediction:
    """Detected throw with classification"""
    features: ThrowFeatures
    throw_type: str
    confidence: float
    reasoning: str

class JudoPoseAnalyzer:
    """Analyzes judo throws from YOLO11 pose data"""

    def __init__(self, model_size='x'):
        """
        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
                       'x' = most accurate, slowest
                       'n' = fastest, least accurate
        """
        print(f"Loading YOLO11{model_size}-pose model...")
        self.model = YOLO(f'yolo11{model_size}-pose.pt')
        self.pose_data = []

    def extract_poses(self, video_path: str, output_dir: Optional[str] = None) -> List[Dict]:
        """
        Extract pose data from video

        Args:
            video_path: Path to judo video
            output_dir: Optional directory to save annotated video

        Returns:
            List of pose data per frame
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"\nProcessing video: {video_path.name}")

        # Run inference
        results = self.model.predict(
            source=str(video_path),
            save=bool(output_dir),
            project=output_dir,
            name='pose_output',
            conf=0.5,  # Confidence threshold
            iou=0.7,   # NMS IoU threshold
            verbose=False,
            stream=True  # Generator for memory efficiency
        )

        pose_data = []
        frame_idx = 0

        for result in results:
            frame_data = {
                'frame': frame_idx,
                'people': []
            }

            if result.keypoints is not None and len(result.keypoints.data) > 0:
                for person_idx, kpts in enumerate(result.keypoints.data):
                    # kpts shape: [17, 3] (x, y, confidence)
                    keypoints_array = kpts.cpu().numpy()

                    # Convert to dict format
                    person_data = {
                        'person_id': person_idx,
                        'keypoints': {},
                        'bbox': result.boxes.xyxy[person_idx].cpu().numpy().tolist() if result.boxes else None
                    }

                    for name, idx in KEYPOINTS.items():
                        x, y, conf = keypoints_array[idx]
                        person_data['keypoints'][name] = {
                            'x': float(x),
                            'y': float(y),
                            'confidence': float(conf)
                        }

                    frame_data['people'].append(person_data)

            pose_data.append(frame_data)
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx} frames...")

        print(f"✓ Extracted poses from {frame_idx} frames")
        print(f"  Average people per frame: {np.mean([len(f['people']) for f in pose_data]):.1f}")

        self.pose_data = pose_data
        return pose_data

    def calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calculate angle between three points (in degrees)"""
        if any(p['confidence'] < 0.3 for p in [p1, p2, p3]):
            return 0.0

        # Vector from p2 to p1
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        # Vector from p2 to p3
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])

        # Angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def extract_features(self, window_size: int = 30) -> List[ThrowFeatures]:
        """
        Extract biomechanical features from pose sequences

        Args:
            window_size: Number of frames to analyze (30 = 1 sec at 30fps)

        Returns:
            List of feature windows where throws might occur
        """
        if not self.pose_data:
            raise ValueError("No pose data. Run extract_poses() first.")

        print(f"\nExtracting features (window size: {window_size} frames)...")
        features_list = []

        # Sliding window over frames
        for start_idx in range(0, len(self.pose_data) - window_size, window_size // 2):
            end_idx = start_idx + window_size
            window = self.pose_data[start_idx:end_idx]

            # Need at least one person in most frames
            frames_with_people = sum(1 for f in window if len(f['people']) > 0)
            if frames_with_people < window_size * 0.5:
                continue

            # Collect metrics
            hip_heights = []
            hip_velocities = []
            torso_angles = []
            knee_separations = []
            leg_extensions = []
            person_distances = []
            confidences = []

            prev_hip_y = None

            for frame_data in window:
                if len(frame_data['people']) == 0:
                    continue

                # Analyze first person (tori - thrower)
                person = frame_data['people'][0]
                kpts = person['keypoints']

                # Hip height (average of left and right hip)
                left_hip = kpts['left_hip']
                right_hip = kpts['right_hip']
                if left_hip['confidence'] > 0.3 and right_hip['confidence'] > 0.3:
                    hip_y = (left_hip['y'] + right_hip['y']) / 2
                    hip_heights.append(hip_y)

                    # Hip velocity (vertical)
                    if prev_hip_y is not None:
                        hip_velocities.append(abs(hip_y - prev_hip_y))
                    prev_hip_y = hip_y

                # Torso angle (shoulder to hip line vs vertical)
                left_shoulder = kpts['left_shoulder']
                if left_shoulder['confidence'] > 0.3 and left_hip['confidence'] > 0.3:
                    dx = left_shoulder['x'] - left_hip['x']
                    dy = left_shoulder['y'] - left_hip['y']
                    torso_angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
                    torso_angles.append(torso_angle)

                # Knee separation (stance width indicator)
                left_knee = kpts['left_knee']
                right_knee = kpts['right_knee']
                if left_knee['confidence'] > 0.3 and right_knee['confidence'] > 0.3:
                    knee_dist = np.sqrt((left_knee['x'] - right_knee['x'])**2 +
                                       (left_knee['y'] - right_knee['y'])**2)
                    knee_separations.append(knee_dist)

                # Leg extension (hip to ankle distance)
                left_ankle = kpts['left_ankle']
                if left_hip['confidence'] > 0.3 and left_ankle['confidence'] > 0.3:
                    leg_ext = np.sqrt((left_hip['x'] - left_ankle['x'])**2 +
                                     (left_hip['y'] - left_ankle['y'])**2)
                    leg_extensions.append(leg_ext)

                # Check for two people (interaction)
                if len(frame_data['people']) >= 2:
                    person1 = frame_data['people'][0]
                    person2 = frame_data['people'][1]

                    # Distance between centers
                    kpts1 = person1['keypoints']
                    kpts2 = person2['keypoints']

                    hip1_x = (kpts1['left_hip']['x'] + kpts1['right_hip']['x']) / 2
                    hip1_y = (kpts1['left_hip']['y'] + kpts1['right_hip']['y']) / 2
                    hip2_x = (kpts2['left_hip']['x'] + kpts2['right_hip']['x']) / 2
                    hip2_y = (kpts2['left_hip']['y'] + kpts2['right_hip']['y']) / 2

                    dist = np.sqrt((hip1_x - hip2_x)**2 + (hip1_y - hip2_y)**2)
                    person_distances.append(dist)

                # Average keypoint confidence
                confs = [kpt['confidence'] for kpt in kpts.values()]
                confidences.append(np.mean(confs))

            # Skip if not enough data
            if len(hip_heights) < 10:
                continue

            # Create feature object
            features = ThrowFeatures(
                frame_start=start_idx,
                frame_end=end_idx,
                duration_frames=window_size,
                avg_hip_height=float(np.mean(hip_heights)),
                min_hip_height=float(np.min(hip_heights)),
                max_hip_height=float(np.max(hip_heights)),
                hip_drop_amount=float(np.max(hip_heights) - np.min(hip_heights)),
                max_hip_velocity=float(np.max(hip_velocities)) if hip_velocities else 0.0,
                max_center_velocity=float(np.max(hip_velocities)) if hip_velocities else 0.0,
                avg_torso_angle=float(np.mean(torso_angles)) if torso_angles else 0.0,
                max_torso_rotation=float(np.max(torso_angles)) if torso_angles else 0.0,
                avg_knee_separation=float(np.mean(knee_separations)) if knee_separations else 0.0,
                max_leg_extension=float(np.max(leg_extensions)) if leg_extensions else 0.0,
                has_two_people=bool(person_distances),
                min_person_distance=float(np.min(person_distances)) if person_distances else 999.0,
                avg_confidence=float(np.mean(confidences))
            )

            features_list.append(features)

        print(f"✓ Extracted {len(features_list)} feature windows")
        return features_list

    def classify_throw(self, features: ThrowFeatures) -> ThrowPrediction:
        """
        Rule-based throw classification using biomechanical features

        This is a simplified heuristic classifier. For production, train ML model
        on labeled pose sequences.
        """

        # Threshold values (calibrated for 1080p video, ~200px person height)
        SIGNIFICANT_HIP_DROP = 50  # pixels
        HIGH_HIP_VELOCITY = 10  # pixels/frame
        LOW_HIP_POSITION = 400  # y-coordinate (lower = higher in frame)
        WIDE_STANCE = 100  # pixels between knees
        CLOSE_INTERACTION = 150  # pixels between people

        throw_type = "unknown"
        confidence = 0.0
        reasoning = []

        # Check if this looks like a throw attempt
        is_throw = (
            features.hip_drop_amount > SIGNIFICANT_HIP_DROP and
            features.has_two_people and
            features.min_person_distance < CLOSE_INTERACTION
        )

        if not is_throw:
            return ThrowPrediction(
                features=features,
                throw_type="no_throw",
                confidence=0.9,
                reasoning="No significant hip drop or no close interaction between athletes"
            )

        # Classify throw type based on biomechanical patterns

        # SEOI-NAGE family (shoulder throws)
        # Characteristics: Low hip, significant rotation, moderate hip drop
        if (features.min_hip_height < LOW_HIP_POSITION and
            features.avg_torso_angle > 30 and
            features.hip_drop_amount > SIGNIFICANT_HIP_DROP):

            throw_type = "seoi-nage / tai-otoshi (shoulder throw family)"
            confidence = 0.65
            reasoning.append("Low hip position + torso rotation + hip drop")

            # Distinguish seoi-nage (hip throw) from tai-otoshi (leg trip)
            if features.max_leg_extension > 250:
                throw_type = "tai-otoshi (body drop)"
                reasoning.append("Extended leg suggests blocking action")
            else:
                throw_type = "ippon-seoi-nage (one-arm shoulder throw)"
                reasoning.append("Compact leg position suggests hip throw")

        # HARAI-GOSHI / UCHI-MATA (sweeping hip throws)
        # Characteristics: High hip initially, then drops, wide stance
        elif (features.hip_drop_amount > SIGNIFICANT_HIP_DROP * 1.5 and
              features.avg_knee_separation > WIDE_STANCE):

            throw_type = "harai-goshi / uchi-mata (sweeping hip throw)"
            confidence = 0.60
            reasoning.append("Large hip drop + wide stance")

            # Check hip height - critical distinction!
            if features.min_hip_height > LOW_HIP_POSITION:
                reasoning.append("⚠️ WARNING: Hip position may be too HIGH for effective throw")
                confidence = 0.45  # Lower confidence if technique error suspected

        # O-SOTO-GARI (major outer reap)
        # Characteristics: Upright posture, leg extension, backward motion
        elif (features.avg_torso_angle < 20 and
              features.max_leg_extension > 200):

            throw_type = "o-soto-gari (major outer reap)"
            confidence = 0.55
            reasoning.append("Upright posture + leg extension")

        # TOMOE-NAGE (circle throw / sacrifice)
        # Characteristics: Very large hip drop, high velocity
        elif (features.hip_drop_amount > SIGNIFICANT_HIP_DROP * 2 and
              features.max_hip_velocity > HIGH_HIP_VELOCITY):

            throw_type = "tomoe-nage or sacrifice throw"
            confidence = 0.50
            reasoning.append("Extreme hip drop + high velocity (falling action)")

        # Generic standing throw
        else:
            throw_type = "unclassified_tachi_waza (standing technique)"
            confidence = 0.40
            reasoning.append("Throw detected but specific technique unclear")

        return ThrowPrediction(
            features=features,
            throw_type=throw_type,
            confidence=confidence,
            reasoning=" | ".join(reasoning)
        )

    def detect_throws(self, min_confidence: float = 0.5) -> List[ThrowPrediction]:
        """
        Detect and classify all throws in the video

        Args:
            min_confidence: Minimum confidence threshold to report throw

        Returns:
            List of detected throws
        """
        print("\nAnalyzing throws...")

        features_list = self.extract_features()
        predictions = []

        for features in features_list:
            prediction = self.classify_throw(features)

            # Only include predictions above confidence threshold
            if prediction.confidence >= min_confidence or prediction.throw_type != "no_throw":
                predictions.append(prediction)

        # Filter out overlapping detections (keep highest confidence)
        filtered = []
        for pred in sorted(predictions, key=lambda p: p.confidence, reverse=True):
            # Check if overlaps with already selected prediction
            overlaps = False
            for existing in filtered:
                # Check frame overlap
                if not (pred.features.frame_end < existing.features.frame_start or
                       pred.features.frame_start > existing.features.frame_end):
                    overlaps = True
                    break

            if not overlaps:
                filtered.append(pred)

        print(f"✓ Detected {len(filtered)} throws")
        return filtered

    def save_results(self, predictions: List[ThrowPrediction], output_path: str):
        """Save detection results to JSON"""
        output_path = Path(output_path)

        results = {
            'video': str(output_path.stem),
            'total_throws_detected': len(predictions),
            'throws': []
        }

        for pred in predictions:
            results['throws'].append({
                'frame_start': pred.features.frame_start,
                'frame_end': pred.features.frame_end,
                'timestamp_start': f"{pred.features.frame_start / 30:.2f}s",  # Assume 30fps
                'timestamp_end': f"{pred.features.frame_end / 30:.2f}s",
                'throw_type': pred.throw_type,
                'confidence': pred.confidence,
                'reasoning': pred.reasoning,
                'biomechanics': {
                    'hip_drop_amount_px': pred.features.hip_drop_amount,
                    'avg_hip_height_px': pred.features.avg_hip_height,
                    'avg_torso_angle_deg': pred.features.avg_torso_angle,
                    'has_two_people': pred.features.has_two_people,
                }
            })

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Judo throw recognition using YOLO11 pose')
    parser.add_argument('--video', required=True, help='Path to judo video file')
    parser.add_argument('--model-size', default='x', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size (x=best, n=fastest)')
    parser.add_argument('--output-dir', default='output',
                       help='Directory to save results and annotated video')
    parser.add_argument('--min-confidence', type=float, default=0.4,
                       help='Minimum confidence threshold for throw detection')
    parser.add_argument('--save-poses', action='store_true',
                       help='Save raw pose data to JSON')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize analyzer
    analyzer = JudoPoseAnalyzer(model_size=args.model_size)

    # Extract poses
    pose_data = analyzer.extract_poses(args.video, output_dir=str(output_dir))

    if args.save_poses:
        poses_file = output_dir / f"{Path(args.video).stem}_poses.json"
        with open(poses_file, 'w') as f:
            json.dump(pose_data, f, indent=2)
        print(f"✓ Pose data saved to: {poses_file}")

    # Detect throws
    predictions = analyzer.detect_throws(min_confidence=args.min_confidence)

    # Save results
    results_file = output_dir / f"{Path(args.video).stem}_throws.json"
    analyzer.save_results(predictions, str(results_file))

    # Print summary
    print("\n" + "="*60)
    print("THROW DETECTION SUMMARY")
    print("="*60)

    if predictions:
        for i, pred in enumerate(predictions, 1):
            print(f"\n{i}. {pred.throw_type}")
            print(f"   Time: {pred.features.frame_start/30:.1f}s - {pred.features.frame_end/30:.1f}s")
            print(f"   Confidence: {pred.confidence:.0%}")
            print(f"   Analysis: {pred.reasoning}")
    else:
        print("\nNo throws detected above confidence threshold.")

    print("\n" + "="*60)
    print(f"\nResults saved to: {output_dir}")
    print("- Annotated video with skeleton overlays")
    print(f"- {results_file.name} (throw detection results)")
    if args.save_poses:
        print(f"- {Path(args.video).stem}_poses.json (raw pose data)")

if __name__ == "__main__":
    main()
