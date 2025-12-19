#!/usr/bin/env python3
"""
LionOfJudo Phase 0/1/2 Implementation
Analyzes judo technique videos using YOLO11 pose estimation

Features:
- Multi-camera video splitting
- YOLO11 pose estimation
- Technique recognition
- Biomechanical analysis
- Report generation
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Dict, Tuple, Optional

# Try to import YOLO - will install if needed
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'ultralytics'])
    from ultralytics import YOLO


class VideoAnalyzer:
    """Analyzes judo videos for technique recognition"""

    def __init__(self, output_dir: Path = Path("analysis/results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load YOLO11 pose model
        print("Loading YOLO11 pose model...")
        self.model = YOLO('yolo11x-pose.pt')
        print("✓ Model loaded successfully")

        # Technique definitions
        self.techniques = {
            'o-soto-gari': 'Major Outer Reap',
            'o-goshi': 'Major Hip Throw',
            'ippon-seoi-nagi': 'One-Arm Shoulder Throw',
            'uki-goshi': 'Floating Hip Throw'
        }

    def detect_multi_camera_layout(self, frame: np.ndarray) -> Optional[str]:
        """
        Detect if frame is split-screen multi-camera
        Returns: 'single', '2h' (2 horizontal), '2v' (2 vertical), '3h', '3v', or None
        """
        h, w = frame.shape[:2]

        # Check for vertical split lines (dark regions indicating screen boundaries)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check for 2 vertical splits (3 screens side by side)
        split_1 = w // 3
        split_2 = 2 * w // 3

        # Sample vertical strips to detect dark boundaries
        strip_1 = gray[:, split_1-5:split_1+5].mean()
        strip_2 = gray[:, split_2-5:split_2+5].mean()
        overall_mean = gray.mean()

        if strip_1 < overall_mean * 0.3 and strip_2 < overall_mean * 0.3:
            return '3h'  # 3 horizontal splits

        # Check for 1 vertical split (2 screens side by side)
        split = w // 2
        strip = gray[:, split-5:split+5].mean()

        if strip < overall_mean * 0.3:
            return '2h'  # 2 horizontal splits

        # Check for horizontal split (2 screens top/bottom)
        split = h // 2
        strip = gray[split-5:split+5, :].mean()

        if strip < overall_mean * 0.3:
            return '2v'  # 2 vertical splits

        return 'single'

    def split_multi_camera_frame(self, frame: np.ndarray, layout: str) -> List[np.ndarray]:
        """Split multi-camera frame into individual camera views"""
        h, w = frame.shape[:2]

        if layout == 'single':
            return [frame]
        elif layout == '2h':
            # 2 screens side by side
            return [frame[:, :w//2], frame[:, w//2:]]
        elif layout == '2v':
            # 2 screens top/bottom
            return [frame[:h//2, :], frame[h//2:, :]]
        elif layout == '3h':
            # 3 screens side by side
            split1 = w // 3
            split2 = 2 * w // 3
            return [frame[:, :split1], frame[:, split1:split2], frame[:, split2:]]
        elif layout == '3v':
            # 3 screens top/bottom
            split1 = h // 3
            split2 = 2 * h // 3
            return [frame[:split1, :], frame[split1:split2, :], frame[split2:, :]]

        return [frame]

    def run_pose_estimation(self, video_path: Path, max_frames: Optional[int] = None) -> Dict:
        """Run YOLO11 pose estimation on video"""
        print(f"\n{'='*60}")
        print(f"Analyzing: {video_path.name}")
        print(f"{'='*60}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"Video info: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")

        # Detect layout from first frame
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")

        layout = self.detect_multi_camera_layout(first_frame)
        print(f"Detected layout: {layout}")

        # Reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Storage for results
        all_poses = []
        throw_events = []

        # Process video
        frame_idx = 0
        process_limit = max_frames if max_frames else total_frames

        print(f"Processing up to {process_limit} frames...")

        while frame_idx < process_limit:
            ret, frame = cap.read()
            if not ret:
                break

            # Split multi-camera frame if needed
            camera_frames = self.split_multi_camera_frame(frame, layout)

            # Process each camera view
            for cam_idx, cam_frame in enumerate(camera_frames):
                # Run YOLO pose estimation
                results = self.model.predict(
                    cam_frame,
                    conf=0.3,  # Lower confidence to catch more poses
                    iou=0.5,
                    verbose=False
                )

                # Extract pose data
                if results and len(results) > 0 and results[0].keypoints is not None:
                    keypoints = results[0].keypoints.data

                    for person_idx, kpts in enumerate(keypoints):
                        pose_data = {
                            'frame': frame_idx,
                            'camera': cam_idx,
                            'person': person_idx,
                            'timestamp': frame_idx / fps,
                            'keypoints': kpts.cpu().numpy().tolist()
                        }
                        all_poses.append(pose_data)

            frame_idx += 1

            # Progress indicator
            if frame_idx % 30 == 0:
                progress = (frame_idx / process_limit) * 100
                print(f"  Progress: {progress:.1f}% ({frame_idx}/{process_limit} frames)", end='\r')

        cap.release()
        print(f"\n✓ Processed {frame_idx} frames")
        print(f"✓ Detected {len(all_poses)} person-poses")

        # Analyze for throw events
        throw_events = self.detect_throw_events(all_poses, fps)

        return {
            'video_path': str(video_path),
            'video_name': video_path.stem,
            'layout': layout,
            'num_cameras': len(self.split_multi_camera_frame(first_frame, layout)),
            'fps': fps,
            'total_frames': frame_idx,
            'duration': frame_idx / fps,
            'poses': all_poses,
            'throw_events': throw_events,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def detect_throw_events(self, poses: List[Dict], fps: float) -> List[Dict]:
        """Detect throw events from pose data"""
        if not poses:
            return []

        throw_events = []

        # Group poses by camera and person
        from collections import defaultdict
        tracks = defaultdict(list)

        for pose in poses:
            key = (pose['camera'], pose['person'])
            tracks[key].append(pose)

        # Analyze each track for throws
        for (cam, person), track in tracks.items():
            if len(track) < 10:  # Need at least 10 frames
                continue

            events = self.analyze_track_for_throws(track, fps)
            throw_events.extend(events)

        # Sort by timestamp
        throw_events.sort(key=lambda x: x['start_time'])

        return throw_events

    def analyze_track_for_throws(self, track: List[Dict], fps: float) -> List[Dict]:
        """Analyze a single person track for throw events"""
        throws = []

        # Extract hip positions over time
        hip_heights = []
        for pose in track:
            kpts = np.array(pose['keypoints'])
            if len(kpts) >= 13:  # Need at least hip keypoints
                # Keypoints 11 and 12 are left and right hips
                left_hip_y = kpts[11][1]
                right_hip_y = kpts[12][1]
                avg_hip_y = (left_hip_y + right_hip_y) / 2
                hip_heights.append((pose['frame'], pose['timestamp'], avg_hip_y))

        if len(hip_heights) < 10:
            return throws

        # Look for significant hip drops (indicating throws)
        window_size = int(fps)  # 1 second window

        for i in range(len(hip_heights) - window_size):
            window = hip_heights[i:i+window_size]

            # Calculate hip drop in this window
            heights = [h[2] for h in window]
            max_height = max(heights)  # Higher Y = lower in frame
            min_height = min(heights)  # Lower Y = higher in frame
            hip_drop = max_height - min_height

            # Significant drop indicates throw (>50 pixels)
            if hip_drop > 50:
                # Check if we already recorded a throw nearby
                start_time = window[0][1]
                if throws and abs(throws[-1]['start_time'] - start_time) < 2.0:
                    continue

                # Calculate features for this throw
                features = self.extract_throw_features(track, i, i+window_size)

                throw = {
                    'start_frame': window[0][0],
                    'end_frame': window[-1][0],
                    'start_time': start_time,
                    'end_time': window[-1][1],
                    'duration': window[-1][1] - start_time,
                    'hip_drop_px': hip_drop,
                    'features': features,
                    'camera': track[0]['camera'],
                    'person': track[0]['person']
                }

                throws.append(throw)

        return throws

    def extract_throw_features(self, track: List[Dict], start_idx: int, end_idx: int) -> Dict:
        """Extract biomechanical features from throw segment"""
        segment = track[start_idx:end_idx]

        features = {
            'avg_hip_height': 0,
            'max_hip_drop': 0,
            'torso_angle_avg': 0,
            'has_rotation': False,
            'stance_width': 0
        }

        hip_ys = []
        torso_angles = []

        for pose in segment:
            kpts = np.array(pose['keypoints'])
            if len(kpts) < 17:
                continue

            # Hip height
            left_hip = kpts[11][:2]
            right_hip = kpts[12][:2]
            avg_hip_y = (left_hip[1] + right_hip[1]) / 2
            hip_ys.append(avg_hip_y)

            # Torso angle (shoulder to hip line)
            left_shoulder = kpts[5][:2]
            shoulder_to_hip_vec = left_hip - left_shoulder

            # Angle from vertical
            angle = np.degrees(np.arctan2(shoulder_to_hip_vec[0], shoulder_to_hip_vec[1]))
            torso_angles.append(abs(angle))

        if hip_ys:
            features['avg_hip_height'] = float(np.mean(hip_ys))
            features['max_hip_drop'] = float(max(hip_ys) - min(hip_ys))

        if torso_angles:
            features['torso_angle_avg'] = float(np.mean(torso_angles))
            features['has_rotation'] = np.mean(torso_angles) > 20  # >20° indicates rotation

        return features

    def classify_technique(self, throw_event: Dict, video_name: str) -> Dict:
        """Classify throw technique based on features and video name"""
        features = throw_event['features']

        # Extract technique from filename if present
        technique_from_name = None
        for tech_key, tech_name in self.techniques.items():
            if tech_key in video_name.lower():
                technique_from_name = tech_key
                break

        # Rule-based classification
        hip_drop = throw_event['hip_drop_px']
        torso_angle = features.get('torso_angle_avg', 0)
        has_rotation = features.get('has_rotation', False)
        avg_hip_height = features.get('avg_hip_height', 500)

        # Classification logic
        if torso_angle > 30 and has_rotation:
            # Shoulder throw family
            if avg_hip_height < 400:  # Low hip
                classified = 'ippon-seoi-nagi'
                confidence = 0.65
            else:
                classified = 'seoi-nage (shoulder throw family)'
                confidence = 0.55
        elif hip_drop > 70 and torso_angle < 20:
            # Upright with big drop = reaping technique
            classified = 'o-soto-gari'
            confidence = 0.60
        elif avg_hip_height > 400:
            # High hip = hip throw
            if torso_angle > 15:
                classified = 'o-goshi'
                confidence = 0.55
            else:
                classified = 'uki-goshi'
                confidence = 0.50
        else:
            classified = 'unknown'
            confidence = 0.30

        # Boost confidence if matches filename
        if technique_from_name and technique_from_name in classified:
            confidence = min(0.95, confidence + 0.25)

        return {
            'technique': classified,
            'technique_full_name': self.techniques.get(classified, 'Unknown Technique'),
            'confidence': confidence,
            'from_filename': technique_from_name,
            'reasoning': self._generate_reasoning(features, classified)
        }

    def _generate_reasoning(self, features: Dict, technique: str) -> str:
        """Generate human-readable reasoning for classification"""
        parts = []

        hip_drop = features.get('max_hip_drop', 0)
        torso_angle = features.get('torso_angle_avg', 0)
        hip_height = features.get('avg_hip_height', 0)

        if hip_drop > 70:
            parts.append(f"Large hip drop ({hip_drop:.0f}px)")
        elif hip_drop > 50:
            parts.append(f"Moderate hip drop ({hip_drop:.0f}px)")

        if torso_angle > 30:
            parts.append(f"Strong torso rotation ({torso_angle:.0f}°)")
        elif torso_angle > 15:
            parts.append(f"Moderate rotation ({torso_angle:.0f}°)")
        else:
            parts.append(f"Upright posture ({torso_angle:.0f}°)")

        if hip_height < 400:
            parts.append("Low hip position")
        elif hip_height > 500:
            parts.append("High hip position")

        return " | ".join(parts) if parts else "Insufficient features"

    def generate_report(self, analysis: Dict) -> str:
        """Generate human-readable analysis report"""
        report = []
        report.append("="*60)
        report.append("JUDO TECHNIQUE ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"\nVideo: {analysis['video_name']}")
        report.append(f"Duration: {analysis['duration']:.1f}s")
        report.append(f"Layout: {analysis['layout']} ({analysis['num_cameras']} camera(s))")
        report.append(f"Total poses detected: {len(analysis['poses'])}")
        report.append(f"\n{'='*60}")
        report.append("THROW DETECTION SUMMARY")
        report.append("="*60)

        throws = analysis['throw_events']

        if not throws:
            report.append("\n⚠️  No throws detected")
            report.append("\nPossible reasons:")
            report.append("  - Video too short")
            report.append("  - No significant body movement")
            report.append("  - Person not clearly visible")
            return "\n".join(report)

        report.append(f"\n✓ Detected {len(throws)} throw(s)\n")

        for i, throw in enumerate(throws, 1):
            # Classify technique
            classification = self.classify_technique(throw, analysis['video_name'])

            report.append(f"\n{i}. {classification['technique'].upper()}")
            if classification['technique_full_name'] != 'Unknown Technique':
                report.append(f"   ({classification['technique_full_name']})")
            report.append(f"   Time: {throw['start_time']:.1f}s - {throw['end_time']:.1f}s")
            report.append(f"   Duration: {throw['duration']:.1f}s")
            report.append(f"   Confidence: {classification['confidence']*100:.0f}%")
            report.append(f"   Camera: {throw['camera']} | Person: {throw['person']}")
            report.append(f"\n   Biomechanics:")
            report.append(f"     • Hip drop: {throw['hip_drop_px']:.0f} pixels")
            report.append(f"     • Avg hip height: {throw['features']['avg_hip_height']:.0f} px")
            report.append(f"     • Torso angle: {throw['features']['torso_angle_avg']:.1f}°")
            report.append(f"     • Rotation detected: {'Yes' if throw['features']['has_rotation'] else 'No'}")
            report.append(f"\n   Analysis: {classification['reasoning']}")

            # Check for potential errors
            warnings = self._check_technique_errors(throw['features'], classification['technique'])
            if warnings:
                report.append(f"\n   ⚠️  Potential issues:")
                for warning in warnings:
                    report.append(f"      - {warning}")

        report.append(f"\n{'='*60}")
        report.append(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*60)

        return "\n".join(report)

    def _check_technique_errors(self, features: Dict, technique: str) -> List[str]:
        """Check for common technique errors"""
        warnings = []

        hip_height = features.get('avg_hip_height', 500)
        torso_angle = features.get('torso_angle_avg', 0)

        if 'goshi' in technique:  # Hip throws
            if hip_height < 350:
                warnings.append("Hip position may be too HIGH (lower number = higher in frame)")
            if torso_angle < 10:
                warnings.append("Insufficient rotation for hip throw")

        if 'seoi' in technique:  # Shoulder throws
            if hip_height > 450:
                warnings.append("Hip position may be too high - should be lower for seoi-nage")
            if torso_angle < 25:
                warnings.append("More torso rotation needed for effective shoulder throw")

        if 'soto-gari' in technique:  # Reaping techniques
            if torso_angle > 25:
                warnings.append("Too much forward lean - should stay more upright for o-soto-gari")

        return warnings

    def save_results(self, analysis: Dict, report: str):
        """Save analysis results to files"""
        video_name = analysis['video_name']

        # Save JSON
        json_path = self.output_dir / f"{video_name}_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n✓ Saved JSON: {json_path}")

        # Save report
        report_path = self.output_dir / f"{video_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✓ Saved report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='LionOfJudo Phase 0/1/2 Video Analysis')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process (default: all)')
    parser.add_argument('--output-dir', type=str, default='analysis/results',
                       help='Output directory for results')

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    # Create analyzer
    analyzer = VideoAnalyzer(output_dir=Path(args.output_dir))

    # Run analysis
    analysis = analyzer.run_pose_estimation(video_path, max_frames=args.max_frames)

    # Generate report
    report = analyzer.generate_report(analysis)

    # Print report
    print("\n" + report)

    # Save results
    analyzer.save_results(analysis, report)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
