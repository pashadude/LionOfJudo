#!/usr/bin/env python3
"""
LionOfJudo Phase 0/1/2 with Visual Output
Generates annotated videos with skeleton keypoints and biomechanical measurements
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Dict, Tuple, Optional

from ultralytics import YOLO


class VisualJudoAnalyzer:
    """Analyzes judo videos with visual skeleton overlay"""

    # COCO keypoint indices
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # Skeleton connections (bones to draw)
    SKELETON = [
        (5, 6),   # shoulders
        (5, 7),   # left arm
        (7, 9),   # left forearm
        (6, 8),   # right arm
        (8, 10),  # right forearm
        (5, 11),  # left torso
        (6, 12),  # right torso
        (11, 12), # hips
        (11, 13), # left thigh
        (13, 15), # left shin
        (12, 14), # right thigh
        (14, 16), # right shin
    ]

    def __init__(self, output_dir: Path = Path("analysis/visual_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading YOLO11 pose model...")
        self.model = YOLO('yolo11x-pose.pt')
        print("✓ Model loaded successfully")

    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, color=(0, 255, 0)):
        """Draw skeleton on frame"""
        # keypoints shape: [17, 3] (x, y, confidence)
        kpts = keypoints[:17, :]  # Ensure we have 17 keypoints

        # Draw bones (connections)
        for start_idx, end_idx in self.SKELETON:
            start_pt = kpts[start_idx]
            end_pt = kpts[end_idx]

            # Only draw if both points are confident
            if start_pt[2] > 0.3 and end_pt[2] > 0.3:
                pt1 = (int(start_pt[0]), int(start_pt[1]))
                pt2 = (int(end_pt[0]), int(end_pt[1]))
                cv2.line(frame, pt1, pt2, color, 2)

        # Draw keypoints (joints)
        for i, kpt in enumerate(kpts):
            if kpt[2] > 0.3:  # confidence threshold
                x, y = int(kpt[0]), int(kpt[1])

                # Color code different body parts
                if i < 5:  # Head
                    point_color = (255, 0, 0)  # Blue
                elif i < 11:  # Arms
                    point_color = (0, 255, 255)  # Yellow
                elif i < 13:  # Hips
                    point_color = (255, 0, 255)  # Magenta
                else:  # Legs
                    point_color = (0, 255, 0)  # Green

                cv2.circle(frame, (x, y), 4, point_color, -1)
                cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)

        return frame

    def add_measurements_overlay(self, frame: np.ndarray, keypoints: np.ndarray, camera_idx: int):
        """Add biomechanical measurements as text overlay"""
        if len(keypoints) < 17:
            return frame

        kpts = keypoints[:17, :]

        # Calculate hip height
        left_hip = kpts[11]
        right_hip = kpts[12]
        if left_hip[2] > 0.3 and right_hip[2] > 0.3:
            hip_y = int((left_hip[1] + right_hip[1]) / 2)
            cv2.putText(frame, f"Hip Y: {hip_y}px", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Calculate torso angle
        left_shoulder = kpts[5]
        if left_shoulder[2] > 0.3 and left_hip[2] > 0.3:
            vec = left_hip[:2] - left_shoulder[:2]
            angle = np.degrees(np.arctan2(vec[0], vec[1]))
            cv2.putText(frame, f"Torso: {abs(angle):.1f}deg", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Calculate knee angle (left leg)
        left_knee = kpts[13]
        left_ankle = kpts[15]
        if left_hip[2] > 0.3 and left_knee[2] > 0.3 and left_ankle[2] > 0.3:
            vec1 = left_hip[:2] - left_knee[:2]
            vec2 = left_ankle[:2] - left_knee[:2]

            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            knee_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            cv2.putText(frame, f"L Knee: {knee_angle:.1f}deg", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Camera label
        cv2.putText(frame, f"Cam {camera_idx}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def detect_multi_camera_layout(self, frame: np.ndarray) -> str:
        """Detect if frame is split-screen multi-camera"""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check for 3 horizontal splits
        split_1 = w // 3
        split_2 = 2 * w // 3
        strip_1 = gray[:, split_1-5:split_1+5].mean()
        strip_2 = gray[:, split_2-5:split_2+5].mean()
        overall_mean = gray.mean()

        if strip_1 < overall_mean * 0.3 and strip_2 < overall_mean * 0.3:
            return '3h'

        # Check for 2 horizontal splits
        split = w // 2
        strip = gray[:, split-5:split+5].mean()
        if strip < overall_mean * 0.3:
            return '2h'

        return 'single'

    def split_multi_camera_frame(self, frame: np.ndarray, layout: str) -> List[np.ndarray]:
        """Split multi-camera frame"""
        h, w = frame.shape[:2]

        if layout == 'single':
            return [frame]
        elif layout == '2h':
            return [frame[:, :w//2], frame[:, w//2:]]
        elif layout == '3h':
            split1 = w // 3
            split2 = 2 * w // 3
            return [frame[:, :split1], frame[:, split1:split2], frame[:, split2:]]

        return [frame]

    def process_video(self, video_path: Path) -> Dict:
        """Process video with visual output"""
        print(f"\n{'='*60}")
        print(f"Analyzing: {video_path.name}")
        print(f"{'='*60}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")

        # Detect layout
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")

        layout = self.detect_multi_camera_layout(first_frame)
        num_cameras = len(self.split_multi_camera_frame(first_frame, layout))
        print(f"Layout: {layout} ({num_cameras} camera(s))")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Prepare output video writers
        camera_frames = self.split_multi_camera_frame(first_frame, layout)
        writers = []

        for cam_idx in range(num_cameras):
            cam_frame = camera_frames[cam_idx]
            h, w = cam_frame.shape[:2]

            output_path = self.output_dir / f"{video_path.stem}_cam{cam_idx}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            writers.append(writer)

        # Process frames
        all_poses = []
        frame_idx = 0

        print("Processing and annotating frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Split cameras
            camera_frames = self.split_multi_camera_frame(frame, layout)

            # Process each camera
            for cam_idx, cam_frame in enumerate(camera_frames):
                # Make a copy for annotation
                annotated = cam_frame.copy()

                # Run YOLO
                results = self.model.predict(
                    cam_frame,
                    conf=0.3,
                    iou=0.5,
                    verbose=False
                )

                # Draw skeletons and measurements
                if results and len(results) > 0 and results[0].keypoints is not None:
                    keypoints = results[0].keypoints.data

                    for person_idx, kpts in enumerate(keypoints):
                        kpts_np = kpts.cpu().numpy()

                        # Draw skeleton
                        annotated = self.draw_skeleton(annotated, kpts_np)

                        # Add measurements
                        annotated = self.add_measurements_overlay(annotated, kpts_np, cam_idx)

                        # Store pose data
                        all_poses.append({
                            'frame': frame_idx,
                            'camera': cam_idx,
                            'person': person_idx,
                            'timestamp': frame_idx / fps,
                            'keypoints': kpts_np.tolist()
                        })

                # Add frame number
                cv2.putText(annotated, f"Frame {frame_idx}", (annotated.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Write frame
                writers[cam_idx].write(annotated)

            frame_idx += 1

            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"  Progress: {progress:.1f}%", end='\r')

        # Cleanup
        cap.release()
        for writer in writers:
            writer.release()

        print(f"\n✓ Processed {frame_idx} frames")
        print(f"✓ Detected {len(all_poses)} person-poses")

        # Analyze for movements
        movements = self.analyze_movements(all_poses, fps)

        # Print output paths
        print(f"\n✓ Annotated videos:")
        for cam_idx in range(num_cameras):
            output_path = self.output_dir / f"{video_path.stem}_cam{cam_idx}_annotated.mp4"
            print(f"   Camera {cam_idx}: {output_path}")

        return {
            'video_path': str(video_path),
            'video_name': video_path.stem,
            'layout': layout,
            'num_cameras': num_cameras,
            'fps': fps,
            'total_frames': frame_idx,
            'duration': frame_idx / fps,
            'poses': all_poses,
            'movements': movements
        }

    def analyze_movements(self, poses: List[Dict], fps: float) -> List[Dict]:
        """Analyze movement patterns (improved for short videos)"""
        if not poses:
            return []

        movements = []

        # Group by camera and person
        from collections import defaultdict
        tracks = defaultdict(list)

        for pose in poses:
            key = (pose['camera'], pose['person'])
            tracks[key].append(pose)

        # Analyze each track
        for (cam, person), track in tracks.items():
            if len(track) < 5:  # Lower threshold for short videos
                continue

            # Extract hip movements
            hip_positions = []
            for pose in track:
                kpts = np.array(pose['keypoints'])
                if len(kpts) >= 13:
                    left_hip = kpts[11]
                    right_hip = kpts[12]
                    if left_hip[2] > 0.3 and right_hip[2] > 0.3:
                        avg_hip_y = (left_hip[1] + right_hip[1]) / 2
                        hip_positions.append({
                            'frame': pose['frame'],
                            'timestamp': pose['timestamp'],
                            'y': avg_hip_y
                        })

            if len(hip_positions) >= 5:
                # Look for any significant movement (lowered threshold)
                ys = [h['y'] for h in hip_positions]
                hip_drop = max(ys) - min(ys)

                if hip_drop > 30:  # Lowered from 50 pixels
                    features = self.extract_features(track)

                    movement = {
                        'type': 'technique_attempt',
                        'start_frame': hip_positions[0]['frame'],
                        'end_frame': hip_positions[-1]['frame'],
                        'start_time': hip_positions[0]['timestamp'],
                        'end_time': hip_positions[-1]['timestamp'],
                        'duration': hip_positions[-1]['timestamp'] - hip_positions[0]['timestamp'],
                        'hip_drop_px': hip_drop,
                        'camera': cam,
                        'person': person,
                        'features': features
                    }

                    movements.append(movement)

        movements.sort(key=lambda x: x['start_time'])
        return movements

    def extract_features(self, track: List[Dict]) -> Dict:
        """Extract biomechanical features"""
        features = {}

        hip_ys = []
        torso_angles = []
        knee_angles = []

        for pose in track:
            kpts = np.array(pose['keypoints'])
            if len(kpts) < 17:
                continue

            # Hip height
            left_hip = kpts[11]
            right_hip = kpts[12]
            if left_hip[2] > 0.3 and right_hip[2] > 0.3:
                hip_ys.append((left_hip[1] + right_hip[1]) / 2)

            # Torso angle
            left_shoulder = kpts[5]
            if left_shoulder[2] > 0.3 and left_hip[2] > 0.3:
                vec = left_hip[:2] - left_shoulder[:2]
                angle = np.degrees(np.arctan2(vec[0], vec[1]))
                torso_angles.append(abs(angle))

            # Knee angle (left)
            left_knee = kpts[13]
            left_ankle = kpts[15]
            if left_hip[2] > 0.3 and left_knee[2] > 0.3 and left_ankle[2] > 0.3:
                vec1 = left_hip[:2] - left_knee[:2]
                vec2 = left_ankle[:2] - left_knee[:2]
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
                knee_angles.append(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

        if hip_ys:
            features['avg_hip_height'] = float(np.mean(hip_ys))
            features['min_hip_height'] = float(min(hip_ys))
            features['max_hip_height'] = float(max(hip_ys))
            features['hip_drop'] = float(max(hip_ys) - min(hip_ys))

        if torso_angles:
            features['avg_torso_angle'] = float(np.mean(torso_angles))
            features['max_torso_angle'] = float(max(torso_angles))

        if knee_angles:
            features['avg_knee_angle'] = float(np.mean(knee_angles))
            features['min_knee_angle'] = float(min(knee_angles))

        return features

    def generate_report(self, analysis: Dict) -> str:
        """Generate analysis report"""
        report = []
        report.append("="*60)
        report.append("JUDO VISUAL ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"\nVideo: {analysis['video_name']}")
        report.append(f"Duration: {analysis['duration']:.2f}s")
        report.append(f"Layout: {analysis['layout']} ({analysis['num_cameras']} cameras)")
        report.append(f"Poses detected: {len(analysis['poses'])}")

        movements = analysis['movements']

        report.append(f"\n{'='*60}")
        report.append("MOVEMENT DETECTION")
        report.append("="*60)

        if not movements:
            report.append("\n⚠️  No significant movements detected")
            return "\n".join(report)

        report.append(f"\n✓ Detected {len(movements)} movement(s)\n")

        for i, mov in enumerate(movements, 1):
            report.append(f"\n{i}. Movement at {mov['start_time']:.2f}s - {mov['end_time']:.2f}s")
            report.append(f"   Duration: {mov['duration']:.2f}s")
            report.append(f"   Camera: {mov['camera']} | Person: {mov['person']}")
            report.append(f"\n   Biomechanics:")
            report.append(f"     • Hip drop: {mov['hip_drop_px']:.1f} pixels")

            features = mov['features']
            if 'avg_hip_height' in features:
                report.append(f"     • Avg hip height: {features['avg_hip_height']:.1f}px")
            if 'avg_torso_angle' in features:
                report.append(f"     • Avg torso angle: {features['avg_torso_angle']:.1f}°")
            if 'avg_knee_angle' in features:
                report.append(f"     • Avg knee angle: {features['avg_knee_angle']:.1f}°")

        report.append(f"\n{'='*60}")
        return "\n".join(report)

    def save_results(self, analysis: Dict, report: str):
        """Save results"""
        video_name = analysis['video_name']

        json_path = self.output_dir / f"{video_name}_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n✓ Saved JSON: {json_path}")

        report_path = self.output_dir / f"{video_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✓ Saved report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='LionOfJudo Visual Analysis')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--output-dir', type=str, default='analysis/visual_results',
                       help='Output directory')

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return

    analyzer = VisualJudoAnalyzer(output_dir=Path(args.output_dir))
    analysis = analyzer.process_video(video_path)
    report = analyzer.generate_report(analysis)

    print("\n" + report)
    analyzer.save_results(analysis, report)

    print(f"\n{'='*60}")
    print("VISUAL ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
