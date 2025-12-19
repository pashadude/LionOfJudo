#!/usr/bin/env python3
"""
LionOfJudo - Detailed Movement Analysis
Extracts comprehensive body movement patterns and generates detailed reports
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class MovementAnalyzer:
    """Analyzes detailed body movements from pose data"""

    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    def __init__(self):
        pass

    def load_analysis(self, json_path: Path) -> Dict:
        """Load analysis JSON"""
        with open(json_path) as f:
            return json.load(f)

    def extract_movement_phases(self, poses: List[Dict]) -> List[Dict]:
        """Extract distinct movement phases from pose data"""
        if not poses:
            return []

        phases = []
        current_phase = {
            'start_frame': 0,
            'poses': []
        }

        # Track hip velocity to identify phase transitions
        prev_hip_y = None
        velocity_threshold = 10  # pixels per frame

        for pose in poses:
            kpts = np.array(pose['keypoints'])
            if len(kpts) < 13:
                continue

            left_hip = kpts[11]
            right_hip = kpts[12]

            if left_hip[2] > 0.3 and right_hip[2] > 0.3:
                hip_y = (left_hip[1] + right_hip[1]) / 2

                if prev_hip_y is not None:
                    velocity = abs(hip_y - prev_hip_y)

                    # Phase transition detected
                    if velocity > velocity_threshold and len(current_phase['poses']) > 5:
                        current_phase['end_frame'] = pose['frame']
                        phases.append(current_phase.copy())

                        current_phase = {
                            'start_frame': pose['frame'],
                            'poses': []
                        }

                current_phase['poses'].append(pose)
                prev_hip_y = hip_y

        # Add final phase
        if current_phase['poses']:
            current_phase['end_frame'] = current_phase['poses'][-1]['frame']
            phases.append(current_phase)

        return phases

    def analyze_body_part_movements(self, poses: List[Dict]) -> Dict:
        """Analyze movements of individual body parts"""
        if not poses:
            return {}

        movements = {
            'head': self._analyze_head_movement(poses),
            'torso': self._analyze_torso_movement(poses),
            'arms': self._analyze_arm_movement(poses),
            'hands': self._analyze_hand_work(poses),
            'hips': self._analyze_hip_movement(poses),
            'legs': self._analyze_leg_movement(poses)
        }

        return movements

    def _analyze_head_movement(self, poses: List[Dict]) -> Dict:
        """Analyze head position and movement"""
        head_positions = []

        for pose in poses:
            kpts = np.array(pose['keypoints'])
            if len(kpts) > 0 and kpts[0][2] > 0.3:  # nose
                head_positions.append({
                    'frame': pose['frame'],
                    'x': kpts[0][0],
                    'y': kpts[0][1]
                })

        if not head_positions:
            return {'status': 'not_detected'}

        # Calculate movement range
        xs = [p['x'] for p in head_positions]
        ys = [p['y'] for p in head_positions]

        return {
            'status': 'detected',
            'horizontal_range': max(xs) - min(xs),
            'vertical_range': max(ys) - min(ys),
            'avg_position': (np.mean(xs), np.mean(ys)),
            'movement_description': self._describe_head_movement(xs, ys)
        }

    def _describe_head_movement(self, xs: List[float], ys: List[float]) -> str:
        """Generate textual description of head movement"""
        h_range = max(xs) - min(xs)
        v_range = max(ys) - min(ys)

        if h_range < 50 and v_range < 50:
            return "Head remains stable throughout movement"
        elif h_range > v_range:
            return f"Head moves horizontally ({h_range:.0f}px range), indicating rotation or lateral movement"
        else:
            return f"Head moves vertically ({v_range:.0f}px range), indicating level change"

    def _analyze_torso_movement(self, poses: List[Dict]) -> Dict:
        """Analyze torso angle and rotation"""
        torso_angles = []
        torso_lengths = []

        for pose in poses:
            kpts = np.array(pose['keypoints'])
            if len(kpts) >= 13:
                left_shoulder = kpts[5]
                left_hip = kpts[11]

                if left_shoulder[2] > 0.3 and left_hip[2] > 0.3:
                    vec = left_hip[:2] - left_shoulder[:2]
                    angle = np.degrees(np.arctan2(vec[0], vec[1]))
                    length = np.linalg.norm(vec)

                    torso_angles.append({
                        'frame': pose['frame'],
                        'angle': abs(angle)
                    })
                    torso_lengths.append(length)

        if not torso_angles:
            return {'status': 'not_detected'}

        angles = [t['angle'] for t in torso_angles]

        return {
            'status': 'detected',
            'min_angle': min(angles),
            'max_angle': max(angles),
            'avg_angle': np.mean(angles),
            'angle_range': max(angles) - min(angles),
            'avg_length': np.mean(torso_lengths) if torso_lengths else 0,
            'movement_description': self._describe_torso_movement(angles)
        }

    def _describe_torso_movement(self, angles: List[float]) -> str:
        """Generate textual description of torso movement"""
        avg_angle = np.mean(angles)
        angle_range = max(angles) - min(angles)

        desc = []

        if avg_angle < 10:
            desc.append("Torso remains very upright (vertical)")
        elif avg_angle < 20:
            desc.append("Torso maintains mostly upright position")
        elif avg_angle < 35:
            desc.append("Torso shows moderate forward lean")
        else:
            desc.append("Torso shows significant forward lean")

        if angle_range > 20:
            desc.append(f"with dynamic rotation ({angle_range:.1f}° range)")
        elif angle_range > 10:
            desc.append(f"with moderate rotation ({angle_range:.1f}° range)")
        else:
            desc.append("with minimal rotation")

        return " ".join(desc)

    def _analyze_arm_movement(self, poses: List[Dict]) -> Dict:
        """Analyze arm positions and movements"""
        left_arm_angles = []
        right_arm_angles = []

        for pose in poses:
            kpts = np.array(pose['keypoints'])
            if len(kpts) >= 11:
                # Left arm
                left_shoulder = kpts[5]
                left_elbow = kpts[7]
                left_wrist = kpts[9]

                if all(k[2] > 0.3 for k in [left_shoulder, left_elbow, left_wrist]):
                    vec1 = left_elbow[:2] - left_shoulder[:2]
                    vec2 = left_wrist[:2] - left_elbow[:2]
                    angle = self._calculate_angle(vec1, vec2)
                    left_arm_angles.append(angle)

                # Right arm
                right_shoulder = kpts[6]
                right_elbow = kpts[8]
                right_wrist = kpts[10]

                if all(k[2] > 0.3 for k in [right_shoulder, right_elbow, right_wrist]):
                    vec1 = right_elbow[:2] - right_shoulder[:2]
                    vec2 = right_wrist[:2] - right_elbow[:2]
                    angle = self._calculate_angle(vec1, vec2)
                    right_arm_angles.append(angle)

        return {
            'left_arm': {
                'status': 'detected' if left_arm_angles else 'not_detected',
                'avg_angle': np.mean(left_arm_angles) if left_arm_angles else 0,
                'min_angle': min(left_arm_angles) if left_arm_angles else 0,
                'max_angle': max(left_arm_angles) if left_arm_angles else 0,
                'range': max(left_arm_angles) - min(left_arm_angles) if left_arm_angles else 0
            },
            'right_arm': {
                'status': 'detected' if right_arm_angles else 'not_detected',
                'avg_angle': np.mean(right_arm_angles) if right_arm_angles else 0,
                'min_angle': min(right_arm_angles) if right_arm_angles else 0,
                'max_angle': max(right_arm_angles) if right_arm_angles else 0,
                'range': max(right_arm_angles) - min(right_arm_angles) if right_arm_angles else 0
            },
            'movement_description': self._describe_arm_movement(left_arm_angles, right_arm_angles)
        }

    def _analyze_hand_work(self, poses: List[Dict]) -> Dict:
        """Analyze detailed hand/wrist positioning and movement"""
        left_wrist_positions = []
        right_wrist_positions = []
        wrist_velocities = []
        grip_positions = []

        for i, pose in enumerate(poses):
            kpts = np.array(pose['keypoints'])
            if len(kpts) >= 11:
                # Track wrist positions
                left_wrist = kpts[9]
                right_wrist = kpts[10]
                left_shoulder = kpts[5]
                right_shoulder = kpts[6]
                left_hip = kpts[11]
                right_hip = kpts[12]

                if left_wrist[2] > 0.3 and left_shoulder[2] > 0.3 and left_hip[2] > 0.3:
                    # Wrist position relative to body
                    shoulder_hip_dist = abs(left_shoulder[1] - left_hip[1])
                    wrist_height = (left_wrist[1] - left_shoulder[1]) / (shoulder_hip_dist + 1)
                    wrist_lateral = left_wrist[0] - left_shoulder[0]

                    left_wrist_positions.append({
                        'frame': pose['frame'],
                        'x': left_wrist[0],
                        'y': left_wrist[1],
                        'height_ratio': wrist_height,
                        'lateral_offset': wrist_lateral
                    })

                if right_wrist[2] > 0.3 and right_shoulder[2] > 0.3 and right_hip[2] > 0.3:
                    shoulder_hip_dist = abs(right_shoulder[1] - right_hip[1])
                    wrist_height = (right_wrist[1] - right_shoulder[1]) / (shoulder_hip_dist + 1)
                    wrist_lateral = right_wrist[0] - right_shoulder[0]

                    right_wrist_positions.append({
                        'frame': pose['frame'],
                        'x': right_wrist[0],
                        'y': right_wrist[1],
                        'height_ratio': wrist_height,
                        'lateral_offset': wrist_lateral
                    })

                # Calculate wrist velocity (pull/push dynamics)
                if i > 0 and left_wrist_positions and len(left_wrist_positions) > 1:
                    prev_pos = left_wrist_positions[-2]
                    curr_pos = left_wrist_positions[-1]
                    velocity = np.sqrt((curr_pos['x'] - prev_pos['x'])**2 +
                                     (curr_pos['y'] - prev_pos['y'])**2)
                    wrist_velocities.append(velocity)

        return {
            'left_wrist_positions': left_wrist_positions,
            'right_wrist_positions': right_wrist_positions,
            'wrist_velocities': wrist_velocities,
            'avg_velocity': np.mean(wrist_velocities) if wrist_velocities else 0,
            'max_velocity': max(wrist_velocities) if wrist_velocities else 0
        }

    def _describe_arm_movement(self, left_angles: List[float], right_angles: List[float]) -> str:
        """Generate textual description of arm movement"""
        if not left_angles and not right_angles:
            return "Arms not clearly visible"

        desc = []

        if left_angles:
            avg_left = np.mean(left_angles)
            range_left = max(left_angles) - min(left_angles)

            if avg_left < 100:
                desc.append("Left arm tightly bent")
            elif avg_left < 140:
                desc.append("Left arm moderately bent")
            else:
                desc.append("Left arm mostly extended")

            # Add movement dynamics
            if range_left > 90:
                desc[-1] += " (strong pull-push action)"
            elif range_left > 50:
                desc[-1] += " (active movement)"

        if right_angles:
            avg_right = np.mean(right_angles)
            range_right = max(right_angles) - min(right_angles)

            if avg_right < 100:
                desc.append("Right arm tightly bent")
            elif avg_right < 140:
                desc.append("Right arm moderately bent")
            else:
                desc.append("Right arm mostly extended")

            # Add movement dynamics
            if range_right > 90:
                desc[-1] += " (strong pull-push action)"
            elif range_right > 50:
                desc[-1] += " (active movement)"

        return ", ".join(desc)

    def _analyze_hip_movement(self, poses: List[Dict]) -> Dict:
        """Analyze hip movement patterns"""
        hip_positions = []

        for pose in poses:
            kpts = np.array(pose['keypoints'])
            if len(kpts) >= 13:
                left_hip = kpts[11]
                right_hip = kpts[12]

                if left_hip[2] > 0.3 and right_hip[2] > 0.3:
                    avg_x = (left_hip[0] + right_hip[0]) / 2
                    avg_y = (left_hip[1] + right_hip[1]) / 2

                    hip_positions.append({
                        'frame': pose['frame'],
                        'x': avg_x,
                        'y': avg_y,
                        'width': abs(left_hip[0] - right_hip[0])
                    })

        if not hip_positions:
            return {'status': 'not_detected'}

        xs = [p['x'] for p in hip_positions]
        ys = [p['y'] for p in hip_positions]
        widths = [p['width'] for p in hip_positions]

        return {
            'status': 'detected',
            'horizontal_movement': max(xs) - min(xs),
            'vertical_movement': max(ys) - min(ys),
            'hip_drop': max(ys) - min(ys),
            'avg_hip_width': np.mean(widths),
            'movement_description': self._describe_hip_movement(xs, ys)
        }

    def _describe_hip_movement(self, xs: List[float], ys: List[float]) -> str:
        """Generate textual description of hip movement"""
        h_movement = max(xs) - min(xs)
        v_movement = max(ys) - min(ys)

        desc = []

        if v_movement > 100:
            desc.append(f"Large vertical hip drop ({v_movement:.0f}px) - significant level change")
        elif v_movement > 50:
            desc.append(f"Moderate hip drop ({v_movement:.0f}px)")
        else:
            desc.append(f"Minimal hip movement ({v_movement:.0f}px)")

        if h_movement > 100:
            desc.append(f"with substantial lateral shift ({h_movement:.0f}px)")
        elif h_movement > 50:
            desc.append(f"with moderate lateral movement")

        return " ".join(desc)

    def _analyze_leg_movement(self, poses: List[Dict]) -> Dict:
        """Analyze leg positions and movements"""
        left_knee_angles = []
        right_knee_angles = []
        stance_widths = []

        for pose in poses:
            kpts = np.array(pose['keypoints'])
            if len(kpts) >= 17:
                # Left leg
                left_hip = kpts[11]
                left_knee = kpts[13]
                left_ankle = kpts[15]

                if all(k[2] > 0.3 for k in [left_hip, left_knee, left_ankle]):
                    vec1 = left_hip[:2] - left_knee[:2]
                    vec2 = left_ankle[:2] - left_knee[:2]
                    angle = self._calculate_angle(vec1, vec2)
                    left_knee_angles.append(angle)

                # Right leg
                right_hip = kpts[12]
                right_knee = kpts[14]
                right_ankle = kpts[16]

                if all(k[2] > 0.3 for k in [right_hip, right_knee, right_ankle]):
                    vec1 = right_hip[:2] - right_knee[:2]
                    vec2 = right_ankle[:2] - right_knee[:2]
                    angle = self._calculate_angle(vec1, vec2)
                    right_knee_angles.append(angle)

                # Stance width
                if left_ankle[2] > 0.3 and right_ankle[2] > 0.3:
                    width = abs(left_ankle[0] - right_ankle[0])
                    stance_widths.append(width)

        return {
            'left_leg': {
                'status': 'detected' if left_knee_angles else 'not_detected',
                'avg_knee_angle': np.mean(left_knee_angles) if left_knee_angles else 0,
                'min_knee_angle': min(left_knee_angles) if left_knee_angles else 0,
                'max_knee_angle': max(left_knee_angles) if left_knee_angles else 0
            },
            'right_leg': {
                'status': 'detected' if right_knee_angles else 'not_detected',
                'avg_knee_angle': np.mean(right_knee_angles) if right_knee_angles else 0,
                'min_knee_angle': min(right_knee_angles) if right_knee_angles else 0,
                'max_knee_angle': max(right_knee_angles) if right_knee_angles else 0
            },
            'stance': {
                'avg_width': np.mean(stance_widths) if stance_widths else 0,
                'min_width': min(stance_widths) if stance_widths else 0,
                'max_width': max(stance_widths) if stance_widths else 0
            },
            'movement_description': self._describe_leg_movement(left_knee_angles, right_knee_angles, stance_widths)
        }

    def _describe_leg_movement(self, left_angles: List[float], right_angles: List[float],
                               widths: List[float]) -> str:
        """Generate textual description of leg movement"""
        desc = []

        if left_angles:
            avg_left = np.mean(left_angles)
            min_left = min(left_angles)

            if min_left < 90:
                desc.append(f"Left leg shows deep bend (min {min_left:.0f}°)")
            elif avg_left < 160:
                desc.append(f"Left leg moderately bent (avg {avg_left:.0f}°)")
            else:
                desc.append("Left leg mostly straight")

        if right_angles:
            avg_right = np.mean(right_angles)
            min_right = min(right_angles)

            if min_right < 90:
                desc.append(f"Right leg shows deep bend (min {min_right:.0f}°)")
            elif avg_right < 160:
                desc.append(f"Right leg moderately bent (avg {avg_right:.0f}°)")
            else:
                desc.append("Right leg mostly straight")

        if widths:
            avg_width = np.mean(widths)
            width_range = max(widths) - min(widths)

            if width_range > 100:
                desc.append(f"Stance width varies significantly ({width_range:.0f}px range)")
            elif avg_width > 200:
                desc.append(f"Wide stance maintained (avg {avg_width:.0f}px)")
            elif avg_width < 100:
                desc.append(f"Narrow stance (avg {avg_width:.0f}px)")

        return ". ".join(desc) if desc else "Leg movements not clearly detected"

    def _calculate_angle(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate angle between two vectors"""
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle

    def generate_detailed_report(self, analysis: Dict) -> str:
        """Generate comprehensive movement analysis report"""
        report = []
        report.append("="*80)
        report.append("DETAILED BODY MOVEMENT ANALYSIS")
        report.append("="*80)

        report.append(f"\nVideo: {analysis['video_name']}")
        report.append(f"Duration: {analysis['duration']:.2f}s")
        report.append(f"Total poses detected: {len(analysis['poses'])}")

        if not analysis['poses']:
            report.append("\n⚠️  No pose data available for analysis")
            return "\n".join(report)

        # Analyze all poses
        movements = self.analyze_body_part_movements(analysis['poses'])

        report.append(f"\n{'='*80}")
        report.append("BODY PART MOVEMENTS")
        report.append("="*80)

        # Head
        report.append(f"\n🧠 HEAD MOVEMENT:")
        if movements['head']['status'] == 'detected':
            report.append(f"   • {movements['head']['movement_description']}")
            report.append(f"   • Horizontal range: {movements['head']['horizontal_range']:.1f}px")
            report.append(f"   • Vertical range: {movements['head']['vertical_range']:.1f}px")
        else:
            report.append("   • Head position not clearly detected")

        # Torso
        report.append(f"\n💪 TORSO MOVEMENT:")
        if movements['torso']['status'] == 'detected':
            report.append(f"   • {movements['torso']['movement_description']}")
            report.append(f"   • Average angle: {movements['torso']['avg_angle']:.1f}°")
            report.append(f"   • Angle range: {movements['torso']['min_angle']:.1f}° to {movements['torso']['max_angle']:.1f}°")
        else:
            report.append("   • Torso movement not clearly detected")

        # Arms
        report.append(f"\n🤚 ARM MOVEMENTS:")
        report.append(f"   • {movements['arms']['movement_description']}")
        if movements['arms']['left_arm']['status'] == 'detected':
            report.append(f"   • Left arm angle range: {movements['arms']['left_arm']['min_angle']:.1f}° to {movements['arms']['right_arm']['max_angle']:.1f}°")
        if movements['arms']['right_arm']['status'] == 'detected':
            report.append(f"   • Right arm angle range: {movements['arms']['right_arm']['min_angle']:.1f}° to {movements['arms']['right_arm']['max_angle']:.1f}°")

        # Hand/Wrist work
        report.append(f"\n✋ HAND/WRIST WORK:")
        if movements['hands']['left_wrist_positions']:
            left_wrists = movements['hands']['left_wrist_positions']
            height_ratios = [w['height_ratio'] for w in left_wrists]
            lateral_offsets = [w['lateral_offset'] for w in left_wrists]

            avg_height = np.mean(height_ratios)
            height_range = max(height_ratios) - min(height_ratios)
            lateral_range = max(lateral_offsets) - min(lateral_offsets)

            # Describe hand positioning
            if avg_height < -0.2:
                position_desc = "Hands positioned high (above shoulders)"
            elif avg_height < 0.5:
                position_desc = "Hands positioned at shoulder/chest level"
            else:
                position_desc = "Hands positioned low (below chest)"

            report.append(f"   • {position_desc}")

            # Describe hand movement dynamics
            if height_range > 1.0:
                report.append(f"   • Strong vertical hand movement ({height_range:.1f} range)")
            elif height_range > 0.5:
                report.append(f"   • Moderate vertical hand movement ({height_range:.1f} range)")

            if lateral_range > 200:
                report.append(f"   • Wide lateral hand movement ({lateral_range:.1f}px range)")
            elif lateral_range > 100:
                report.append(f"   • Moderate lateral hand movement ({lateral_range:.1f}px range)")

        # Pull/push dynamics
        if movements['hands']['avg_velocity'] > 0:
            avg_vel = movements['hands']['avg_velocity']
            max_vel = movements['hands']['max_velocity']

            if max_vel > 50:
                report.append(f"   • Strong pull/push action (max velocity: {max_vel:.1f}px/frame)")
            elif max_vel > 30:
                report.append(f"   • Active pull/push dynamics (max velocity: {max_vel:.1f}px/frame)")
            elif max_vel > 15:
                report.append(f"   • Moderate hand movement (avg velocity: {avg_vel:.1f}px/frame)")
            else:
                report.append(f"   • Controlled, steady hand positioning")

        # Hips
        report.append(f"\n🔵 HIP MOVEMENT:")
        if movements['hips']['status'] == 'detected':
            report.append(f"   • {movements['hips']['movement_description']}")
            report.append(f"   • Total hip drop: {movements['hips']['hip_drop']:.1f}px")
            report.append(f"   • Horizontal shift: {movements['hips']['horizontal_movement']:.1f}px")
        else:
            report.append("   • Hip movement not clearly detected")

        # Legs
        report.append(f"\n🦵 LEG MOVEMENTS:")
        report.append(f"   • {movements['legs']['movement_description']}")
        if movements['legs']['stance']['avg_width'] > 0:
            report.append(f"   • Average stance width: {movements['legs']['stance']['avg_width']:.1f}px")

        # Movement summary
        if analysis.get('movements'):
            report.append(f"\n{'='*80}")
            report.append("TECHNIQUE EXECUTION SUMMARY")
            report.append("="*80)

            for i, mov in enumerate(analysis['movements'], 1):
                report.append(f"\n{i}. Movement Phase {i}")
                report.append(f"   Time: {mov['start_time']:.2f}s - {mov['end_time']:.2f}s")
                report.append(f"   Duration: {mov['duration']:.2f}s")
                report.append(f"   Hip drop: {mov['hip_drop_px']:.1f}px")

                if 'features' in mov:
                    features = mov['features']
                    if 'avg_hip_height' in features:
                        report.append(f"   Average hip height: {features['avg_hip_height']:.1f}px")
                    if 'avg_torso_angle' in features:
                        report.append(f"   Average torso angle: {features['avg_torso_angle']:.1f}°")
                    if 'avg_knee_angle' in features:
                        report.append(f"   Average knee angle: {features['avg_knee_angle']:.1f}°")

        report.append(f"\n{'='*80}")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Detailed movement analysis')
    parser.add_argument('json_file', type=str, help='Path to analysis JSON file')
    parser.add_argument('--output-dir', type=str, default='analysis/movement_reports',
                       help='Output directory for detailed reports')

    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = MovementAnalyzer()

    print(f"Loading analysis from: {json_path}")
    analysis = analyzer.load_analysis(json_path)

    print("Generating detailed movement report...")
    report = analyzer.generate_detailed_report(analysis)

    print("\n" + report)

    # Save report
    output_file = output_dir / f"{analysis['video_name']}_detailed_movement.txt"
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"\n✓ Saved detailed report: {output_file}")


if __name__ == "__main__":
    main()
