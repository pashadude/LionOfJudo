#!/usr/bin/env python3
"""
Comprehensive Wrestling/Judo Analysis Pipeline
Processes all videos and generates detailed movement reports and comparisons
"""

import subprocess
import sys
from pathlib import Path
import json
import time


class WrestlingAnalysisPipeline:
    """Complete analysis pipeline for wrestling/judo videos"""

    def __init__(self):
        self.training_videos = {
            'uki-goshi': 'examples/uki-goshi.mov',
            'o-soto-gari': 'examples/o-soto-gari.mov',
            'o-goshi': 'examples/o-goshi.mov',
            'ippon-seoi-nagi': 'examples/Ippon-seoi-Nagi.mov'
        }

        self.pupil_videos = {
            'o-soto-gari-x-uki-goshi': 'data/Lav/o-soto-gari-x-uki-goshi.mov'
        }

        self.results_dir = Path('analysis/wrestling_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def check_analysis_exists(self, video_name: str) -> bool:
        """Check if analysis already exists"""
        json_file = Path(f'analysis/visual_results/{video_name}_analysis.json')
        return json_file.exists()

    def generate_detailed_movement_report(self, video_name: str):
        """Generate detailed movement analysis report"""
        json_file = Path(f'analysis/visual_results/{video_name}_analysis.json')

        if not json_file.exists():
            print(f"⚠️  Analysis not found for {video_name}")
            return False

        print(f"\n📊 Generating detailed movement report for {video_name}...")

        cmd = [
            sys.executable,
            'movement_analysis.py',
            str(json_file),
            '--output-dir', str(self.results_dir / 'detailed_movements')
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=False)
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Error generating report: {e}")
            return False

    def create_technique_comparison(self, technique: str, training_video: str, performance_video: str):
        """Create comparison between training and performance"""
        training_json = Path(f'analysis/visual_results/{training_video}_analysis.json')
        performance_json = Path(f'analysis/visual_results/{performance_video}_analysis.json')

        if not training_json.exists() or not performance_json.exists():
            print(f"⚠️  Cannot compare - missing analysis files")
            return False

        print(f"\n🔄 Creating comparison: {technique}")
        print(f"   Training: {training_video}")
        print(f"   Performance: {performance_video}")

        cmd = [
            sys.executable,
            'compare_techniques.py',
            '--training', str(training_json),
            '--performance', str(performance_json),
            '--create-video',
            '--max-duration', '15'
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=False)
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Error creating comparison: {e}")
            return False

    def generate_master_report(self):
        """Generate master summary report of all analyses"""
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE WRESTLING/JUDO ANALYSIS - MASTER REPORT")
        report.append("="*80)
        report.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Training videos section
        report.append(f"\n{'='*80}")
        report.append("TRAINING VIDEOS (Rubber Band Practice)")
        report.append("="*80)

        for technique, video_path in self.training_videos.items():
            video_name = Path(video_path).stem
            json_file = Path(f'analysis/visual_results/{video_name}_analysis.json')

            if json_file.exists():
                with open(json_file) as f:
                    data = json.load(f)

                report.append(f"\n📹 {technique.upper()}")
                report.append(f"   File: {video_path}")
                report.append(f"   Duration: {data['duration']:.2f}s")
                report.append(f"   Poses detected: {len(data['poses'])}")
                report.append(f"   Movements detected: {len(data.get('movements', []))}")

                if data.get('movements'):
                    for i, mov in enumerate(data['movements'][:3], 1):  # First 3 movements
                        report.append(f"   Movement {i}: {mov['duration']:.2f}s, Hip drop: {mov['hip_drop_px']:.1f}px")
            else:
                report.append(f"\n📹 {technique.upper()}")
                report.append(f"   ⚠️  Analysis pending or failed")

        # Pupil videos section
        report.append(f"\n{'='*80}")
        report.append("PUPIL PERFORMANCE VIDEOS")
        report.append("="*80)

        for video_name, video_path in self.pupil_videos.items():
            json_file = Path(f'analysis/visual_results/{video_name}_analysis.json')

            if json_file.exists():
                with open(json_file) as f:
                    data = json.load(f)

                report.append(f"\n📹 {video_name.upper()}")
                report.append(f"   File: {video_path}")
                report.append(f"   Duration: {data['duration']:.2f}s")
                report.append(f"   Layout: {data.get('layout', 'unknown')} ({data.get('num_cameras', 0)} cameras)")
                report.append(f"   Poses detected: {len(data['poses'])}")
                report.append(f"   Movements detected: {len(data.get('movements', []))}")

                if data.get('movements'):
                    report.append(f"\n   Movement Summary:")
                    for i, mov in enumerate(data['movements'][:5], 1):  # First 5 movements
                        report.append(f"   {i}. Time: {mov['start_time']:.1f}s-{mov['end_time']:.1f}s, "
                                    f"Hip drop: {mov['hip_drop_px']:.1f}px, "
                                    f"Person: {mov['person']}, Camera: {mov['camera']}")
            else:
                report.append(f"\n📹 {video_name.upper()}")
                report.append(f"   ⚠️  Analysis pending or failed")

        # Generated outputs
        report.append(f"\n{'='*80}")
        report.append("GENERATED OUTPUTS")
        report.append("="*80)

        annotated_videos = list(Path('analysis/visual_results').glob('*_annotated.mp4'))
        report.append(f"\n✓ Annotated videos: {len(annotated_videos)}")
        for video in annotated_videos[:10]:  # First 10
            size_mb = video.stat().st_size / (1024 * 1024)
            report.append(f"   - {video.name} ({size_mb:.1f}MB)")

        comparison_videos = list(Path('analysis/comparisons').glob('*.mp4'))
        report.append(f"\n✓ Comparison videos: {len(comparison_videos)}")
        for video in comparison_videos:
            size_mb = video.stat().st_size / (1024 * 1024)
            report.append(f"   - {video.name} ({size_mb:.1f}MB)")

        detailed_reports = list(Path(self.results_dir / 'detailed_movements').glob('*.txt'))
        report.append(f"\n✓ Detailed movement reports: {len(detailed_reports)}")

        report.append(f"\n{'='*80}")
        report.append("ANALYSIS COMPLETE")
        report.append("="*80)

        report_text = "\n".join(report)

        # Save master report
        master_report_path = self.results_dir / 'MASTER_ANALYSIS_REPORT.txt'
        with open(master_report_path, 'w') as f:
            f.write(report_text)

        print("\n" + report_text)
        print(f"\n✓ Saved master report: {master_report_path}")

    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("="*80)
        print("COMPREHENSIVE WRESTLING/JUDO ANALYSIS PIPELINE")
        print("="*80)

        # Check which videos are already processed
        print("\n📋 Checking existing analyses...")

        processed_training = []
        pending_training = []

        for technique, video_path in self.training_videos.items():
            video_name = Path(video_path).stem
            if self.check_analysis_exists(video_name):
                processed_training.append(technique)
                print(f"   ✓ {technique}: Already processed")
            else:
                pending_training.append(technique)
                print(f"   ⏳ {technique}: Processing or pending")

        processed_pupil = []
        pending_pupil = []

        for video_name, video_path in self.pupil_videos.items():
            if self.check_analysis_exists(video_name):
                processed_pupil.append(video_name)
                print(f"   ✓ {video_name}: Already processed")
            else:
                pending_pupil.append(video_name)
                print(f"   ⏳ {video_name}: Processing or pending")

        # Generate detailed movement reports
        print(f"\n{'='*80}")
        print("GENERATING DETAILED MOVEMENT REPORTS")
        print("="*80)

        for technique in processed_training:
            video_name = Path(self.training_videos[technique]).stem
            self.generate_detailed_movement_report(video_name)

        for video_name in processed_pupil:
            self.generate_detailed_movement_report(video_name)

        # Create comparisons
        print(f"\n{'='*80}")
        print("CREATING TECHNIQUE COMPARISONS")
        print("="*80)

        # Map pupil video techniques to training videos
        comparisons = [
            ('o-soto-gari', 'o-soto-gari', 'o-soto-gari-x-uki-goshi'),
            ('uki-goshi', 'uki-goshi', 'o-soto-gari-x-uki-goshi'),
        ]

        for technique, training_key, pupil_key in comparisons:
            training_video = Path(self.training_videos.get(training_key, '')).stem
            if self.check_analysis_exists(training_video) and self.check_analysis_exists(pupil_key):
                self.create_technique_comparison(technique, training_video, pupil_key)
            else:
                print(f"⏳ Cannot compare {technique} - analyses not ready")

        # Generate master report
        print(f"\n{'='*80}")
        print("GENERATING MASTER SUMMARY REPORT")
        print("="*80)

        self.generate_master_report()

        print(f"\n{'='*80}")
        print("✅ COMPLETE ANALYSIS PIPELINE FINISHED")
        print("="*80)
        print(f"\nResults saved in: {self.results_dir}")


def main():
    pipeline = WrestlingAnalysisPipeline()
    pipeline.run_complete_analysis()


if __name__ == "__main__":
    main()
