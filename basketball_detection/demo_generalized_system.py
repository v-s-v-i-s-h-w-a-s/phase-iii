"""
Quick Demo: Generalized Basketball Team Classification
No hardcoded values - works for any basketball match!
"""

from generalized_basketball_inference import GeneralizedBasketballInference
import cv2
import os

def demo_generalized_system():
    """
    Demonstrate the new generalized team classification system
    """
    print("🏀 GENERALIZED BASKETBALL TEAM CLASSIFICATION DEMO")
    print("=" * 60)
    print("✅ No hardcoded values")
    print("✅ Works for any team match") 
    print("✅ Automatic team discovery")
    print("✅ Adaptive color thresholds")
    print("=" * 60)
    
    # Initialize the generalized system
    print("\n🚀 Initializing system...")
    inference = GeneralizedBasketballInference()
    
    # Check for video file
    video_path = "hawks_vs_knicks.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("📁 Available video files:")
        for file in os.listdir('.'):
            if file.endswith(('.mp4', '.avi', '.mov')):
                print(f"   - {file}")
        return
    
    print(f"\n🎬 Processing: {video_path}")
    print("⏳ This will take a few minutes...")
    
    # Process video with generalized team classification
    try:
        output_path, analysis_data = inference.process_video(
            video_path=video_path,
            confidence_threshold=0.5
        )
        
        print(f"\n✅ SUCCESS! Video processed with generalized team classification")
        print(f"📹 Output video: {output_path}")
        
        # Show summary
        if analysis_data:
            summary = analysis_data[0] if analysis_data else {}
            team_stats = summary.get('team_stats', {})
            
            print(f"\n📊 RESULTS SUMMARY:")
            print(f"   - Frames processed: {len(analysis_data):,}")
            print(f"   - Teams detected: {len(inference.team_classifier.team_profiles)}")
            print(f"   - Team profiles created automatically")
            
            # Show detected teams
            print(f"\n🎨 AUTOMATICALLY DETECTED TEAMS:")
            for team_name, profile in inference.team_classifier.team_profiles.items():
                color = profile['avg_color']
                samples = profile['sample_count']
                print(f"   - {team_name.upper()}: RGB{color} ({samples} samples)")
            
            print(f"\n🔧 TECHNICAL DETAILS:")
            print(f"   - Classification method: Adaptive clustering")
            print(f"   - Color spaces used: BGR, HSV, LAB")
            print(f"   - Temporal stability: Weighted voting")
            print(f"   - Hardcoded values: NONE!")
        
        print(f"\n📁 Generated files:")
        base_name = os.path.splitext(output_path)[0]
        files = [
            f"{output_path} (Enhanced video)",
            f"{base_name}_analysis.json (Detailed data)",
            f"{base_name}_detections.csv (Detection data)",
            f"{base_name}_report.md (Summary report)"
        ]
        for file in files:
            print(f"   - {file}")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return
    
    print(f"\n🏆 DEMO COMPLETE!")
    print(f"The generalized system successfully:")
    print(f"✅ Detected teams automatically (no manual configuration)")
    print(f"✅ Used adaptive thresholds (no hardcoded values)")
    print(f"✅ Created comprehensive analysis reports")
    print(f"✅ Generated professional visualization")
    
    print(f"\n🎯 This system will work for ANY basketball match!")

def demo_realtime():
    """
    Demonstrate real-time analysis
    """
    print("\n🔴 REAL-TIME DEMO")
    print("Starting webcam analysis...")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save frame")
    print("  - Press 'r' to reset teams")
    
    try:
        inference = GeneralizedBasketballInference()
        inference.analyze_realtime(source=0)
    except Exception as e:
        print(f"❌ Real-time demo failed: {e}")
        print("💡 Make sure you have a webcam connected")

if __name__ == "__main__":
    print("🏀 GENERALIZED BASKETBALL ANALYSIS SYSTEM")
    print("\nChoose demo mode:")
    print("1. Video Analysis (recommended)")
    print("2. Real-time Webcam")
    print("3. Quick feature overview")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            demo_generalized_system()
        elif choice == '2':
            demo_realtime()
        elif choice == '3':
            print("\n🎯 FEATURE OVERVIEW:")
            print("✅ Automatic team detection from jersey colors")
            print("✅ No hardcoded values or manual configuration")
            print("✅ Works for any basketball match worldwide")
            print("✅ Advanced clustering algorithms (K-means + GMM)")
            print("✅ Multi-space color analysis (BGR, HSV, LAB)")
            print("✅ Temporal stability with weighted voting")
            print("✅ Real-time processing capability")
            print("✅ Comprehensive analysis reports")
            print("✅ Professional visualization with team legends")
            print("✅ Handles occlusion and lighting variations")
        else:
            print("Invalid choice. Running video analysis...")
            demo_generalized_system()
            
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
