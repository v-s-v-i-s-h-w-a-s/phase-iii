#!/usr/bin/env python3
"""
Basketball GNN System Status Check
Verifies all components are working correctly
"""

import os
import sys
from pathlib import Path

def check_system_status():
    """Check if the Basketball GNN system is ready for use."""
    
    print("🏀 Basketball GNN System Status Check")
    print("=" * 50)
    print()
    
    # Check project structure
    required_files = [
        "main.py",
        "analyze_video.py", 
        "video_processor.py",
        "requirements.txt",
        "config.json",
        "README.md"
    ]
    
    required_dirs = [
        "gnn_model",
        "graph_builder", 
        "vis",
        "utils"
    ]
    
    print("📁 Project Structure:")
    all_files_present = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            all_files_present = False
    
    for dir in required_dirs:
        if os.path.isdir(dir):
            print(f"  ✅ {dir}/")
        else:
            print(f"  ❌ {dir}/ - MISSING")
            all_files_present = False
    
    print()
    
    # Check if analysis results exist
    print("📊 Previous Analysis Results:")
    if os.path.exists("results"):
        result_files = list(Path("results").glob("*.png")) + list(Path("results").glob("*.gif"))
        if result_files:
            print(f"  ✅ Found {len(result_files)} visualization files")
            for f in result_files[:3]:  # Show first 3
                print(f"    • {f.name}")
        else:
            print("  ⚠️  No visualization files (run analysis to generate)")
    else:
        print("  ⚠️  No results folder (will be created during analysis)")
    
    if os.path.exists("models"):
        model_files = list(Path("models").glob("*.pth"))
        if model_files:
            print(f"  ✅ Found {len(model_files)} trained model(s)")
        else:
            print("  ⚠️  No trained models (will be created during training)")
    else:
        print("  ⚠️  No models folder (will be created during training)")
    
    print()
    
    # Check video analysis results
    video_dirs = [d for d in os.listdir(".") if d.startswith("video_analysis_")]
    if video_dirs:
        print(f"🎥 Video Analysis Results: {len(video_dirs)} previous analysis found")
        for vd in video_dirs[:2]:  # Show first 2
            print(f"  ✅ {vd}")
    else:
        print("🎥 Video Analysis Results: None yet (process videos to generate)")
    
    print()
    
    # Check imports
    print("🔧 Dependencies Check:")
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError:
        print("  ❌ PyTorch - Install with: pip install torch")
        return False
    
    try:
        import torch_geometric
        print(f"  ✅ PyTorch Geometric {torch_geometric.__version__}")
    except ImportError:
        print("  ❌ PyTorch Geometric - Install with: pip install torch-geometric")
        return False
    
    try:
        import cv2
        print(f"  ✅ OpenCV")
    except ImportError:
        print("  ❌ OpenCV - Install with: pip install opencv-python")
        return False
    
    try:
        import ultralytics
        print(f"  ✅ Ultralytics (YOLOv8)")
    except ImportError:
        print("  ❌ Ultralytics - Install with: pip install ultralytics")
        return False
    
    print()
    
    # System recommendations
    print("💡 Quick Start Recommendations:")
    print()
    
    if not video_dirs:
        print("🎯 First Time User:")
        print("  1. Run: python analyze_video.py your_basketball_video.mp4")
        print("  2. Check: video_analysis_[name]/ for tracking data")
        print("  3. View: results/ for tactical visualizations")
    else:
        print("🚀 Experienced User:")
        print("  • Try different videos with various settings")
        print("  • Experiment with --confidence and --epochs parameters")
        print("  • Compare results across different game segments")
    
    print()
    print("📖 Documentation:")
    print("  • Full guide: README.md")
    print("  • Video guide: VIDEO_ANALYSIS_GUIDE.md")
    print("  • Help: python analyze_video.py --help")
    
    print()
    
    if all_files_present:
        print("🎉 System Status: READY FOR BASKETBALL ANALYSIS!")
        print("   Start with: python analyze_video.py your_video.mp4")
    else:
        print("⚠️  System Status: INCOMPLETE - Missing required files")
        print("   Please ensure all files are present")
    
    return all_files_present

if __name__ == "__main__":
    check_system_status()
