"""
Quick setup test for Basketball GNN project
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported."""
    
    print("Testing Basketball GNN setup...")
    
    try:
        # Test core dependencies
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import torch_geometric
        print("✅ PyTorch Geometric")
        
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib")
        
        import networkx as nx
        print("✅ NetworkX")
        
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
        
        # Test project modules
        sys.path.append('.')
        
        from graph_builder.build_graph import BasketballGraphBuilder
        print("✅ Graph Builder")
        
        from gnn_model.model import create_model
        print("✅ GNN Models")
        
        from vis.visualize_graph import BasketballGraphVisualizer
        print("✅ Visualization")
        
        from utils.yolo_tracking_parser import YOLOTrackingParser
        print("✅ YOLO Parser")
        
        from utils.pose_loader import PoseDataLoader
        print("✅ Pose Loader")
        
        print("\n🎉 All imports successful! Ready to use Basketball GNN.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nPlease install missing dependencies:")
        print("pip install -r requirements.txt")
        return False

def test_quick_demo():
    """Run a quick demo to test functionality."""
    
    print("\n" + "="*50)
    print("Running Quick Demo...")
    print("="*50)
    
    try:
        # Import modules
        from graph_builder.build_graph import BasketballGraphBuilder, create_dummy_tracking_data
        from gnn_model.model import create_model
        import torch
        
        # Create dummy data
        print("1. Creating dummy tracking data...")
        tracking_data = create_dummy_tracking_data(num_frames=10, num_players=6)
        print(f"   Generated {len(tracking_data)} tracking records")
        
        # Build graphs
        print("2. Building graphs...")
        builder = BasketballGraphBuilder()
        graphs = builder.build_sequence_graphs(tracking_data)
        print(f"   Built {len(graphs)} graphs")
        
        if graphs:
            sample_graph = graphs[0]
            print(f"   Sample graph: {sample_graph.num_nodes} nodes, {sample_graph.edge_index.shape[1]} edges")
            
            # Test model creation
            print("3. Testing model creation...")
            model = create_model("gcn", 
                               in_channels=sample_graph.x.shape[1], 
                               hidden_channels=16, 
                               out_channels=8)
            
            # Test forward pass
            with torch.no_grad():
                embeddings = model(sample_graph.x, sample_graph.edge_index)
            print(f"   Model output shape: {embeddings.shape}")
            
            print("\n✅ Quick demo completed successfully!")
            print("You can now run the full pipeline with:")
            print("python main.py --demo --train --epochs 10")
            
        else:
            print("❌ No graphs generated")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def check_gpu():
    """Check GPU availability."""
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("💻 Using CPU (GPU not available)")
    except Exception:
        print("❓ Could not check GPU status")

def main():
    """Main test function."""
    
    print("Basketball GNN Setup Test")
    print("=" * 40)
    
    # Check imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Check GPU
        print("\n" + "-" * 40)
        check_gpu()
        
        # Run demo
        test_quick_demo()
        
        print("\n" + "=" * 50)
        print("🏀 Basketball GNN is ready to use!")
        print("Next steps:")
        print("1. Run full demo: python main.py --demo --train")
        print("2. Use your data: python main.py --tracking your_data.csv")
        print("3. Check README.md for detailed instructions")
        print("=" * 50)
    else:
        print("\n❌ Setup incomplete. Please fix imports first.")

if __name__ == "__main__":
    main()
