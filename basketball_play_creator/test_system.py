"""
Basketball Play Creator - System Test
Tests the complete pipeline from natural language to simulation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nlp_parser import PlayParser
from gnn_engine import PlaySimulator, PlayerState, PlayerRole
import json

def test_nlp_pipeline():
    """Test the NLP parsing pipeline"""
    print("🧠 Testing NLP Parser...")
    
    parser = PlayParser()
    
    test_descriptions = [
        "PG passes to SG, center sets screen, cut to basket",
        "Point guard dribbles left, power forward sets pick, shooting guard cuts to corner",
        "Run a pick and roll with center, then pass to wing for three-pointer"
    ]
    
    for i, description in enumerate(test_descriptions, 1):
        print(f"\nTest {i}: '{description}'")
        
        result = parser.parse(description)
        
        print(f"  ✅ Play Name: {result.name}")
        print(f"  ✅ Confidence: {result.confidence:.2f}")
        print(f"  ✅ Actions: {len(result.actions)}")
        print(f"  ✅ Players: {len(result.players)}")
        
        # Convert to JSON to test serialization
        play_dict = result.to_dict()
        print(f"  ✅ JSON serialization: {len(json.dumps(play_dict))} chars")
    
    print("\n✅ NLP Pipeline test passed!")

def test_gnn_pipeline():
    """Test the GNN simulation pipeline"""
    print("\n🧠 Testing GNN Simulator...")
    
    simulator = PlaySimulator()
    
    # Create test players
    test_players = [
        PlayerState(
            position=(470, 400),
            velocity=(0, 0),
            role=PlayerRole.BALL_HANDLER,
            has_ball=True,
            fatigue=0.0,
            skill_level=0.8
        ),
        PlayerState(
            position=(600, 350),
            velocity=(0, 0),
            role=PlayerRole.SHOOTER,
            has_ball=False,
            fatigue=0.0,
            skill_level=0.7
        ),
        PlayerState(
            position=(340, 350),
            velocity=(0, 0),
            role=PlayerRole.CUTTER,
            has_ball=False,
            fatigue=0.0,
            skill_level=0.6
        ),
        PlayerState(
            position=(470, 200),
            velocity=(0, 0),
            role=PlayerRole.SCREENER,
            has_ball=False,
            fatigue=0.0,
            skill_level=0.7
        ),
        PlayerState(
            position=(550, 300),
            velocity=(0, 0),
            role=PlayerRole.SUPPORT,
            has_ball=False,
            fatigue=0.0,
            skill_level=0.6
        )
    ]
    
    print(f"  ✅ Created {len(test_players)} test players")
    
    # Run simulation
    result = simulator.simulate_play(test_players, [], duration=5.0)
    
    print("  ✅ Simulation completed")
    print(f"  ✅ States generated: {len(result.states)}")
    print(f"  ✅ Success probability: {result.success_probability:.2f}")
    print(f"  ✅ Final score probability: {result.final_score_probability:.2f}")
    print(f"  ✅ Key interactions: {len(result.key_interactions)}")
    print(f"  ✅ Tactical analysis: {len(result.tactical_analysis)} metrics")
    print(f"  ✅ Optimization suggestions: {len(result.optimization_suggestions)}")
    
    print("\n✅ GNN Pipeline test passed!")

def test_integrated_pipeline():
    """Test the complete integrated pipeline"""
    print("\n🏀 Testing Integrated Pipeline...")
    
    # Parse a play
    parser = PlayParser()
    description = "PG passes to SG, center sets screen for point guard, cut to basket"
    parsed_play = parser.parse(description)
    
    print(f"  ✅ Parsed play: {parsed_play.name}")
    
    # Convert parsed players to simulation format
    sim_players = []
    for player in parsed_play.players:
        sim_player = PlayerState(
            position=player.location,
            velocity=(0, 0),
            role=PlayerRole.BALL_HANDLER if player.id == "player_1" else PlayerRole.SUPPORT,
            has_ball=player.id == "player_1",
            fatigue=0.0,
            skill_level=0.7
        )
        sim_players.append(sim_player)
    
    # Run simulation with parsed actions
    simulator = PlaySimulator()
    sim_result = simulator.simulate_play(sim_players, parsed_play.actions, parsed_play.duration)
    
    print(f"  ✅ Simulation success: {sim_result.success_probability:.2f}")
    
    # Generate optimization suggestions
    if sim_result.optimization_suggestions:
        print(f"  ✅ Generated {len(sim_result.optimization_suggestions)} optimizations")
        for suggestion in sim_result.optimization_suggestions:
            print(f"    • {suggestion}")
    
    print("\n✅ Integrated Pipeline test passed!")

def test_api_compatibility():
    """Test API data format compatibility"""
    print("\n🔧 Testing API Compatibility...")
    
    # Test NLP output format
    parser = PlayParser()
    parsed = parser.parse("PG passes to SG")
    play_data = parsed.to_dict()
    
    required_fields = ['name', 'description', 'actions', 'players', 'formation', 'duration', 'confidence']
    for field in required_fields:
        assert field in play_data, f"Missing required field: {field}"
    
    print("  ✅ NLP output format valid")
    
    # Test GNN simulation format
    simulator = PlaySimulator()
    test_player = PlayerState((470, 400), (0, 0), PlayerRole.BALL_HANDLER, True, 0.0, 0.8)
    result = simulator.simulate_play([test_player], [], 2.0)
    
    required_sim_fields = ['states', 'final_score_probability', 'success_probability', 'key_interactions', 'tactical_analysis', 'optimization_suggestions']
    for field in required_sim_fields:
        assert hasattr(result, field), f"Missing required simulation field: {field}"
    
    print("  ✅ GNN output format valid")
    
    print("\n✅ API Compatibility test passed!")

def main():
    """Run all tests"""
    print("🏀 Basketball Play Creator - System Test")
    print("="*50)
    
    try:
        test_nlp_pipeline()
        test_gnn_pipeline()
        test_integrated_pipeline()
        test_api_compatibility()
        
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nTo start the application:")
        print("  • Windows: run start.bat")
        print("  • macOS/Linux: run ./start.sh")
        print("  • Manual: Follow instructions in SETUP.md")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nPlease check:")
        print("  • Python dependencies are installed")
        print("  • spaCy model is downloaded")
        print("  • PyTorch is properly installed")
        print("  • All modules are in the correct directories")
        
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
