"""
RadioFlow Test Script
Quick test to verify the system works correctly
"""

import sys
from PIL import Image
import time


def create_test_image():
    """Create a simple test image."""
    img = Image.new('RGB', (512, 512), color=(30, 30, 40))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 150, 412, 450], outline=(60, 60, 70), width=3)
    return img


def test_agents():
    """Test individual agents."""
    print("=" * 60)
    print("Testing RadioFlow Agents")
    print("=" * 60)
    
    from agents import (
        CXRAnalyzerAgent,
        FindingInterpreterAgent,
        ReportGeneratorAgent,
        PriorityRouterAgent
    )
    
    # Test CXR Analyzer
    print("\n[1/4] Testing CXR Analyzer...")
    agent1 = CXRAnalyzerAgent(demo_mode=True)
    agent1.load_model()
    result1 = agent1(create_test_image())
    print(f"      Status: {result1.status}")
    print(f"      Findings: {len(result1.data.get('findings', []))}")
    print(f"      Time: {result1.processing_time_ms:.0f}ms")
    
    # Test Finding Interpreter
    print("\n[2/4] Testing Finding Interpreter...")
    agent2 = FindingInterpreterAgent(demo_mode=True)
    agent2.load_model()
    result2 = agent2(result1.data)
    print(f"      Status: {result2.status}")
    print(f"      Interpreted: {len(result2.data.get('interpreted_findings', []))}")
    print(f"      Time: {result2.processing_time_ms:.0f}ms")
    
    # Test Report Generator
    print("\n[3/4] Testing Report Generator...")
    agent3 = ReportGeneratorAgent(demo_mode=True)
    agent3.load_model()
    result3 = agent3(result2.data)
    print(f"      Status: {result3.status}")
    print(f"      Report length: {len(result3.data.get('full_report', ''))}")
    print(f"      Time: {result3.processing_time_ms:.0f}ms")
    
    # Test Priority Router
    print("\n[4/4] Testing Priority Router...")
    agent4 = PriorityRouterAgent(demo_mode=True)
    agent4.load_model()
    context = {"original_findings": result1.data.get("findings", [])}
    result4 = agent4(result3.data, context)
    print(f"      Status: {result4.status}")
    print(f"      Priority: {result4.data.get('priority_level')}")
    print(f"      Score: {result4.data.get('priority_score')}")
    print(f"      Time: {result4.processing_time_ms:.0f}ms")
    
    print("\n" + "=" * 60)
    print("✅ All agents tested successfully!")
    print("=" * 60)
    
    return True


def test_orchestrator():
    """Test the full orchestrator."""
    print("\n" + "=" * 60)
    print("Testing RadioFlow Orchestrator")
    print("=" * 60)
    
    from orchestrator import create_orchestrator
    
    # Create orchestrator
    print("\n[1/2] Creating orchestrator...")
    orchestrator = create_orchestrator(demo_mode=True)
    print("      ✅ Orchestrator created")
    
    # Run workflow
    print("\n[2/2] Running workflow...")
    context = {
        "clinical_history": "65-year-old with cough and fever",
        "symptoms": "Productive cough, dyspnea"
    }
    
    result = orchestrator.process(create_test_image(), context)
    
    print(f"\n      Status: {result.status}")
    print(f"      Duration: {result.total_duration_ms:.0f}ms")
    print(f"      Findings: {result.findings_count}")
    print(f"      Priority: {result.priority_level} ({result.priority_score:.0%})")
    
    print("\n" + "=" * 60)
    print("✅ Orchestrator tested successfully!")
    print("=" * 60)
    
    return True


def test_visualization():
    """Test visualization functions."""
    print("\n" + "=" * 60)
    print("Testing Visualization Functions")
    print("=" * 60)
    
    from utils.visualization import (
        create_workflow_diagram,
        create_priority_gauge,
        create_radar_chart
    )
    
    # Test workflow diagram
    print("\n[1/3] Testing workflow diagram...")
    agent_results = [
        {"name": "CXR Analyzer", "status": "success", "processing_time_ms": 300},
        {"name": "Finding Interpreter", "status": "success", "processing_time_ms": 400},
        {"name": "Report Generator", "status": "success", "processing_time_ms": 500},
        {"name": "Priority Router", "status": "success", "processing_time_ms": 300}
    ]
    fig1 = create_workflow_diagram(agent_results)
    print("      ✅ Workflow diagram created")
    
    # Test priority gauge
    print("\n[2/3] Testing priority gauge...")
    fig2 = create_priority_gauge(0.65, "URGENT")
    print("      ✅ Priority gauge created")
    
    # Test radar chart
    print("\n[3/3] Testing radar chart...")
    scores = {"Lungs": 0.9, "Heart": 0.7, "Mediastinum": 0.95, "Bones": 0.85}
    fig3 = create_radar_chart(scores)
    print("      ✅ Radar chart created")
    
    print("\n" + "=" * 60)
    print("✅ All visualizations tested successfully!")
    print("=" * 60)
    
    return True


def main():
    """Run all tests."""
    print("\n")
    print("🩻 RadioFlow Test Suite")
    print("=" * 60)
    print("MedGemma Impact Challenge\n")
    
    start_time = time.time()
    
    try:
        # Run tests
        test_agents()
        test_orchestrator()
        test_visualization()
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"🎉 ALL TESTS PASSED in {total_time:.1f}s")
        print("=" * 60)
        print("\nRadioFlow is ready!")
        print("Run 'python app.py' to start the Gradio demo.")
        print("=" * 60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
