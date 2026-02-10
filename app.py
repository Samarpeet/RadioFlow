"""
RadioFlow: AI-Powered Radiology Workflow Agent
Main Gradio Application for MedGemma Impact Challenge

This application demonstrates a multi-agent system for chest X-ray analysis
using Google's Health AI Developer Foundations (HAI-DEF) models.

Now with REAL MedGemma inference via MLX (local) or ZeroGPU (HuggingFace).
"""

import os
import gradio as gr
from PIL import Image
import time
from typing import Optional, Tuple, List, Dict
import json

# HuggingFace Spaces detection
SPACES_AVAILABLE = os.environ.get("SPACE_ID") is not None

# Import our modules
from orchestrator import RadioFlowOrchestrator, WorkflowResult, create_orchestrator
from utils.visualization import (
    create_workflow_diagram,
    create_radar_chart,
    create_priority_gauge,
    create_timeline_chart
)

# Check if we're on HuggingFace Spaces with ZeroGPU
IS_SPACES = os.environ.get("SPACE_ID") is not None
USE_ZEROGPU = IS_SPACES and os.environ.get("ZEROGPU_ENABLED") == "true"

# Determine if we should use demo mode
# - Local with MLX: Use real model (demo_mode=False)
# - HuggingFace without GPU: Use demo mode (demo_mode=True) 
# - HuggingFace with ZeroGPU: Use real model (demo_mode=False)
FORCE_DEMO_MODE = os.environ.get("FORCE_DEMO_MODE", "false").lower() == "true"

# Global orchestrator instance
orchestrator: Optional[RadioFlowOrchestrator] = None
engine_status = "Not initialized"


def initialize_system():
    """Initialize the RadioFlow system."""
    global orchestrator, engine_status
    
    if orchestrator is None:
        # Check if we're on HuggingFace Spaces (CPU) - use demo mode for fast startup
        # Real MedGemma runs locally (MLX) or on Kaggle (GPU)
        demo_mode = IS_SPACES or FORCE_DEMO_MODE
        
        if not demo_mode:
            try:
                # Try to load the MedGemma engine (only for local/GPU)
                from agents.medgemma_engine import get_engine
                engine = get_engine(force_demo=False)
                engine_status = f"MedGemma: {engine.backend}"
                
                if engine.backend == "demo":
                    demo_mode = True
            except Exception as e:
                print(f"Could not initialize MedGemma engine: {e}")
                engine_status = "Demo mode"
                demo_mode = True
        else:
            engine_status = "Demo mode (HuggingFace CPU)"
        
        orchestrator = create_orchestrator(demo_mode=demo_mode)
        
    return f"✅ RadioFlow System Initialized ({engine_status})"


def process_xray(
    image: Optional[Image.Image],
    clinical_history: str,
    patient_age: str,
    symptoms: str,
    progress=gr.Progress()
) -> Tuple[str, str, str, str, str, dict, dict, dict]:
    """
    Process a chest X-ray through the RadioFlow pipeline.
    Uses real MedGemma inference with GPU acceleration.
    
    Returns:
        Tuple of (report, priority_html, findings_json, metrics, status, 
                  workflow_fig, radar_fig, priority_fig)
    """
    global orchestrator
    
    if image is None:
        return (
            "⚠️ Please upload a chest X-ray image.",
            "",
            "{}",
            "",
            "No image uploaded",
            None, None, None
        )
    
    # Initialize if needed
    if orchestrator is None:
        initialize_system()
    
    # Prepare clinical context
    context = {
        "clinical_history": clinical_history or "Not provided",
        "age": patient_age or "Not provided",
        "symptoms": symptoms or "Not provided"
    }
    
    # Progress updates
    progress(0.1, desc="🔬 Analyzing chest X-ray...")
    time.sleep(0.2)
    
    progress(0.3, desc="📋 Interpreting findings...")
    
    # Run the workflow
    result = orchestrator.process(image, context)
    
    progress(0.6, desc="📝 Generating report...")
    time.sleep(0.1)
    
    progress(0.8, desc="🚦 Assessing priority...")
    time.sleep(0.1)
    
    progress(1.0, desc="✅ Complete!")
    
    # Format outputs
    report = result.final_report if result.final_report else "Report generation failed."
    
    # Priority HTML
    priority_html = format_priority_display(result)
    
    # Findings JSON
    findings = []
    if result.cxr_analysis and result.cxr_analysis.data:
        findings = result.cxr_analysis.data.get("findings", [])
    findings_json = json.dumps(findings, indent=2)
    
    # Metrics
    metrics = format_metrics(result)
    
    # Status
    status = f"✅ Workflow {result.status.upper()} | {result.total_duration_ms:.0f}ms"
    
    # Create visualizations
    agent_results = []
    for agent_result in [result.cxr_analysis, result.finding_interpretation, 
                         result.report, result.priority_routing]:
        if agent_result:
            agent_results.append({
                "name": agent_result.agent_name,
                "status": agent_result.status,
                "processing_time_ms": agent_result.processing_time_ms
            })
    
    workflow_fig = create_workflow_diagram(agent_results)
    
    # Radar chart for analysis scores
    if result.cxr_analysis and result.cxr_analysis.data:
        region_analysis = result.cxr_analysis.data.get("region_analysis", {})
        scores = {}
        for region, data in list(region_analysis.items())[:5]:
            clean_name = region.replace("_", " ").title()
            scores[clean_name] = data.get("confidence", 0.5)
        if scores:
            radar_fig = create_radar_chart(scores, "Regional Confidence Scores")
        else:
            radar_fig = None
    else:
        radar_fig = None
    
    # Priority gauge
    priority_fig = create_priority_gauge(result.priority_score, result.priority_level)
    
    return (
        report,
        priority_html,
        findings_json,
        metrics,
        status,
        workflow_fig,
        radar_fig,
        priority_fig
    )


def format_priority_display(result: WorkflowResult) -> str:
    """Format priority information as HTML."""
    level = result.priority_level
    score = result.priority_score
    
    colors = {
        "STAT": "#ef4444",
        "URGENT": "#f59e0b",
        "ROUTINE": "#22c55e"
    }
    color = colors.get(level, "#6b7280")
    
    critical_html = ""
    if result.critical_findings:
        critical_html = f"""
        <div style="margin-top: 10px; padding: 10px; background: #fef2f2; border-radius: 5px; border-left: 4px solid #ef4444;">
            <strong>⚠️ Critical Findings:</strong>
            <ul style="margin: 5px 0 0 20px;">
                {"".join(f"<li>{f}</li>" for f in result.critical_findings)}
            </ul>
        </div>
        """
    
    routing_html = ""
    if result.priority_routing and result.priority_routing.data:
        routing = result.priority_routing.data.get("routing_recommendation", {})
        if routing:
            routing_html = f"""
            <div style="margin-top: 10px; padding: 10px; background: #f0f9ff; border-radius: 5px;">
                <strong>📍 Routing:</strong> {routing.get("destination", "Standard Queue")}
            </div>
            """
    
    return f"""
    <div style="padding: 15px; border-radius: 10px; background: linear-gradient(135deg, {color}22, {color}11);">
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="
                width: 60px; 
                height: 60px; 
                background: {color}; 
                border-radius: 50%; 
                display: flex; 
                align-items: center; 
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 14px;
            ">
                {level}
            </div>
            <div>
                <div style="font-size: 24px; font-weight: bold; color: {color};">
                    Priority Score: {score:.0%}
                </div>
                <div style="color: #666; font-size: 14px;">
                    {result.findings_count} finding(s) detected
                </div>
            </div>
        </div>
        {critical_html}
        {routing_html}
    </div>
    """


def format_metrics(result: WorkflowResult) -> str:
    """Format workflow metrics."""
    lines = [
        "## 📊 Workflow Metrics",
        "",
        f"**Total Duration:** {result.total_duration_ms:.0f}ms",
        f"**Status:** {result.status.upper()}",
        f"**Findings Detected:** {result.findings_count}",
        "",
        "### Agent Performance",
        ""
    ]
    
    agents = [
        ("CXR Analyzer", result.cxr_analysis),
        ("Finding Interpreter", result.finding_interpretation),
        ("Report Generator", result.report),
        ("Priority Router", result.priority_routing)
    ]
    
    for name, agent_result in agents:
        if agent_result:
            status_icon = "✅" if agent_result.status == "success" else "❌"
            lines.append(f"- {status_icon} **{name}:** {agent_result.processing_time_ms:.0f}ms")
    
    return "\n".join(lines)


def get_sample_image():
    """Return a sample X-ray image for demo purposes."""
    # Create a simple placeholder image
    img = Image.new('RGB', (512, 512), color=(20, 20, 30))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw a simple chest outline
    draw.ellipse([100, 150, 412, 450], outline=(60, 60, 70), width=3)
    draw.ellipse([150, 180, 280, 350], outline=(50, 50, 60), width=2)
    draw.ellipse([232, 180, 362, 350], outline=(50, 50, 60), width=2)
    
    # Add text
    draw.text((150, 50), "Sample CXR", fill=(80, 80, 90))
    draw.text((120, 470), "Upload real X-ray for analysis", fill=(80, 80, 90))
    
    return img


# ============================================
# GRADIO INTERFACE
# ============================================

# Custom CSS for professional styling
custom_css = """
/* Main container */
.gradio-container {
    max-width: 1400px !important;
    margin: auto;
}

/* Header styling */
.header-container {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    color: white;
}

/* Agent cards */
.agent-card {
    background: white;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 5px;
}

/* Priority badges */
.priority-stat { background: #ef4444; }
.priority-urgent { background: #f59e0b; }
.priority-routine { background: #22c55e; }

/* Code blocks */
pre {
    background: #1e293b !important;
    border-radius: 8px;
}

/* Tabs */
.tab-nav button {
    font-weight: 600 !important;
}
"""

# Create the Gradio interface
with gr.Blocks(
    title="RadioFlow - AI Radiology Workflow",
    css=custom_css
) as demo:
    
    # Header
    gr.HTML("""
    <div style="
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 25px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    ">
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="font-size: 40px;">🩻</div>
            <div>
                <h1 style="margin: 0; font-size: 28px; font-weight: 700;">RadioFlow</h1>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 16px;">
                    AI-Powered Radiology Workflow Agent | MedGemma Impact Challenge
                </p>
            </div>
        </div>
        <div style="
            display: flex; 
            gap: 20px; 
            margin-top: 15px; 
            padding-top: 15px; 
            border-top: 1px solid rgba(255,255,255,0.2);
        ">
            <div>
                <span style="opacity: 0.7;">Powered by</span>
                <strong>MedGemma + CXR Foundation</strong>
            </div>
            <div>
                <span style="opacity: 0.7;">Architecture</span>
                <strong>4-Agent Pipeline</strong>
            </div>
            <div>
                <span style="opacity: 0.7;">Prize Track</span>
                <strong>Main + Agentic Workflow</strong>
            </div>
        </div>
    </div>
    """)
    
    # Agent Pipeline Visualization
    gr.HTML("""
    <div style="
        background: #f8fafc; 
        padding: 15px 20px; 
        border-radius: 10px; 
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
    ">
        <div style="text-align: center; margin-bottom: 10px; font-weight: 600; color: #475569;">
            Multi-Agent Pipeline
        </div>
        <div style="display: flex; justify-content: center; align-items: center; gap: 10px; flex-wrap: wrap;">
            <div style="background: #3b82f6; color: white; padding: 10px 15px; border-radius: 8px; font-weight: 500;">
                1️⃣ CXR Analyzer
            </div>
            <div style="color: #94a3b8;">→</div>
            <div style="background: #8b5cf6; color: white; padding: 10px 15px; border-radius: 8px; font-weight: 500;">
                2️⃣ Finding Interpreter
            </div>
            <div style="color: #94a3b8;">→</div>
            <div style="background: #ec4899; color: white; padding: 10px 15px; border-radius: 8px; font-weight: 500;">
                3️⃣ Report Generator
            </div>
            <div style="color: #94a3b8;">→</div>
            <div style="background: #f59e0b; color: white; padding: 10px 15px; border-radius: 8px; font-weight: 500;">
                4️⃣ Priority Router
            </div>
        </div>
    </div>
    """)
    
    with gr.Row():
        # Left Column - Input
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            
            image_input = gr.Image(
                label="Chest X-Ray Image",
                type="pil",
                height=300
            )
            
            with gr.Accordion("Clinical Context (Optional)", open=False):
                clinical_history = gr.Textbox(
                    label="Clinical History",
                    placeholder="e.g., 65-year-old male with cough and fever for 3 days",
                    lines=2
                )
                patient_age = gr.Textbox(
                    label="Patient Age",
                    placeholder="e.g., 65"
                )
                symptoms = gr.Textbox(
                    label="Presenting Symptoms",
                    placeholder="e.g., Cough, fever, shortness of breath",
                    lines=2
                )
            
            with gr.Row():
                analyze_btn = gr.Button(
                    "🔬 Analyze X-Ray",
                    variant="primary",
                    size="lg"
                )
                clear_btn = gr.Button("🗑️ Clear", size="lg")
            
            status_display = gr.Textbox(
                label="Status",
                interactive=False,
                value="Ready to analyze"
            )
        
        # Right Column - Output
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Analysis Results")
            
            with gr.Tabs():
                with gr.Tab("📋 Report"):
                    priority_display = gr.HTML(label="Priority Assessment")
                    report_output = gr.Textbox(
                        label="Radiology Report",
                        lines=20,
                        max_lines=30,
                        interactive=False
                    )
                
                with gr.Tab("📈 Visualizations"):
                    with gr.Row():
                        workflow_plot = gr.Plot(label="Agent Pipeline Status")
                        priority_plot = gr.Plot(label="Priority Gauge")
                    radar_plot = gr.Plot(label="Analysis Confidence")
                
                with gr.Tab("🔍 Findings"):
                    findings_output = gr.Code(
                        label="Detected Findings (JSON)",
                        language="json",
                        lines=15
                    )
                
                with gr.Tab("⚡ Metrics"):
                    metrics_output = gr.Markdown()
    
    # Footer
    gr.HTML("""
    <div style="
        margin-top: 30px;
        padding: 20px;
        background: #f1f5f9;
        border-radius: 10px;
        text-align: center;
    ">
        <div style="font-weight: 600; margin-bottom: 10px;">
            🏆 MedGemma Impact Challenge Submission
        </div>
        <div style="color: #64748b; font-size: 14px;">
            Built with Google HAI-DEF: MedGemma + CXR Foundation | 
            Targeting: Main Track + Agentic Workflow Prize
        </div>
        <div style="margin-top: 10px; font-size: 12px; color: #94a3b8;">
            ⚠️ For demonstration purposes only. Not for clinical use.
            This AI system requires radiologist verification.
        </div>
    </div>
    """)
    
    # Event handlers
    analyze_btn.click(
        fn=process_xray,
        inputs=[image_input, clinical_history, patient_age, symptoms],
        outputs=[
            report_output,
            priority_display,
            findings_output,
            metrics_output,
            status_display,
            workflow_plot,
            radar_plot,
            priority_plot
        ]
    )
    
    clear_btn.click(
        fn=lambda: (None, "", "", "", "Ready to analyze", None, None, None, None),
        outputs=[
            image_input,
            report_output,
            priority_display,
            findings_output,
            status_display,
            workflow_plot,
            radar_plot,
            priority_plot
        ]
    )
    
    # Initialize on load
    demo.load(fn=initialize_system, outputs=[])


# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    # Initialize the system
    print("🚀 Starting RadioFlow...")
    initialize_system()
    
    # Launch the demo on port 7861
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
