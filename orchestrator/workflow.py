"""
RadioFlow Orchestrator
Coordinates the multi-agent workflow for radiology analysis
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from PIL import Image

from agents import (
    CXRAnalyzerAgent,
    FindingInterpreterAgent,
    ReportGeneratorAgent,
    PriorityRouterAgent,
    BaseAgent,
    AgentResult
)
from utils.metrics import MetricsTracker


@dataclass
class WorkflowResult:
    """Complete result from the RadioFlow workflow"""
    workflow_id: str
    status: str  # "success", "partial", "error"
    start_time: str
    end_time: str
    total_duration_ms: float
    
    # Agent results
    cxr_analysis: Optional[AgentResult] = None
    finding_interpretation: Optional[AgentResult] = None
    report: Optional[AgentResult] = None
    priority_routing: Optional[AgentResult] = None
    
    # Aggregated outputs
    final_report: str = ""
    priority_level: str = "ROUTINE"
    priority_score: float = 0.0
    findings_count: int = 0
    critical_findings: List[str] = field(default_factory=list)
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "final_report": self.final_report,
            "priority_level": self.priority_level,
            "priority_score": self.priority_score,
            "findings_count": self.findings_count,
            "critical_findings": self.critical_findings,
            "agent_results": {
                "cxr_analysis": self.cxr_analysis.to_dict() if self.cxr_analysis else None,
                "finding_interpretation": self.finding_interpretation.to_dict() if self.finding_interpretation else None,
                "report": self.report.to_dict() if self.report else None,
                "priority_routing": self.priority_routing.to_dict() if self.priority_routing else None,
            },
            "errors": self.errors
        }


class RadioFlowOrchestrator:
    """
    Main orchestrator for the RadioFlow multi-agent system.
    
    Coordinates the sequential execution of:
    1. CXR Analyzer (Image Analysis)
    2. Finding Interpreter (Clinical Interpretation)
    3. Report Generator (Structured Report)
    4. Priority Router (Urgency Assessment)
    """
    
    def __init__(self, demo_mode: bool = True):
        """
        Initialize the orchestrator.
        
        Args:
            demo_mode: If True, agents use simulated outputs for faster demos
        """
        self.demo_mode = demo_mode
        self.metrics = MetricsTracker()
        
        # Initialize agents
        self.agents: Dict[str, BaseAgent] = {
            "cxr_analyzer": CXRAnalyzerAgent(demo_mode=demo_mode),
            "finding_interpreter": FindingInterpreterAgent(demo_mode=demo_mode),
            "report_generator": ReportGeneratorAgent(demo_mode=demo_mode),
            "priority_router": PriorityRouterAgent(demo_mode=demo_mode)
        }
        
        # Workflow state
        self._current_workflow_id: Optional[str] = None
        self._workflow_callbacks: List[Callable] = []
        
        # Agent order for pipeline
        self._agent_order = [
            "cxr_analyzer",
            "finding_interpreter", 
            "report_generator",
            "priority_router"
        ]
    
    def load_all_models(self) -> Dict[str, bool]:
        """Load all agent models. Returns dict of agent_name -> success."""
        results = {}
        for name, agent in self.agents.items():
            try:
                results[name] = agent.load_model()
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                results[name] = False
        return results
    
    def add_callback(self, callback: Callable[[str, AgentResult], None]):
        """Add a callback to be called after each agent completes."""
        self._workflow_callbacks.append(callback)
    
    def _notify_callbacks(self, agent_name: str, result: AgentResult):
        """Notify all callbacks of agent completion."""
        for callback in self._workflow_callbacks:
            try:
                callback(agent_name, result)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def process(
        self,
        image: Image.Image,
        clinical_context: Optional[Dict] = None,
        workflow_id: Optional[str] = None
    ) -> WorkflowResult:
        """
        Run the complete RadioFlow workflow.
        
        Args:
            image: Chest X-ray image (PIL Image)
            clinical_context: Optional clinical information
            workflow_id: Optional ID for tracking
        
        Returns:
            WorkflowResult with complete analysis
        """
        # Initialize workflow
        start_time = time.time()
        start_timestamp = datetime.now().isoformat()
        
        if workflow_id is None:
            workflow_id = f"rf_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self._current_workflow_id = workflow_id
        self.metrics.start_workflow(workflow_id)
        
        # Prepare context
        context = clinical_context or {}
        
        # Initialize result
        result = WorkflowResult(
            workflow_id=workflow_id,
            status="processing",
            start_time=start_timestamp,
            end_time="",
            total_duration_ms=0
        )
        
        errors = []
        
        try:
            # ============================================
            # STAGE 1: CXR Analysis
            # ============================================
            print(f"[{workflow_id}] Stage 1: CXR Analysis...")
            cxr_result = self.agents["cxr_analyzer"](image, context)
            result.cxr_analysis = cxr_result
            self.metrics.record_agent("CXR Analyzer", cxr_result.processing_time_ms, cxr_result.status == "success")
            self._notify_callbacks("cxr_analyzer", cxr_result)
            
            if cxr_result.status == "error":
                errors.append(f"CXR Analyzer: {cxr_result.error_message}")
            
            # ============================================
            # STAGE 2: Finding Interpretation
            # ============================================
            print(f"[{workflow_id}] Stage 2: Finding Interpretation...")
            interpretation_input = cxr_result.data if cxr_result.status == "success" else {}
            interpretation_result = self.agents["finding_interpreter"](interpretation_input, context)
            result.finding_interpretation = interpretation_result
            self.metrics.record_agent("Finding Interpreter", interpretation_result.processing_time_ms, interpretation_result.status == "success")
            self._notify_callbacks("finding_interpreter", interpretation_result)
            
            if interpretation_result.status == "error":
                errors.append(f"Finding Interpreter: {interpretation_result.error_message}")
            
            # ============================================
            # STAGE 3: Report Generation
            # ============================================
            print(f"[{workflow_id}] Stage 3: Report Generation...")
            report_input = interpretation_result.data if interpretation_result.status == "success" else {}
            report_result = self.agents["report_generator"](report_input, context)
            result.report = report_result
            self.metrics.record_agent("Report Generator", report_result.processing_time_ms, report_result.status == "success")
            self._notify_callbacks("report_generator", report_result)
            
            if report_result.status == "error":
                errors.append(f"Report Generator: {report_result.error_message}")
            
            # ============================================
            # STAGE 4: Priority Routing
            # ============================================
            print(f"[{workflow_id}] Stage 4: Priority Routing...")
            # Pass original findings through context for priority assessment
            priority_context = {
                **context,
                "original_findings": cxr_result.data.get("findings", []) if cxr_result.data else []
            }
            priority_input = report_result.data if report_result.status == "success" else {}
            priority_result = self.agents["priority_router"](priority_input, priority_context)
            result.priority_routing = priority_result
            self.metrics.record_agent("Priority Router", priority_result.processing_time_ms, priority_result.status == "success")
            self._notify_callbacks("priority_router", priority_result)
            
            if priority_result.status == "error":
                errors.append(f"Priority Router: {priority_result.error_message}")
            
            # ============================================
            # Aggregate Results
            # ============================================
            result.final_report = report_result.data.get("full_report", "") if report_result.data else ""
            result.priority_level = priority_result.data.get("priority_level", "ROUTINE") if priority_result.data else "ROUTINE"
            result.priority_score = priority_result.data.get("priority_score", 0.0) if priority_result.data else 0.0
            result.findings_count = len(cxr_result.data.get("findings", [])) if cxr_result.data else 0
            result.critical_findings = priority_result.data.get("critical_findings_detected", []) if priority_result.data else []
            
            # Determine overall status
            if not errors:
                result.status = "success"
            elif len(errors) < 4:
                result.status = "partial"
            else:
                result.status = "error"
            
            result.errors = errors
            
        except Exception as e:
            result.status = "error"
            result.errors = [str(e)]
            print(f"[{workflow_id}] Workflow error: {e}")
        
        finally:
            # Finalize timing
            end_time = time.time()
            result.end_time = datetime.now().isoformat()
            result.total_duration_ms = (end_time - start_time) * 1000
            
            # Record metrics
            self.metrics.end_workflow(
                findings_count=result.findings_count,
                priority_score=result.priority_score,
                status=result.status
            )
            
            print(f"[{workflow_id}] Workflow complete in {result.total_duration_ms:.0f}ms")
        
        return result
    
    def get_agent_statuses(self) -> Dict[str, Dict]:
        """Get status of all agents."""
        return {
            name: {
                "name": agent.name,
                "model": agent.model_name,
                "loaded": agent.is_loaded,
                "metrics": agent.get_metrics()
            }
            for name, agent in self.agents.items()
        }
    
    def get_workflow_metrics(self) -> str:
        """Get formatted workflow metrics."""
        return self.metrics.format_for_display()
    
    def reset(self):
        """Reset orchestrator state."""
        self._current_workflow_id = None
        for agent in self.agents.values():
            agent.reset_metrics()
        self.metrics = MetricsTracker()


def create_orchestrator(demo_mode: bool = True) -> RadioFlowOrchestrator:
    """Factory function to create an orchestrator instance."""
    orchestrator = RadioFlowOrchestrator(demo_mode=demo_mode)
    orchestrator.load_all_models()
    return orchestrator
