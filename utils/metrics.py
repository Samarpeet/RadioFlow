"""
RadioFlow Metrics Tracking
Performance monitoring and analytics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import json


@dataclass
class WorkflowMetrics:
    """Metrics for a single workflow execution"""
    workflow_id: str
    start_time: str
    end_time: Optional[str] = None
    total_duration_ms: float = 0
    agent_durations: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    findings_count: int = 0
    priority_score: float = 0
    error_count: int = 0


class MetricsTracker:
    """
    Track and analyze RadioFlow performance metrics.
    Useful for demo and competition presentation.
    """
    
    def __init__(self):
        self.workflows: List[WorkflowMetrics] = []
        self.current_workflow: Optional[WorkflowMetrics] = None
        self._start_time: Optional[float] = None
    
    def start_workflow(self, workflow_id: Optional[str] = None) -> str:
        """Start tracking a new workflow."""
        if workflow_id is None:
            workflow_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_workflow = WorkflowMetrics(
            workflow_id=workflow_id,
            start_time=datetime.now().isoformat()
        )
        self._start_time = time.time()
        
        return workflow_id
    
    def record_agent(self, agent_name: str, duration_ms: float, success: bool = True):
        """Record an agent's execution."""
        if self.current_workflow:
            self.current_workflow.agent_durations[agent_name] = duration_ms
            if not success:
                self.current_workflow.error_count += 1
    
    def end_workflow(
        self,
        findings_count: int = 0,
        priority_score: float = 0,
        status: str = "success"
    ):
        """Complete the current workflow tracking."""
        if self.current_workflow and self._start_time:
            self.current_workflow.end_time = datetime.now().isoformat()
            self.current_workflow.total_duration_ms = (time.time() - self._start_time) * 1000
            self.current_workflow.findings_count = findings_count
            self.current_workflow.priority_score = priority_score
            self.current_workflow.status = status
            
            self.workflows.append(self.current_workflow)
            self.current_workflow = None
            self._start_time = None
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all workflows."""
        if not self.workflows:
            return {
                "total_workflows": 0,
                "avg_duration_ms": 0,
                "success_rate": 0,
                "avg_findings": 0,
                "agent_avg_times": {}
            }
        
        total = len(self.workflows)
        successful = sum(1 for w in self.workflows if w.status == "success")
        
        # Calculate agent average times
        agent_times: Dict[str, List[float]] = {}
        for workflow in self.workflows:
            for agent, duration in workflow.agent_durations.items():
                if agent not in agent_times:
                    agent_times[agent] = []
                agent_times[agent].append(duration)
        
        agent_avg = {
            agent: sum(times) / len(times)
            for agent, times in agent_times.items()
        }
        
        return {
            "total_workflows": total,
            "avg_duration_ms": sum(w.total_duration_ms for w in self.workflows) / total,
            "success_rate": successful / total * 100,
            "avg_findings": sum(w.findings_count for w in self.workflows) / total,
            "avg_priority": sum(w.priority_score for w in self.workflows) / total,
            "agent_avg_times": agent_avg
        }
    
    def get_latest_workflow(self) -> Optional[WorkflowMetrics]:
        """Get the most recent completed workflow."""
        return self.workflows[-1] if self.workflows else None
    
    def export_metrics(self) -> str:
        """Export all metrics as JSON."""
        data = {
            "summary": self.get_summary_stats(),
            "workflows": [
                {
                    "workflow_id": w.workflow_id,
                    "start_time": w.start_time,
                    "end_time": w.end_time,
                    "total_duration_ms": w.total_duration_ms,
                    "agent_durations": w.agent_durations,
                    "status": w.status,
                    "findings_count": w.findings_count,
                    "priority_score": w.priority_score
                }
                for w in self.workflows
            ]
        }
        return json.dumps(data, indent=2)
    
    def format_for_display(self) -> str:
        """Format metrics for UI display."""
        stats = self.get_summary_stats()
        
        lines = [
            "📊 **RadioFlow Performance Metrics**",
            "",
            f"**Total Analyses:** {stats['total_workflows']}",
            f"**Success Rate:** {stats['success_rate']:.1f}%",
            f"**Avg Processing Time:** {stats['avg_duration_ms']:.0f}ms",
            f"**Avg Findings per Study:** {stats['avg_findings']:.1f}",
            "",
            "**Agent Performance:**"
        ]
        
        for agent, avg_time in stats.get('agent_avg_times', {}).items():
            lines.append(f"  • {agent}: {avg_time:.0f}ms avg")
        
        return "\n".join(lines)
