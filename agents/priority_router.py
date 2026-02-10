"""
Agent 4: Priority Router
Uses MedGemma to assess urgency and route cases appropriately
"""

import time
from typing import Any, Dict, Optional, List

from .base_agent import BaseAgent, AgentResult

# Import the unified MedGemma engine
try:
    from .medgemma_engine import get_engine, MedGemmaEngine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False


class PriorityRouterAgent(BaseAgent):
    """
    Agent 4: MedGemma Priority Router
    
    Assesses case urgency and determines appropriate routing
    based on radiology report and findings using MedGemma.
    """
    
    # Priority level definitions
    PRIORITY_LEVELS = {
        "STAT": {
            "score_range": (0.8, 1.0),
            "color": "#ef4444",
            "description": "Critical finding requiring immediate attention",
            "response_time": "< 30 minutes",
            "actions": ["Page radiologist immediately", "Direct communication with ordering physician"]
        },
        "URGENT": {
            "score_range": (0.5, 0.8),
            "color": "#f59e0b",
            "description": "Significant finding requiring prompt review",
            "response_time": "< 4 hours",
            "actions": ["Prioritize in reading queue", "Flag for senior review"]
        },
        "ROUTINE": {
            "score_range": (0.0, 0.5),
            "color": "#22c55e",
            "description": "Standard workflow processing",
            "response_time": "< 24 hours",
            "actions": ["Standard reading queue", "Routine workflow"]
        }
    }
    
    # Critical findings that require immediate communication
    CRITICAL_FINDINGS = [
        "pneumothorax",
        "tension pneumothorax",
        "aortic dissection",
        "pulmonary embolism",
        "massive pleural effusion",
        "mediastinal mass",
        "severe cardiomegaly",
        "pulmonary edema"
    ]
    
    def __init__(self, demo_mode: bool = False):
        super().__init__(
            name="Priority Router",
            model_name="google/medgemma-4b-it"
        )
        self.demo_mode = demo_mode
        self.engine = None
    
    def load_model(self) -> bool:
        """Load MedGemma model via unified engine."""
        if self.demo_mode or not ENGINE_AVAILABLE:
            self.is_loaded = True
            return True
        
        try:
            self.engine = get_engine(force_demo=self.demo_mode)
            self.is_loaded = self.engine.is_loaded
            return True
        except Exception as e:
            print(f"Failed to load MedGemma engine: {e}")
            self.demo_mode = True
            self.is_loaded = True
            return True
    
    def process(self, input_data: Any, context: Optional[Dict] = None) -> AgentResult:
        """
        Assess priority and route the case.
        
        Args:
            input_data: Dictionary from Report Generator agent
            context: Additional context
        
        Returns:
            AgentResult with priority assessment and routing
        """
        start_time = time.time()
        
        if not isinstance(input_data, dict):
            return AgentResult(
                agent_name=self.name,
                status="error",
                data={},
                processing_time_ms=(time.time() - start_time) * 1000,
                error_message="Invalid input: expected dictionary from Report Generator"
            )
        
        # Extract report data
        report_sections = input_data.get("sections", {})
        full_report = input_data.get("full_report", "")
        findings_count = input_data.get("findings_count", 0)
        
        # Get original findings if passed through context
        original_findings = context.get("original_findings", []) if context else []
        
        # Process - always try to use real model if available
        if self.engine and self.engine.is_loaded and self.engine.backend != "demo":
            routing = self._run_model_inference(
                report_sections, full_report, findings_count, original_findings, context
            )
        else:
            routing = self._simulate_priority_assessment(
                report_sections, full_report, findings_count, original_findings, context
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return AgentResult(
            agent_name=self.name,
            status="success",
            data=routing,
            processing_time_ms=processing_time
        )
    
    def _run_model_inference(
        self,
        report_sections: Dict,
        full_report: str,
        findings_count: int,
        original_findings: List[Dict],
        context: Optional[Dict]
    ) -> Dict:
        """Use MedGemma to assess priority via unified engine."""
        try:
            prompt = self._build_priority_prompt(full_report, original_findings)
            
            # Use the unified engine to assess priority
            response = self.engine.generate(prompt, max_tokens=256)
            
            return self._parse_priority_response(response, original_findings)
            
        except Exception as e:
            print(f"Priority assessment error: {e}")
            return self._simulate_priority_assessment(
                report_sections, full_report, findings_count, original_findings, context
            )
    
    def _simulate_priority_assessment(
        self,
        report_sections: Dict,
        full_report: str,
        findings_count: int,
        original_findings: List[Dict],
        context: Optional[Dict]
    ) -> Dict:
        """Simulate priority assessment for demo."""
        time.sleep(0.3)  # Simulate processing
        
        # Calculate priority score based on findings
        priority_score = self._calculate_priority_score(original_findings)
        priority_level = self._get_priority_level(priority_score)
        
        # Check for critical findings
        critical_findings = self._check_critical_findings(original_findings, full_report)
        
        # Determine routing
        routing_recommendation = self._determine_routing(priority_level, critical_findings)
        
        # Generate action items
        action_items = self._generate_action_items(priority_level, critical_findings)
        
        # Communication requirements
        communication = self._determine_communication_requirements(priority_level, critical_findings)
        
        return {
            "priority_score": round(priority_score, 2),
            "priority_level": priority_level,
            "priority_details": self.PRIORITY_LEVELS[priority_level],
            "critical_findings_detected": critical_findings,
            "routing_recommendation": routing_recommendation,
            "action_items": action_items,
            "communication_requirements": communication,
            "estimated_response_time": self.PRIORITY_LEVELS[priority_level]["response_time"],
            "workflow_status": "routed",
            "model_used": f"{self.model_name} (demo mode)"
        }
    
    def _build_priority_prompt(self, full_report: str, original_findings: List[Dict]) -> str:
        """Build prompt for priority assessment."""
        findings_summary = "\n".join([
            f"- {f.get('type', 'Unknown')}: {f.get('severity', 'Unknown')} severity"
            for f in original_findings
        ])
        
        return f"""You are a clinical decision support system assessing radiology case priority.

**Radiology Report:**
{full_report[:1500]}  # Truncate for context length

**Detected Findings Summary:**
{findings_summary if findings_summary else "No significant findings"}

Based on this information, provide:
1. PRIORITY LEVEL: STAT, URGENT, or ROUTINE
2. PRIORITY SCORE: 0.0 to 1.0 (1.0 = most urgent)
3. CRITICAL FINDINGS: List any findings requiring immediate communication
4. RECOMMENDED ACTIONS: Specific next steps

Be conservative - err on the side of higher priority for concerning findings."""
    
    def _parse_priority_response(self, response: str, original_findings: List[Dict]) -> Dict:
        """Parse MedGemma response for priority information."""
        # Basic parsing - extract priority level and score
        priority_level = "ROUTINE"
        priority_score = 0.3
        
        response_lower = response.lower()
        if "stat" in response_lower:
            priority_level = "STAT"
            priority_score = 0.9
        elif "urgent" in response_lower:
            priority_level = "URGENT"
            priority_score = 0.65
        
        return {
            "priority_score": priority_score,
            "priority_level": priority_level,
            "priority_details": self.PRIORITY_LEVELS[priority_level],
            "critical_findings_detected": [],
            "routing_recommendation": self._determine_routing(priority_level, []),
            "action_items": self.PRIORITY_LEVELS[priority_level]["actions"],
            "model_response": response,
            "model_used": self.model_name
        }
    
    def _calculate_priority_score(self, findings: List[Dict]) -> float:
        """Calculate priority score based on findings."""
        if not findings:
            return 0.2  # Low baseline for normal studies
        
        severity_scores = {
            "critical": 1.0,
            "high": 0.8,
            "moderate": 0.5,
            "low": 0.3
        }
        
        # Get maximum severity
        max_score = 0.0
        for finding in findings:
            severity = finding.get("severity", "low")
            score = severity_scores.get(severity, 0.3)
            max_score = max(max_score, score)
        
        # Boost for multiple findings
        if len(findings) > 2:
            max_score = min(1.0, max_score + 0.1)
        
        return max_score
    
    def _get_priority_level(self, score: float) -> str:
        """Convert score to priority level."""
        for level, details in self.PRIORITY_LEVELS.items():
            min_score, max_score = details["score_range"]
            if min_score <= score <= max_score:
                return level
        return "ROUTINE"
    
    def _check_critical_findings(self, findings: List[Dict], report_text: str) -> List[str]:
        """Check for critical findings that require immediate communication."""
        detected_critical = []
        
        # Only check actual findings from the analysis, not report text
        # (Report text may contain "no pneumothorax" which would false-positive)
        for finding in findings:
            finding_type = finding.get("type", "").lower()
            severity = finding.get("severity", "").lower()
            
            # Only flag as critical if it's actually a critical finding type
            # AND has high/critical severity
            if finding_type in self.CRITICAL_FINDINGS and severity in ["critical", "high", "moderate"]:
                name = finding_type.replace("_", " ").title()
                if name not in detected_critical:
                    detected_critical.append(name)
        
        # Also check for specific high-severity findings
        for finding in findings:
            severity = finding.get("severity", "").lower()
            if severity == "critical":
                finding_type = finding.get("type", "Unknown").replace("_", " ").title()
                if finding_type not in detected_critical:
                    detected_critical.append(f"{finding_type} (Critical)")
        
        return detected_critical
    
    def _determine_routing(self, priority_level: str, critical_findings: List[str]) -> Dict:
        """Determine case routing based on priority."""
        routing = {
            "destination": "",
            "notification_list": [],
            "escalation_path": []
        }
        
        if priority_level == "STAT" or critical_findings:
            routing["destination"] = "STAT Reading Queue"
            routing["notification_list"] = [
                "On-call Radiologist",
                "Ordering Physician",
                "Nurse Station"
            ]
            routing["escalation_path"] = [
                "Attending Radiologist",
                "Department Chair"
            ]
        elif priority_level == "URGENT":
            routing["destination"] = "Priority Reading Queue"
            routing["notification_list"] = ["Assigned Radiologist"]
            routing["escalation_path"] = ["Senior Radiologist"]
        else:
            routing["destination"] = "Standard Reading Queue"
            routing["notification_list"] = []
            routing["escalation_path"] = []
        
        return routing
    
    def _generate_action_items(self, priority_level: str, critical_findings: List[str]) -> List[str]:
        """Generate specific action items."""
        actions = list(self.PRIORITY_LEVELS[priority_level]["actions"])
        
        if critical_findings:
            actions.insert(0, f"CRITICAL: Communicate findings immediately - {', '.join(critical_findings)}")
            actions.append("Document communication in medical record")
        
        return actions
    
    def _determine_communication_requirements(
        self,
        priority_level: str,
        critical_findings: List[str]
    ) -> Dict:
        """Determine communication requirements."""
        return {
            "immediate_notification_required": priority_level == "STAT" or len(critical_findings) > 0,
            "verbal_communication_required": len(critical_findings) > 0,
            "documentation_required": True,
            "critical_results_protocol": len(critical_findings) > 0,
            "recipients": self._determine_routing(priority_level, critical_findings)["notification_list"]
        }
