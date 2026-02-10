"""
Agent 2: Finding Interpreter
Uses MedGemma to interpret CXR findings into clinical language
"""

import time
from typing import Any, Dict, Optional, List
from PIL import Image

from .base_agent import BaseAgent, AgentResult

# Import the unified MedGemma engine
try:
    from .medgemma_engine import get_engine, MedGemmaEngine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False


class FindingInterpreterAgent(BaseAgent):
    """
    Agent 2: MedGemma Finding Interpreter
    
    Takes CXR analysis results and generates clinical interpretations
    using Google's MedGemma model via the unified engine.
    """
    
    def __init__(self, demo_mode: bool = False):
        super().__init__(
            name="Finding Interpreter",
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
        Interpret CXR findings.
        
        Args:
            input_data: Dictionary from CXR Analyzer agent
            context: Additional context (patient info, clinical history)
        
        Returns:
            AgentResult with interpreted findings
        """
        start_time = time.time()
        
        if not isinstance(input_data, dict):
            return AgentResult(
                agent_name=self.name,
                status="error",
                data={},
                processing_time_ms=(time.time() - start_time) * 1000,
                error_message="Invalid input: expected dictionary from CXR Analyzer"
            )
        
        # Extract findings from CXR analysis
        findings = input_data.get("findings", [])
        region_analysis = input_data.get("region_analysis", {})
        
        # Process - always try to use real model if available
        if self.engine and self.engine.is_loaded and self.engine.backend != "demo":
            interpretation = self._run_model_inference(findings, region_analysis, context)
        else:
            interpretation = self._simulate_interpretation(findings, region_analysis, context)
        
        processing_time = (time.time() - start_time) * 1000
        
        return AgentResult(
            agent_name=self.name,
            status="success",
            data=interpretation,
            processing_time_ms=processing_time
        )
    
    def _run_model_inference(
        self,
        findings: List[Dict],
        region_analysis: Dict,
        context: Optional[Dict]
    ) -> Dict:
        """Run actual MedGemma inference using the unified engine."""
        try:
            clinical_context = context.get("clinical_history", "Not provided") if context else "Not provided"
            
            # Generate interpretations for each finding using real MedGemma
            interpreted_findings = []
            for finding in findings:
                prompt = f"""As a radiologist, interpret this chest X-ray finding:

Finding: {finding.get('type', 'Unknown')}
Region: {finding.get('region', 'Unknown')}
Severity: {finding.get('severity', 'Unknown')}
Description: {finding.get('description', 'No description')}
Clinical History: {clinical_context}

Provide:
1. Clinical significance (1-2 sentences)
2. Top 3 differential diagnoses
3. Recommended follow-up

Be concise and clinically relevant."""

                response = self.engine.generate(prompt, max_tokens=200)
                
                interpreted = {
                    "original": finding,
                    "clinical_significance": self._extract_significance(response, finding),
                    "differential_diagnoses": self._get_differentials(finding),
                    "recommended_followup": self._get_followup(finding),
                    "medgemma_interpretation": response,
                    "correlation_notes": f"MedGemma analysis: {response[:100]}..."
                }
                interpreted_findings.append(interpreted)
            
            # Generate clinical summary
            clinical_summary = self._generate_clinical_summary(interpreted_findings, clinical_context)
            key_concerns = self._identify_key_concerns(interpreted_findings)
            
            return {
                "interpreted_findings": interpreted_findings,
                "clinical_summary": clinical_summary,
                "key_concerns": key_concerns,
                "abnormal_regions": [
                    region for region, data in region_analysis.items()
                    if data.get("status") == "abnormal"
                ],
                "confidence_level": "high",
                "model_used": f"MedGemma ({self.engine.backend})"
            }
            
        except Exception as e:
            print(f"MedGemma inference error: {e}")
            return self._simulate_interpretation(findings, region_analysis, context)
    
    def _extract_significance(self, response: str, finding: Dict) -> str:
        """Extract clinical significance from MedGemma response."""
        # Take first meaningful sentence
        sentences = response.split('.')
        if sentences and len(sentences[0]) > 10:
            return sentences[0].strip() + "."
        return self._get_significance(finding)
    
    def _simulate_interpretation(
        self,
        findings: List[Dict],
        region_analysis: Dict,
        context: Optional[Dict]
    ) -> Dict:
        """Simulate MedGemma interpretation for demo."""
        time.sleep(0.4)  # Simulate processing
        
        clinical_context = context.get("clinical_history", "Not provided") if context else "Not provided"
        
        # Generate detailed interpretations for each finding
        interpreted_findings = []
        for finding in findings:
            interpreted = {
                "original": finding,
                "clinical_significance": self._get_significance(finding),
                "differential_diagnoses": self._get_differentials(finding),
                "recommended_followup": self._get_followup(finding),
                "correlation_notes": self._get_correlation_notes(finding, clinical_context)
            }
            interpreted_findings.append(interpreted)
        
        # Generate overall clinical summary
        clinical_summary = self._generate_clinical_summary(interpreted_findings, clinical_context)
        
        # Identify key concerns
        key_concerns = self._identify_key_concerns(interpreted_findings)
        
        return {
            "interpreted_findings": interpreted_findings,
            "clinical_summary": clinical_summary,
            "key_concerns": key_concerns,
            "abnormal_regions": [
                region for region, data in region_analysis.items()
                if data.get("status") == "abnormal"
            ],
            "confidence_level": "high" if all(f.get("confidence", 0) > 0.8 for f in findings) else "moderate",
            "model_used": f"{self.model_name} (demo mode)"
        }
    
    def _build_prompt(self, findings: List[Dict], region_analysis: Dict, context: Optional[Dict]) -> str:
        """Build prompt for MedGemma."""
        findings_text = "\n".join([
            f"- {f.get('type', 'Unknown')}: {f.get('description', 'No description')} "
            f"(Confidence: {f.get('confidence', 0):.0%}, Region: {f.get('region', 'Unknown')})"
            for f in findings
        ])
        
        context_text = ""
        if context:
            context_text = f"""
Clinical History: {context.get('clinical_history', 'Not provided')}
Patient Age: {context.get('age', 'Not provided')}
Symptoms: {context.get('symptoms', 'Not provided')}
"""
        
        return f"""You are an expert radiologist interpreting chest X-ray findings.

**Detected Findings:**
{findings_text if findings_text else "No significant findings detected."}

**Clinical Context:**
{context_text if context_text else "No additional context provided."}

Please provide:
1. Clinical significance of each finding
2. Differential diagnoses to consider
3. Recommended follow-up actions
4. Any correlations with clinical history

Respond in clear, professional medical language."""
    
    def _parse_model_response(self, response: str, findings: List[Dict]) -> Dict:
        """Parse MedGemma response into structured format."""
        # Basic parsing - in production, use more sophisticated parsing
        return {
            "interpreted_findings": [
                {
                    "original": f,
                    "interpretation": response
                }
                for f in findings
            ],
            "clinical_summary": response,
            "key_concerns": [],
            "model_used": self.model_name
        }
    
    def _get_significance(self, finding: Dict) -> str:
        """Get clinical significance for a finding."""
        significance_map = {
            "opacity": "Pulmonary opacity may indicate infection, inflammation, atelectasis, or malignancy. Clinical correlation required.",
            "consolidation": "Consolidation suggests airspace disease, commonly pneumonia. Consider infectious vs non-infectious etiologies.",
            "cardiomegaly": "Cardiac enlargement may indicate heart failure, cardiomyopathy, or pericardial effusion.",
            "pleural_effusion": "Pleural fluid collection can be transudative or exudative. Consider cardiac, infectious, or malignant causes.",
            "pneumothorax": "Air in the pleural space requires urgent evaluation. Assess for tension physiology.",
            "nodule": "Pulmonary nodules require characterization and may need follow-up imaging or biopsy.",
            "mass": "Pulmonary masses are concerning for malignancy and require further workup."
        }
        finding_type = finding.get("type", "").lower()
        return significance_map.get(finding_type, "Clinical correlation recommended for proper interpretation.")
    
    def _get_differentials(self, finding: Dict) -> List[str]:
        """Get differential diagnoses for a finding."""
        differential_map = {
            "opacity": ["Pneumonia", "Atelectasis", "Pulmonary edema", "Aspiration", "Lung cancer"],
            "consolidation": ["Bacterial pneumonia", "Viral pneumonia", "Aspiration pneumonitis", "Organizing pneumonia"],
            "cardiomegaly": ["Heart failure", "Dilated cardiomyopathy", "Pericardial effusion", "Valvular disease"],
            "pleural_effusion": ["Heart failure", "Parapneumonic effusion", "Malignancy", "Liver disease"],
            "pneumothorax": ["Spontaneous", "Traumatic", "Iatrogenic", "COPD-related"],
            "nodule": ["Granuloma", "Primary lung cancer", "Metastasis", "Hamartoma"],
            "mass": ["Primary lung cancer", "Metastatic disease", "Lymphoma"]
        }
        finding_type = finding.get("type", "").lower()
        return differential_map.get(finding_type, ["Clinical correlation needed"])
    
    def _get_followup(self, finding: Dict) -> str:
        """Get recommended follow-up for a finding."""
        severity = finding.get("severity", "low")
        finding_type = finding.get("type", "").lower()
        
        if severity in ["critical", "high"]:
            return "Urgent clinical evaluation recommended. Consider emergent imaging or intervention."
        elif finding_type in ["nodule", "mass"]:
            return "Consider CT chest for further characterization. Follow Fleischner criteria if nodule."
        elif finding_type in ["pleural_effusion", "cardiomegaly"]:
            return "Correlate with clinical findings. Consider echocardiogram if cardiac etiology suspected."
        else:
            return "Follow-up imaging may be warranted based on clinical course."
    
    def _get_correlation_notes(self, finding: Dict, clinical_context: str) -> str:
        """Generate correlation notes with clinical context."""
        return f"Finding should be correlated with presenting symptoms and clinical history."
    
    def _generate_clinical_summary(self, interpreted_findings: List[Dict], clinical_context: str) -> str:
        """Generate overall clinical summary."""
        if not interpreted_findings:
            return "No significant abnormalities identified. Normal chest radiograph appearance."
        
        finding_count = len(interpreted_findings)
        severities = [f["original"].get("severity", "low") for f in interpreted_findings]
        
        if "critical" in severities or "high" in severities:
            urgency = "requiring prompt attention"
        elif "moderate" in severities:
            urgency = "warranting clinical correlation"
        else:
            urgency = "of uncertain clinical significance"
        
        return f"Chest radiograph demonstrates {finding_count} finding(s) {urgency}. Clinical correlation with patient presentation is recommended."
    
    def _identify_key_concerns(self, interpreted_findings: List[Dict]) -> List[str]:
        """Identify key clinical concerns from findings."""
        concerns = []
        
        for finding in interpreted_findings:
            original = finding["original"]
            severity = original.get("severity", "low")
            finding_type = original.get("type", "")
            
            if severity in ["critical", "high"]:
                concerns.append(f"Significant {finding_type} requiring attention")
            elif finding_type in ["mass", "nodule"]:
                concerns.append(f"Pulmonary {finding_type} requires characterization")
        
        return concerns if concerns else ["No immediate concerns identified"]
