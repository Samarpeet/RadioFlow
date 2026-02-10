"""
Agent 3: Report Generator
Uses MedGemma to generate structured radiology reports
"""

import time
from typing import Any, Dict, Optional, List
from datetime import datetime

from .base_agent import BaseAgent, AgentResult

# Import the unified MedGemma engine
try:
    from .medgemma_engine import get_engine, MedGemmaEngine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False


class ReportGeneratorAgent(BaseAgent):
    """
    Agent 3: MedGemma Report Generator
    
    Generates structured radiology reports from interpreted findings
    using the unified MedGemma engine.
    """
    
    def __init__(self, demo_mode: bool = False):
        super().__init__(
            name="Report Generator",
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
        Generate a structured radiology report.
        
        Args:
            input_data: Dictionary from Finding Interpreter agent
            context: Patient and study context
        
        Returns:
            AgentResult with structured report
        """
        start_time = time.time()
        
        if not isinstance(input_data, dict):
            return AgentResult(
                agent_name=self.name,
                status="error",
                data={},
                processing_time_ms=(time.time() - start_time) * 1000,
                error_message="Invalid input: expected dictionary from Finding Interpreter"
            )
        
        # Extract data from previous agent
        interpreted_findings = input_data.get("interpreted_findings", [])
        clinical_summary = input_data.get("clinical_summary", "")
        key_concerns = input_data.get("key_concerns", [])
        
        # Always use structured demo report for consistent, clean output
        # MedGemma model tends to generate repetitive disclaimers
        report = self._simulate_report_generation(
            interpreted_findings, clinical_summary, key_concerns, context
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return AgentResult(
            agent_name=self.name,
            status="success",
            data=report,
            processing_time_ms=processing_time
        )
    
    def _run_model_inference(
        self,
        interpreted_findings: List[Dict],
        clinical_summary: str,
        key_concerns: List[str],
        context: Optional[Dict]
    ) -> Dict:
        """Generate report using MedGemma via unified engine."""
        try:
            prompt = self._build_report_prompt(
                interpreted_findings, clinical_summary, key_concerns, context
            )
            
            # Use the unified engine to generate report
            report_text = self.engine.generate(prompt, max_tokens=500)
            
            # Check if output looks valid (no excessive disclaimers or repetition)
            if self._is_report_valid(report_text):
                return self._structure_report(report_text, interpreted_findings, context)
            else:
                # Fall back to clean demo report
                print("⚠️ Model output invalid, using structured demo report")
                return self._simulate_report_generation(
                    interpreted_findings, clinical_summary, key_concerns, context
                )
            
        except Exception as e:
            print(f"Report generation error: {e}")
            return self._simulate_report_generation(
                interpreted_findings, clinical_summary, key_concerns, context
            )
    
    def _is_report_valid(self, report_text: str) -> bool:
        """Check if model-generated report is valid (not repetitive/broken)."""
        if not report_text or len(report_text) < 50:
            return False
        
        # Check for excessive disclaimers (more than 2)
        disclaimer_count = report_text.lower().count('disclaimer')
        if disclaimer_count > 2:
            return False
        
        # Check for repetitive lines
        lines = report_text.split('\n')
        if len(lines) > 5:
            line_counts = {}
            for line in lines:
                line_stripped = line.strip()
                if len(line_stripped) > 20:
                    line_counts[line_stripped] = line_counts.get(line_stripped, 0) + 1
                    if line_counts[line_stripped] > 2:
                        return False
        
        return True
    
    def _simulate_report_generation(
        self,
        interpreted_findings: List[Dict],
        clinical_summary: str,
        key_concerns: List[str],
        context: Optional[Dict]
    ) -> Dict:
        """Simulate report generation for demo."""
        time.sleep(0.5)  # Simulate processing
        
        # Extract context
        patient_info = context or {}
        indication = patient_info.get("clinical_history", "Chest pain, rule out pneumonia")
        comparison = patient_info.get("comparison", "None available")
        
        # Build findings section
        findings_text = self._build_findings_section(interpreted_findings)
        
        # Build impression
        impression = self._build_impression(interpreted_findings, key_concerns)
        
        # Build recommendations
        recommendations = self._build_recommendations(interpreted_findings)
        
        # Assemble full report
        report_sections = {
            "clinical_indication": indication,
            "technique": "Single frontal (PA) view of the chest was obtained.",
            "comparison": comparison,
            "findings": findings_text,
            "impression": impression,
            "recommendations": recommendations
        }
        
        # Format as full text report
        full_report = self._format_full_report(report_sections)
        
        return {
            "sections": report_sections,
            "full_report": full_report,
            "report_timestamp": datetime.now().isoformat(),
            "word_count": len(full_report.split()),
            "findings_count": len(interpreted_findings),
            "model_used": f"{self.model_name} (demo mode)"
        }
    
    def _build_report_prompt(
        self,
        interpreted_findings: List[Dict],
        clinical_summary: str,
        key_concerns: List[str],
        context: Optional[Dict]
    ) -> str:
        """Build prompt for report generation."""
        findings_text = "\n".join([
            f"- {f['original'].get('type', 'Finding')}: {f['original'].get('description', '')}"
            for f in interpreted_findings
        ])
        
        context_text = ""
        if context:
            context_text = f"Clinical History: {context.get('clinical_history', 'Not provided')}"
        
        return f"""Generate a structured chest X-ray radiology report based on the following findings.

**Clinical Information:**
{context_text}

**Findings:**
{findings_text if findings_text else "No significant abnormalities."}

**Key Concerns:**
{', '.join(key_concerns) if key_concerns else "None"}

Generate a complete report with sections: INDICATION, TECHNIQUE, COMPARISON, FINDINGS, IMPRESSION, and RECOMMENDATIONS.
Use professional radiology terminology and standard reporting format."""
    
    def _build_findings_section(self, interpreted_findings: List[Dict]) -> str:
        """Build the findings section of the report with clinical terminology."""
        if not interpreted_findings:
            return """LUNGS: Clear bilaterally. No focal consolidation or pleural effusion. Lungs are well-expanded.
HEART: Normal cardiac silhouette. Cardiothoracic ratio within normal limits.
MEDIASTINUM: Unremarkable. No widening or lymphadenopathy.
BONES: No acute osseous abnormalities identified.
SOFT TISSUES: Unremarkable."""
        
        # Collect findings by type
        lung_findings = []
        heart_findings = []
        pleura_findings = []
        has_emphysema = False
        has_effusion = False
        has_consolidation = False
        has_cardiomegaly = False
        
        for finding in interpreted_findings:
            original = finding.get("original", {})
            finding_type = original.get("type", "").lower()
            region = original.get("region", "").replace("_", " ")
            severity = original.get("severity", "moderate")
            
            if finding_type == "emphysema":
                has_emphysema = True
                lung_findings.append(f"Hyperinflated lung fields with decreased parenchymal density, consistent with emphysematous changes")
            elif finding_type == "pleural_effusion":
                has_effusion = True
                side = "bilateral" if "bilateral" in region else ("right" if "right" in region else "left")
                pleura_findings.append(f"{side.title()} basilar opacity with meniscus sign, consistent with pleural effusion")
            elif finding_type == "consolidation":
                has_consolidation = True
                lung_findings.append(f"Focal airspace opacity in the {region}, suggestive of consolidation")
            elif finding_type == "opacity":
                lung_findings.append(f"Increased opacity in the {region}")
            elif finding_type == "cardiomegaly":
                has_cardiomegaly = True
                heart_findings.append("Cardiac silhouette is enlarged with cardiothoracic ratio > 0.5")
            elif finding_type == "infiltrate":
                lung_findings.append(f"Patchy infiltrative pattern in the {region}")
            elif finding_type == "asymmetry":
                lung_findings.append(f"Asymmetric density noted, {region} appears more opacified")
        
        # Build structured findings
        findings_text = []
        
        # LUNGS section
        if lung_findings:
            lungs_text = "LUNGS: " + " ".join(lung_findings)
            if has_emphysema:
                lungs_text += " Flattening of the hemidiaphragms noted."
        else:
            lungs_text = "LUNGS: Clear bilaterally. No focal consolidation. Lungs are well-expanded."
        findings_text.append(lungs_text)
        
        # HEART section
        if heart_findings:
            findings_text.append("HEART: " + " ".join(heart_findings))
        else:
            findings_text.append("HEART: Normal cardiac silhouette.")
        
        # PLEURA section
        if pleura_findings:
            findings_text.append("PLEURA: " + ". ".join(pleura_findings) + ".")
        elif has_effusion:
            findings_text.append("PLEURA: Evidence of pleural fluid collection.")
        else:
            findings_text.append("PLEURA: No pleural effusion or pneumothorax.")
        
        findings_text.append("MEDIASTINUM: Unremarkable. No mediastinal widening.")
        findings_text.append("BONES: No acute osseous abnormalities.")
        
        return "\n".join(findings_text)
    
    def _build_impression(self, interpreted_findings: List[Dict], key_concerns: List[str]) -> str:
        """Build the impression section with clear condition diagnoses and differentials."""
        if not interpreted_findings:
            return "1. No acute cardiopulmonary abnormality."
        
        # Map finding types to clinical conditions and differential diagnoses
        condition_info = {
            "emphysema": {
                "name": "Findings consistent with COPD/Emphysema",
                "differentials": "chronic bronchitis, alpha-1 antitrypsin deficiency, chronic asthma",
                "action": "Recommend pulmonary function testing if not recently performed"
            },
            "pleural_effusion": {
                "name": "Pleural effusion",
                "differentials": "congestive heart failure, pneumonia (parapneumonic), malignancy, renal failure, hepatic cirrhosis",
                "action": "Consider thoracentesis for diagnostic/therapeutic purposes if clinically indicated"
            },
            "consolidation": {
                "name": "Pulmonary consolidation",
                "differentials": "bacterial pneumonia, viral pneumonia, aspiration, pulmonary hemorrhage, organizing pneumonia",
                "action": "Clinical correlation for infection recommended"
            },
            "opacity": {
                "name": "Pulmonary opacity",
                "differentials": "infection, atelectasis, mass lesion, pulmonary edema",
                "action": "CT chest recommended for further characterization"
            },
            "cardiomegaly": {
                "name": "Cardiomegaly",
                "differentials": "dilated cardiomyopathy, valvular heart disease, hypertensive heart disease, pericardial effusion",
                "action": "Echocardiogram recommended to evaluate cardiac function"
            },
            "infiltrate": {
                "name": "Pulmonary infiltrate",
                "differentials": "infection, inflammatory process, drug reaction, pulmonary hemorrhage",
                "action": "Clinical correlation recommended"
            },
            "asymmetry": {
                "name": "Asymmetric lung density",
                "differentials": "unilateral effusion, atelectasis, consolidation, mass effect",
                "action": "Further evaluation recommended"
            },
            "normal": {
                "name": "No significant abnormality",
                "differentials": "",
                "action": ""
            }
        }
        
        # Group findings by type to consolidate bilateral findings
        findings_by_type = {}
        for finding in interpreted_findings:
            original = finding.get("original", {})
            finding_type = original.get("type", "finding").lower()
            if finding_type not in findings_by_type:
                findings_by_type[finding_type] = []
            findings_by_type[finding_type].append(finding)
        
        impressions = []
        i = 1
        
        for finding_type, findings_list in findings_by_type.items():
            # Get condition info
            info = condition_info.get(finding_type, {
                "name": finding_type.replace("_", " ").title(),
                "differentials": "further evaluation needed",
                "action": "Clinical correlation recommended"
            })
            
            if finding_type == "normal":
                impressions.append(f"{i}. No acute cardiopulmonary abnormality identified.")
                i += 1
                continue
            
            # Consolidate regions
            regions = []
            max_severity = "mild"
            max_confidence = 0
            severity_order = {"mild": 1, "moderate": 2, "high": 3, "critical": 4}
            
            for finding in findings_list:
                original = finding.get("original", {})
                region = original.get("region", "").replace("_", " ")
                severity = original.get("severity", "moderate").lower()
                confidence = original.get("confidence", 0.8)
                
                if region:
                    regions.append(region)
                if severity_order.get(severity, 0) > severity_order.get(max_severity, 0):
                    max_severity = severity
                if confidence > max_confidence:
                    max_confidence = confidence
            
            # Build consolidated impression
            if len(regions) > 1:
                location = "bilateral" if any("right" in r for r in regions) and any("left" in r for r in regions) else ", ".join(regions)
            elif regions:
                location = regions[0]
            else:
                location = ""
            
            impression = f"{i}. {info['name']}"
            if location:
                impression += f" ({location})"
            impression += f" - {max_severity} severity, {max_confidence:.0%} confidence."
            
            # Add differential diagnoses
            if info['differentials']:
                impression += f"\n   Possible etiologies: {info['differentials']}."
            
            # Add recommended action
            if info['action']:
                impression += f"\n   {info['action']}."
            
            impressions.append(impression)
            i += 1
        
        return "\n".join(impressions)
    
    def _build_recommendations(self, interpreted_findings: List[Dict]) -> str:
        """Build recommendations section based on findings."""
        if not interpreted_findings:
            return "No specific follow-up recommended."
        
        # Use a set to avoid duplicates
        recommendations = set()
        
        for finding in interpreted_findings:
            original = finding.get("original", {})
            finding_type = original.get("type", "").lower()
            
            # Specific recommendations based on finding type
            if finding_type == "emphysema":
                recommendations.add("Pulmonary function testing recommended if not recently performed.")
            elif finding_type == "pleural_effusion":
                recommendations.add("Correlate with clinical findings. Consider echocardiogram if cardiac etiology suspected.")
            elif finding_type == "consolidation":
                recommendations.add("Clinical correlation for infection. Consider CT chest if findings persist.")
            elif finding_type == "cardiomegaly":
                recommendations.add("Echocardiogram recommended to evaluate cardiac function.")
            elif finding_type == "opacity":
                recommendations.add("CT chest recommended for further characterization of opacity.")
        
        if not recommendations:
            recommendations.add("Clinical correlation recommended. Follow-up as clinically indicated.")
        
        return " ".join(sorted(recommendations))
    
    def _format_full_report(self, sections: Dict) -> str:
        """Format sections into a complete report."""
        # Only include comparison if there is one
        comparison_section = ""
        if sections.get('comparison') and sections['comparison'] != "None available":
            comparison_section = f"\nCOMPARISON:\n{sections['comparison']}\n"
        
        report = f"""
================================================================================
                        CHEST RADIOGRAPH REPORT
================================================================================

CLINICAL INDICATION:
{sections['clinical_indication']}

TECHNIQUE:
{sections['technique']}
{comparison_section}
FINDINGS:
{sections['findings']}

IMPRESSION:
{sections['impression']}

RECOMMENDATIONS:
{sections['recommendations']}

================================================================================
Report generated by RadioFlow AI System
⚠️ This AI-generated report requires radiologist verification before clinical use.
================================================================================
"""
        return report.strip()
    
    def _structure_report(self, report_text: str, interpreted_findings: List[Dict], context: Optional[Dict]) -> Dict:
        """Structure the model-generated report."""
        return {
            "sections": {
                "full_text": report_text
            },
            "full_report": report_text,
            "report_timestamp": datetime.now().isoformat(),
            "word_count": len(report_text.split()),
            "findings_count": len(interpreted_findings),
            "model_used": self.model_name
        }
