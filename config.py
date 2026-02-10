"""
RadioFlow Configuration
Model settings and environment configuration
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for HAI-DEF models"""
    
    # CXR Foundation Model
    CXR_MODEL_ID: str = "google/cxr-foundation"
    
    # MedGemma Models
    MEDGEMMA_MODEL_ID: str = "google/medgemma-4b-it"
    MEDGEMMA_MULTIMODAL_ID: str = "google/medgemma-4b-it"  # Same model handles both
    
    # Model loading settings
    DEVICE: str = "auto"  # "cuda", "cpu", or "auto"
    TORCH_DTYPE: str = "bfloat16"  # "float16", "bfloat16", or "float32"
    LOW_MEMORY_MODE: bool = True  # Use 4-bit quantization if needed
    
    # Inference settings
    MAX_NEW_TOKENS: int = 1024
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9
    DO_SAMPLE: bool = True


@dataclass
class AppConfig:
    """Application configuration"""
    
    # App settings
    APP_TITLE: str = "RadioFlow: AI-Powered Radiology Workflow Agent"
    APP_DESCRIPTION: str = "Multi-agent system for chest X-ray analysis using HAI-DEF models"
    
    # Priority thresholds
    CRITICAL_THRESHOLD: float = 0.8
    HIGH_THRESHOLD: float = 0.6
    MODERATE_THRESHOLD: float = 0.4
    
    # Workflow settings
    ENABLE_CACHING: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Demo mode (uses simulated outputs for faster demos)
    DEMO_MODE: bool = False
    
    # HuggingFace settings
    HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")


# Global configuration instances
MODEL_CONFIG = ModelConfig()
APP_CONFIG = AppConfig()


# Prompt templates for agents
PROMPTS = {
    "finding_interpreter": """You are a radiologist AI assistant analyzing chest X-ray findings.

Given the following image analysis results from a CXR Foundation model, provide a detailed clinical interpretation:

**Image Analysis Results:**
{cxr_analysis}

**Clinical Context:**
{clinical_context}

Please provide:
1. A summary of detected abnormalities
2. Clinical significance of each finding
3. Differential diagnoses to consider
4. Any areas of concern

Format your response in clear, professional medical language suitable for a radiology report.""",

    "report_generator": """You are an expert radiologist generating a structured radiology report.

Based on the following clinical findings, generate a complete chest X-ray report:

**Clinical Findings:**
{findings}

**Patient Context:**
{patient_context}

Generate a structured report with:
1. CLINICAL INDICATION
2. TECHNIQUE
3. COMPARISON (if available)
4. FINDINGS (detailed, organized by anatomical region)
5. IMPRESSION (numbered summary of key findings)
6. RECOMMENDATIONS

Use standard radiology reporting conventions and professional terminology.""",

    "priority_router": """You are a clinical decision support AI assessing radiology case priority.

Based on the following radiology report and findings, determine the urgency and appropriate routing:

**Radiology Report:**
{report}

**Key Findings:**
{findings}

Assess and provide:
1. PRIORITY LEVEL: [STAT/URGENT/ROUTINE] with justification
2. PRIORITY SCORE: A number from 0.0 to 1.0 (1.0 = most urgent)
3. RECOMMENDED ACTIONS: Immediate steps if any
4. ROUTING: Which department/specialist should be notified
5. CRITICAL FINDINGS: Any findings requiring immediate communication

Be specific about time-sensitive conditions that require immediate attention."""
}


# Finding categories for visualization
FINDING_CATEGORIES = [
    "Opacity/Consolidation",
    "Nodule/Mass",
    "Cardiomegaly",
    "Pleural Effusion",
    "Pneumothorax",
    "Atelectasis",
    "Emphysema",
    "Fracture",
    "Medical Devices",
    "Other"
]

# Priority level mapping
PRIORITY_LEVELS = {
    "STAT": {"score_range": (0.8, 1.0), "color": "#ef4444", "description": "Immediate attention required"},
    "URGENT": {"score_range": (0.5, 0.8), "color": "#f59e0b", "description": "Review within hours"},
    "ROUTINE": {"score_range": (0.0, 0.5), "color": "#22c55e", "description": "Standard workflow"}
}
