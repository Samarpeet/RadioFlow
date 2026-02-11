# %% [markdown]
# # 🩻 RadioFlow: AI-Powered Radiology Workflow Agent
# ## MedGemma Impact Challenge Submission
#
# **Author:** Samarpeet Garad | **Date:** February 2026
#
# **GitHub:** https://github.com/Samarpeet/RadioFlow

# %% [markdown]
# ## 1. Setup

# %%
import subprocess
import sys

# Install packages
print("📦 Installing packages...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "accelerate", "sentencepiece"],
    check=False,
)
print("✅ Done!")

# %%
import os
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from IPython.display import HTML, display
import plotly.graph_objects as go

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
GPU_AVAILABLE = torch.cuda.is_available()

# %% [markdown]
# ## 2. Authenticate with Hugging Face

# %%
try:
    from kaggle_secrets import UserSecretsClient

    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    login(token=HF_TOKEN)
    print("✅ Authenticated with HuggingFace")
except:
    print("⚠️ Add HF_TOKEN as Kaggle secret")

# %% [markdown]
# ## 3. Load MedGemma

# %%
MODEL_NAME = "google/medgemma-4b-it"
MODEL_LOADED = False
model = None
tokenizer = None

print(f"🔄 Loading {MODEL_NAME}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    MODEL_LOADED = True
    print("✅ MedGemma loaded!")
except Exception as e:
    print(f"❌ Failed: {e}")
    MODEL_LOADED = False


# %%
def generate_response(prompt, max_tokens=256):
    if not MODEL_LOADED:
        return "[Demo mode]"
    try:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        inputs = inputs.to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            outputs[0][inputs.shape[1] :], skip_special_tokens=True
        ).strip()
    except:
        return "[Error]"


if MODEL_LOADED:
    print("🧪 Testing...")
    print(generate_response("List 3 chest X-ray findings briefly.", 60)[:100])

# %% [markdown]
# ## 4. Agents


# %%
@dataclass
class AgentResult:
    agent_name: str
    status: str
    data: Dict[str, Any]
    processing_time_ms: float


class CXRAnalyzer:
    def __call__(self, image, context=None):
        start = time.time()
        arr = np.array(image.convert("L"))
        h, w = arr.shape
        stats = {
            "right": arr[:, w // 2 :].mean(),
            "left": arr[:, : w // 2].mean(),
            "lower": arr[h // 2 :, :].mean(),
            "overall": arr.mean(),
        }
        findings = []
        if stats["lower"] > stats["overall"] + 10:
            findings.append(
                {
                    "type": "opacity",
                    "region": "lower_lungs",
                    "severity": "moderate",
                    "description": "Increased lower lung density",
                }
            )
        if abs(stats["right"] - stats["left"]) > 12:
            findings.append(
                {
                    "type": "asymmetry",
                    "region": "hemithorax",
                    "severity": "mild",
                    "description": "Asymmetric lung density",
                }
            )
        if not findings:
            findings.append(
                {
                    "type": "normal",
                    "region": "lungs",
                    "severity": "none",
                    "description": "No abnormalities",
                }
            )
        return AgentResult(
            "CXR Analyzer",
            "success",
            {"findings": findings},
            (time.time() - start) * 1000,
        )


class FindingInterpreter:
    def __call__(self, data, context=None):
        start = time.time()
        interpreted = []
        for f in data.get("findings", []):
            if MODEL_LOADED and f["type"] != "normal":
                prompt = f"Interpret this X-ray finding briefly: {f['description']}"
                interp = generate_response(prompt, 80)
            else:
                interp = f["description"]
            interpreted.append({"original": f, "interpretation": interp})
        return AgentResult(
            "Interpreter",
            "success",
            {"interpreted": interpreted},
            (time.time() - start) * 1000,
        )


class ReportGenerator:
    def __call__(self, data, context=None):
        start = time.time()
        findings_text = "\n".join(
            [f"- {i['original']['description']}" for i in data.get("interpreted", [])]
        )
        history = (
            context.get("clinical_history", "Not provided")
            if context
            else "Not provided"
        )

        if MODEL_LOADED:
            prompt = f"Generate a brief chest X-ray report.\nHistory: {history}\nFindings:\n{findings_text}\n\nInclude: TECHNIQUE, FINDINGS, IMPRESSION"
            report = generate_response(prompt, 300)
        else:
            report = f"TECHNIQUE: PA chest X-ray\n\nFINDINGS:\n{findings_text}\n\nIMPRESSION: Clinical correlation recommended."

        full = f"\n{'=' * 60}\nCHEST RADIOGRAPH REPORT\n{'=' * 60}\n\n{report}\n\n{'=' * 60}\nGenerated by RadioFlow\n{'=' * 60}"
        return AgentResult(
            "Reporter", "success", {"report": full}, (time.time() - start) * 1000
        )


class PriorityRouter:
    def __call__(self, data, context=None):
        start = time.time()
        findings = context.get("findings", []) if context else []
        score = 0.2
        for f in findings:
            if f.get("severity") == "moderate":
                score = max(score, 0.5)
            if f.get("severity") == "high":
                score = max(score, 0.8)
        level = "STAT" if score >= 0.8 else "URGENT" if score >= 0.5 else "ROUTINE"
        return AgentResult(
            "Router",
            "success",
            {"level": level, "score": score},
            (time.time() - start) * 1000,
        )


print("✅ Agents ready")

# %% [markdown]
# ## 5. Orchestrator


# %%
class RadioFlowOrchestrator:
    def __init__(self):
        self.analyzer = CXRAnalyzer()
        self.interpreter = FindingInterpreter()
        self.reporter = ReportGenerator()
        self.router = PriorityRouter()

    def process(self, image, context=None):
        context = context or {}
        print("\n🔬 Stage 1: Analyzing...")
        r1 = self.analyzer(image, context)
        print(f"   Found {len(r1.data['findings'])} findings")

        print("📋 Stage 2: Interpreting...")
        r2 = self.interpreter(r1.data, context)

        print("📝 Stage 3: Generating report...")
        r3 = self.reporter(r2.data, context)

        print("🚦 Stage 4: Priority...")
        ctx = {**context, "findings": r1.data["findings"]}
        r4 = self.router(r3.data, ctx)
        print(f"   Priority: {r4.data['level']}")

        return {
            "report": r3.data["report"],
            "priority": r4.data["level"],
            "score": r4.data["score"],
            "findings_count": len(r1.data["findings"]),
            "agents": [r1, r2, r3, r4],
        }


orchestrator = RadioFlowOrchestrator()
print("🚀 RadioFlow ready!")

# %% [markdown]
# ## 6. Demo


# %%
def create_sample_xray(size=(512, 512)):
    np.random.seed(42)
    img = Image.new("L", size, 30)
    draw = ImageDraw.Draw(img)
    w, h = size
    draw.ellipse([50, 80, w // 2 - 20, h - 50], fill=20)
    draw.ellipse([w // 2 + 20, 80, w - 50, h - 50], fill=28)
    draw.ellipse([w // 3, h // 3, 2 * w // 3, 2 * h // 3], fill=75)
    arr = np.array(img)
    arr = np.clip(arr + np.random.normal(0, 4, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


sample_image = create_sample_xray()
plt.figure(figsize=(6, 6))
plt.imshow(sample_image, cmap="gray")
plt.title("Sample Chest X-Ray")
plt.axis("off")
plt.show()

# %%
result = orchestrator.process(
    sample_image, {"clinical_history": "65yo with cough and fever"}
)

# %% [markdown]
# ## 7. Results

# %%
print(result["report"])

# %%
colors = {"STAT": "#ef4444", "URGENT": "#f59e0b", "ROUTINE": "#22c55e"}
display(
    HTML(f"""
<div style="padding:20px; background:linear-gradient(135deg,#1e3a5f,#2d5a87); border-radius:12px; color:white;">
<h2>🚦 Priority: <span style="color:{colors[result["priority"]]}">{result["priority"]}</span></h2>
<p>Score: {result["score"]:.0%} | Findings: {result["findings_count"]}</p>
</div>
""")
)

# %%
metrics = pd.DataFrame(
    [
        {"Agent": a.agent_name, "Time (ms)": f"{a.processing_time_ms:.0f}"}
        for a in result["agents"]
    ]
)
display(metrics)

# %% [markdown]
# ## Summary
#
# ✅ **RadioFlow** - 4-agent radiology AI using MedGemma
#
# | Feature | Implementation |
# |---------|----------------|
# | MedGemma | Real inference |
# | Agents | 4-stage pipeline |
# | Reports | Structured format |
#
# **GitHub:** https://github.com/Samarpeet/RadioFlow

# %%
print("\n🏆 RadioFlow Complete!")
print(f"Model: {'MedGemma (REAL)' if MODEL_LOADED else 'Demo'}")
print(f"Priority: {result['priority']}")
print("GitHub: https://github.com/Samarpeet/RadioFlow")
