---
title: RadioFlow - AI Radiology Workflow Agent
emoji: 🩻
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.19.0
app_file: app.py
pinned: true
license: cc-by-4.0
tags:
  - medical
  - radiology
  - chest-x-ray
  - medgemma
  - hai-def
  - healthcare
  - agentic
  - multi-agent
---

# RadioFlow: AI-Powered Radiology Workflow Agent

> **MedGemma Impact Challenge Submission**  
> Targeting: Main Track + Agentic Workflow Prize

## Overview

RadioFlow is a multi-agent AI system that streamlines radiology workflows by processing chest X-rays through an orchestrated pipeline of specialized agents. Built with Google's Health AI Developer Foundations (HAI-DEF) models.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RADIOFLOW ORCHESTRATOR                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Agent 1    │───▶│   Agent 2    │───▶│   Agent 3    │       │
│  │ CXR Analyzer │    │   Finding    │    │   Report     │       │
│  │              │    │ Interpreter  │    │  Generator   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                                        │               │
│         │                                        ▼               │
│         │                              ┌──────────────┐          │
│         │                              │   Agent 4    │          │
│         └─────────────────────────────▶│   Priority   │          │
│                                        │    Router    │          │
│                                        └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Agents

| Agent | Model | Function |
|-------|-------|----------|
| **CXR Analyzer** | CXR Foundation | Process chest X-ray, extract features, detect abnormalities |
| **Finding Interpreter** | MedGemma | Interpret visual findings into clinical language |
| **Report Generator** | MedGemma | Create structured radiology report |
| **Priority Router** | MedGemma | Assess urgency, route to care pathway |

## HAI-DEF Models Used

- **CXR Foundation**: [google/cxr-foundation](https://huggingface.co/google/cxr-foundation)
- **MedGemma**: [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)

## Usage

1. Upload a chest X-ray image
2. (Optional) Add clinical context
3. Click "Analyze X-Ray"
4. View the generated report, priority assessment, and visualizations

## License

This project is submitted under CC BY 4.0 as required by the competition.

## Disclaimer

⚠️ **For demonstration purposes only. Not for clinical use.**
This AI system requires radiologist verification before any clinical decisions.

## Acknowledgments

- Google Health AI Developer Foundations team
- MedGemma Impact Challenge organizers
