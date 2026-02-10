# RadioFlow Deployment Guide

## Deploying to HuggingFace Spaces

### Step 1: Create a HuggingFace Account
1. Go to [huggingface.co](https://huggingface.co)
2. Sign up or log in
3. Go to Settings → Access Tokens
4. Create a new token with write permissions

### Step 2: Create a New Space
1. Click on your profile → New Space
2. Configure:
   - **Space name**: `radioflow`
   - **License**: CC BY 4.0
   - **SDK**: Gradio
   - **Hardware**: CPU basic (or GPU if available)
   - **Visibility**: Public

### Step 3: Upload Files
You can either:

#### Option A: Git Push
```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/radioflow
cd radioflow

# Copy all project files
cp -r /path/to/project/* .

# Push
git add .
git commit -m "Initial RadioFlow deployment"
git push
```

#### Option B: Web Upload
1. Go to your Space's Files tab
2. Click "Upload files"
3. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `agents/` folder
   - `orchestrator/` folder
   - `utils/` folder
   - `config.py`

### Step 4: Configure Environment
1. Go to Space Settings → Variables and secrets
2. Add your HuggingFace token:
   - **Name**: `HF_TOKEN`
   - **Value**: Your token (keep secret)

### Step 5: Wait for Build
- The Space will automatically build
- Check the Logs tab for any errors
- First build takes 5-10 minutes

### Step 6: Test Your Deployment
1. Visit `https://huggingface.co/spaces/YOUR_USERNAME/radioflow`
2. Upload a test chest X-ray
3. Verify the workflow completes

---

## Local Development

### Prerequisites
- Python 3.10+
- pip or conda

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (for model access)
huggingface-cli login
```

### Run Locally
```bash
# Run tests first
python test_radioflow.py

# Start the app
python app.py
```

The app will be available at `http://localhost:7860`

---

## Troubleshooting

### "Model not found" Error
- Ensure you've accepted the model license on HuggingFace
- Check that HF_TOKEN is set correctly
- For gated models, you may need to request access

### Out of Memory
- Enable `LOW_MEMORY_MODE` in `config.py`
- Use CPU-only mode
- Reduce `MAX_NEW_TOKENS`

### Slow Inference
- Demo mode uses simulated outputs for speed
- For real inference, GPU is recommended
- Consider model quantization

### Build Fails on Spaces
1. Check the build logs
2. Verify all files are uploaded
3. Ensure requirements.txt versions are compatible
4. Try removing version pins if issues persist

---

## File Structure for Deployment

```
radioflow/
├── app.py                  # Main Gradio app (required)
├── requirements.txt        # Dependencies (required)
├── README.md              # Description for Space
├── space.yaml             # HuggingFace config
├── config.py              # Configuration
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── cxr_analyzer.py
│   ├── finding_interpreter.py
│   ├── report_generator.py
│   └── priority_router.py
├── orchestrator/
│   ├── __init__.py
│   └── workflow.py
└── utils/
    ├── __init__.py
    ├── visualization.py
    └── metrics.py
```

---

## Competition Submission Checklist

Before submitting to Kaggle:

- [ ] HuggingFace Space is live and working
- [ ] Public GitHub repository created
- [ ] 3-minute video recorded and uploaded
- [ ] Writeup completed (3 pages max)
- [ ] All links tested and accessible
- [ ] Submitted before deadline

### Required Links for Submission
1. **Video URL**: YouTube, Loom, or Google Drive
2. **Code Repository**: GitHub link
3. **Live Demo**: HuggingFace Spaces link
4. **Model (Bonus)**: HuggingFace model link (if fine-tuned)
