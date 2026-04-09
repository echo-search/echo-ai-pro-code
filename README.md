# Echo AI Pro Code

## Description
From-blank HTML/CSS/JS generative AI.  
Two main parts: `train/` for training model, `api/` for backend inference, `docs/` for live single-page editor.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train model (optional starter dataset provided):
```bash
python train/train.py
```

3. Run API backend:
```bash
uvicorn api.inference:app --reload
```

4. Open `docs/index.html` in browser and connect to backend to generate HTML/CSS/JS.
