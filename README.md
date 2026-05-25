# SmartPlate 🍽️

**AI-powered nutrition assistant** — photograph your meal and get instant nutritional analysis with personalized health advice.

SmartPlate combines three AI blocks in a single end-to-end pipeline:
1. **Computer Vision** — identifies the dish using a fine-tuned Vision Transformer (ViT)
2. **ML Health Classifier** — looks up nutritional data from Open Food Facts and scores the meal with XGBoost
3. **RAG Chatbot** — explains the result and gives personalized tips, grounded in WHO/DGE dietary guidelines

> ZHAW Module: KI-Anwendungen (FS 2026) | Deadline: 07. Juni 2026

---

## Architecture

```
Photo Input
    │
    ▼
┌─────────────────────────────────┐
│  Block 1 · Computer Vision      │
│  ViT fine-tuned on Food-101     │
│  → dish label (e.g. "pizza")    │
└───────────────┬─────────────────┘
                │ dish label
                ▼
┌─────────────────────────────────┐
│  Block 2 · ML Health Classifier │
│  Open Food Facts lookup         │
│  XGBoost health score (0–100)   │
│  → nutrients + health category  │
└───────────────┬─────────────────┘
                │ nutrients + score
                ▼
┌─────────────────────────────────┐
│  Block 3 · RAG + LLM            │
│  ChromaDB over WHO/DGE docs     │
│  gpt-4o-mini generation         │
│  → natural language explanation │
└───────────────┬─────────────────┘
                │
                ▼
          Gradio UI
```

Each block's output is the next block's input — this is a **pipeline**, not isolated models.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| CV | HuggingFace Transformers, ViT (`google/vit-base-patch16-224`), PyTorch |
| ML | scikit-learn, XGBoost, pandas |
| NLP/RAG | ChromaDB, sentence-transformers, OpenAI API (`gpt-4o-mini`) |
| UI | Gradio |
| Training | Google Colab (GPU), Jupyter notebooks |
| Deployment | Hugging Face Spaces (`Gianone/SmartPlate`) |

---

## Running locally

1. Activate venv: `source .venv/bin/activate`
2. Install deps: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add OPENAI_API_KEY
4. Run: `python app.py`
5. Open browser at http://127.0.0.1:7860

---

## Local Setup

### Prerequisites
- Python 3.11
- `pip` or `conda`
- An OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://huggingface.co/spaces/Gianone/SmartPlate
cd SmartPlate

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run the App

```bash
python app.py
```

The Gradio interface will open at `http://localhost:7860`.

---

## Training the Models

Training is intentionally **separated from inference**. Run the notebooks in order inside Google Colab (GPU recommended for the ViT fine-tuning):

| Notebook | Purpose |
|---|---|
| `notebooks/01_eda_food101.ipynb` | Explore Food-101 dataset |
| `notebooks/02_train_vit_cv.ipynb` | Fine-tune ViT on Food-101 subset |
| `notebooks/03_ml_health_classifier.ipynb` | Train XGBoost health classifier |
| `notebooks/04_rag_setup.ipynb` | Build ChromaDB vector store |

After training, export model artifacts to `models/` before running the app.

---

## Deploy to Hugging Face Spaces

```bash
# The Space is already configured at:
# https://huggingface.co/spaces/Gianone/SmartPlate

# Push updates via git:
git push origin main
```

Make sure to add `OPENAI_API_KEY` as a **Space Secret** in the HF Spaces settings (not in code).

---

## Demo

**Deployment URL:** _coming soon — will be published before 07.06.2026_

**Screenshots:**

| Step | Preview |
|---|---|
| Upload photo | _see assets/screenshots/_ |
| Nutrition result | _see assets/screenshots/_ |
| Health explanation | _see assets/screenshots/_ |

---

## Project Structure

```
smartplate/
├── app.py                    # Gradio inference app
├── requirements.txt
├── .env.example
├── notebooks/                # Training notebooks (run in Colab)
├── src/                      # Inference source code
│   ├── cv_model.py
│   ├── ml_model.py
│   ├── nlp_rag.py
│   ├── pipeline.py
│   └── data_loader.py
├── data/knowledge_base/      # RAG source documents (committed)
├── models/                   # Trained model artifacts (gitignored)
└── tests/
    └── test_pipeline.py
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for ZHAW KI-Anwendungen FS 2026 by Gianpiero Della Quila*
