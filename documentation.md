# SmartPlate — Project Documentation

**Module:** KI-Anwendungen (FS 2026) | ZHAW School of Engineering
**Student:** Gianpiero Della Quila
**Deadline:** 07. Juni 2026
**HF Space:** https://huggingface.co/spaces/Gianone/SmartPlate

---

## 1. Project Idea & Methodology

### What problem does the project solve?

> TODO: Describe the core problem — people often lack awareness of the nutritional quality of their meals. Manual calorie counting is tedious and error-prone. SmartPlate automates the entire process from photo to personalized advice.

### Why is this a realistic use case?

> TODO: Explain real-world applicability — smartphone camera always available, Open Food Facts has 3M+ products, WHO/DGE guidelines are publicly available. The three-block pipeline mirrors how a real nutritionist would think.

### How does the project combine the three AI blocks into a pipeline?

> TODO: Explain that the output of Block 1 (dish label) feeds Block 2 (lookup + scoring), whose output (nutrients + health score) feeds Block 3 (RAG + LLM explanation). Each block is meaningless without the previous one — the pipeline is the product.

### Which 20–30 Food-101 classes were selected and why?

> TODO: List the chosen classes (e.g., pizza, sushi, salad, hamburger, ...). Justify the selection: class balance, relevance to Swiss eating habits, coverage across health score spectrum (some healthy, some not), training efficiency.

### What is explicitly out of scope?

> TODO: List exclusions — real-time video, barcode scanning, restaurant menus, allergy detection, medical advice. State that the system is for educational/informational use only.

---

## 2. Data & Preprocessing

### Block 1 — Computer Vision: Food-101

#### Data Source

> TODO: Describe Food-101 (Kaggle / TensorFlow Datasets / HuggingFace Datasets). Cite: Bossard et al., 2014. N=101,000 images, 101 classes, 750 train + 250 test per class.

#### Subset Selection

> TODO: Which 20–30 classes? How was the subset created (stratified sampling, keeping original 750/250 split, or resplit)?

#### Preprocessing Pipeline

> TODO: Describe the image preprocessing applied for ViT:
> - Resize to 224×224
> - Normalize with ImageNet mean/std ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225])
> - Data augmentation during training (random horizontal flip, random crop, color jitter)
> - No augmentation during validation/test

#### EDA Findings

> TODO: Document key EDA findings from `01_eda_food101.ipynb`:
> - Class distribution (balanced?)
> - Typical image sizes before resizing
> - Any quality issues (blurry, mislabeled)?
> - Sample images per class

---

### Block 2 — ML Health Classifier: Open Food Facts

#### Data Source

> TODO: Open Food Facts CSV dump (approx. 3M products). URL: https://world.openfoodfacts.org/data. Describe which columns were selected (product_name, energy_100g, fat_100g, saturated-fat_100g, sugars_100g, fiber_100g, proteins_100g, salt_100g, nutriscore_grade).

#### Data Cleaning

> TODO: Describe cleaning steps:
> - Remove rows with >X% missing nutritional values
> - Filter to food items matching the 20–30 Food-101 class names
> - Handle outliers (e.g., energy_100g > 4000 kcal)
> - Imputation strategy for remaining missing values (median imputation)

#### Feature Engineering

> TODO: Describe features engineered for XGBoost:
> - Raw nutritional values per 100g
> - Derived: sugar_to_protein_ratio, saturated_fat_share
> - Target: health_label (healthy / moderate / unhealthy) — defined by Nutri-Score A/B = healthy, C = moderate, D/E = unhealthy

#### EDA Findings

> TODO: Document from `03_ml_health_classifier.ipynb`:
> - Class distribution of health labels
> - Correlation heatmap of nutritional features
> - Most predictive features (SHAP values)

---

### Block 3 — NLP/RAG: Nutrition Guidelines

#### Data Source

> TODO: List the documents used for the knowledge base:
> - WHO Healthy Diet Factsheet (PDF)
> - DGE 10 Regeln der gesunden Ernährung (PDF/HTML)
> - Harvard T.H. Chan School of Public Health — Nutrition Source (scraped HTML)
> Stored in: `data/knowledge_base/`

#### Preprocessing Pipeline

> TODO: Describe RAG preprocessing:
> - PDF parsing with pypdf
> - Text cleaning (remove headers/footers, fix encoding)
> - Chunking strategy: chunk size = X tokens, overlap = Y tokens
> - Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (or similar)
> - Storage: ChromaDB persistent collection

---

## 3. Modeling & Implementation

### Block 1 — ViT Fine-Tuning

#### Model Choice

> TODO: Justify `google/vit-base-patch16-224`:
> - Pre-trained on ImageNet-21k + fine-tuned on ImageNet-1k
> - Strong baseline for image classification
> - Efficient patch-based attention mechanism
> - Available via HuggingFace Transformers

#### Training Configuration

> TODO: Document from `02_train_vit_cv.ipynb`:
> - Epochs: ?
> - Learning rate: ? (with warmup scheduler)
> - Batch size: ? (constrained by Colab GPU memory)
> - Optimizer: AdamW
> - Loss: CrossEntropyLoss
> - Hardware: Colab T4 / A100
> - Training time: ? minutes

#### Iterations & Experiments

> TODO: Describe at least 2 training runs with different hyperparameters and what changed.

---

### Block 2 — XGBoost Health Classifier

#### Model Choice

> TODO: Justify XGBoost:
> - Handles tabular nutritional data well
> - Robust to missing values
> - Interpretable via SHAP
> - Faster to train than neural alternatives for this task size

#### Training Configuration

> TODO: Document from `03_ml_health_classifier.ipynb`:
> - n_estimators: ?
> - max_depth: ?
> - learning_rate: ?
> - Train/val/test split: 70/15/15
> - Cross-validation: 5-fold

#### Baseline Comparison

> TODO: Compare XGBoost against at least one baseline (e.g., Logistic Regression, Random Forest). Show metrics table.

---

### Block 3 — RAG Pipeline

#### Architecture

> TODO: Describe the RAG setup:
> - Retriever: ChromaDB + sentence-transformers (cosine similarity, top-k=3)
> - Generator: OpenAI `gpt-4o-mini`
> - Prompt template: [paste the system prompt here]
> - How nutrients + health score from Block 2 are injected into the prompt context

#### Design Decisions

> TODO: Explain choices:
> - Why ChromaDB over FAISS / Pinecone?
> - Why `gpt-4o-mini` (cost, speed, quality trade-off)?
> - Chunk size selection rationale

---

## 4. Evaluation & Analysis

### Block 1 — CV Evaluation

#### Metrics

> TODO: Report on test set:
> - Top-1 Accuracy: ?%
> - Top-5 Accuracy: ?%
> - Per-class F1 scores (confusion matrix)

#### Error Analysis

> TODO: Which classes are most confused and why? Show example failure cases with images.

---

### Block 2 — ML Evaluation

#### Metrics

> TODO: Report on test set:
> - Accuracy: ?%
> - Macro F1: ?%
> - Confusion matrix (3 classes: healthy / moderate / unhealthy)
> - SHAP feature importance plot

#### Error Analysis

> TODO: Are errors clustered in the "moderate" class? Why might boundary cases be hard?

---

### Block 3 — RAG/LLM Evaluation

#### Qualitative Evaluation

> TODO: Show 3–5 example inputs/outputs. Rate: relevance, groundedness (does it cite guidelines?), helpfulness.

#### Retrieval Quality

> TODO: Check if retrieved chunks are actually relevant to the query. Report retrieval precision@3 for a small sample.

---

### End-to-End Pipeline Evaluation

> TODO: Run 10–20 test images through the full pipeline. Report:
> - How often does a correct dish label lead to the correct health score?
> - Where does the pipeline break most often? (CV error propagation)
> - Latency: time per pipeline run (seconds)

---

## 5. Deployment

### Deployment URL

> TODO: https://huggingface.co/spaces/Gianone/SmartPlate

### Screenshots

> TODO: Add screenshots of:
> - The Gradio UI with a food photo input
> - A sample output showing dish label + nutrients + health score + explanation

### Training / Inference Separation

> TODO: Explain the architectural decision:
> - Notebooks run in Google Colab for training (GPU access, disposable environment)
> - `app.py` + `src/` contain only inference code (loads pre-trained artifacts)
> - Model artifacts are exported and uploaded separately (e.g., HF Hub or Space artifacts)
> - No training code in `app.py`

### HF Spaces Configuration

> TODO: Describe `README.md` YAML frontmatter for HF Spaces:
> ```yaml
> title: SmartPlate
> emoji: 🍽️
> colorFrom: green
> colorTo: blue
> sdk: gradio
> sdk_version: 4.44.0
> app_file: app.py
> pinned: false
> ```

---

## 6. Execution Instructions

### Local Execution (Inference)

```bash
# 1. Clone repo
git clone https://huggingface.co/spaces/Gianone/SmartPlate && cd SmartPlate

# 2. Create environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# → Edit .env: add OPENAI_API_KEY=sk-...

# 5. Run app
python app.py
# → Open http://localhost:7860
```

### Google Colab Execution (Training)

> TODO: Describe how to open each notebook in Colab:
> 1. Upload notebook or open from GitHub
> 2. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
> 3. Run `!pip install -r requirements.txt` in first cell
> 4. Set OPENAI_API_KEY via Colab Secrets (key icon in sidebar)
> 5. Run all cells in order

### Reproducibility Notes

> TODO: Document:
> - Random seeds used (set in each notebook)
> - Exact dataset version (Food-101 from HuggingFace Datasets, version X)
> - Model checkpoint saved at: `models/vit_food101_finetuned/`
> - XGBoost model saved at: `models/xgb_health_classifier.pkl`
> - ChromaDB collection persisted at: `chroma_db/`

---

*Last updated: 2026-05-24*
