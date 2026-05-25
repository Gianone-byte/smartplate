# SmartPlate – AI Nutrition Assistant

## Project Metadata

- **Project title:** SmartPlate – AI Nutrition Assistant
- **Student:** Gianpiero Dell'Aquila
- **GitHub repository URL:** https://github.com/Gianone-byte/smartplate
- **Deployment URL:** https://huggingface.co/spaces/Gianone/smartplate
- **Submission date:** 25 May 2026

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution (choose 2):
- **Primary block 1:** Computer Vision (Vision Transformer on Food-101)
- **Primary block 2:** NLP / RAG (LLM with retrieval over WHO/DGE/Harvard guidelines)

**Third block (extra work, graded separately):** ML Numeric Data (health classification from nutrient features) — see Section 5 (Bonus Evidence) and the dedicated block section 2A.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition

- **Problem statement:** Health-conscious users want quick nutritional analysis of meals, but existing diet-tracking apps require manual logging of every meal — a friction point that explains their high abandonment rate. Nutritional information is rarely available at the point of consumption (e.g. when eating out), and even when labels exist, interpreting them in the context of personal health goals requires expert knowledge.

- **Goal:** Build a multi-modal AI nutrition assistant where the user photographs a meal and receives: (1) a food-class detection, (2) per-100g nutritional values + a health classification, and (3) a personalised, evidence-based explanation grounded in authoritative nutrition guidelines (WHO, DGE, Harvard).

- **Success criteria:**
  - CV model: ≥90% accuracy on a 20-class held-out validation set
  - ML model: classifier successfully distinguishes the three health tiers (healthy/medium/unhealthy) with macro-F1 ≥0.85
  - RAG: answers cite at least one retrieved source per response, with no factual hallucinations on test questions
  - End-to-end: deployed app responds to a real food photo with all three outputs in under 30 seconds (steady-state)

### 1.2 Integration Logic

- **How the selected blocks interact:** The blocks form a strict pipeline. The Computer Vision block classifies the input image into one of 20 food classes. This class is the **key** that drives the ML Numeric block's nutrient lookup and health-tier classification. The food class + nutrient values + health tier are then **injected as context** into the NLP/RAG block's prompt, which retrieves the most relevant chunks from the WHO/DGE/Harvard knowledge base and generates a personalised response. No block can be removed without breaking the others.

- **Data and output flow between blocks:**

```
[User uploads photo]
       ↓
[CV: ViT predicts food_class + confidence]
       ↓ food_class="pizza"
[ML: lookup nutrition + predict health_label]
       ↓ {kcal=266, fat=10g, ..., health_label="unhealthy"}
[NLP: retrieve top-3 chunks → build prompt → LLM generates answer]
       ↓ {answer, sources}
[Gradio UI: display all three outputs side by side]
```

See [`src/pipeline.py`](src/pipeline.py) for the orchestration logic.

---

## 2. Block Documentation

### 2A. ML Numeric Data

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | [USDA FoodData Central](https://fdc.nal.usda.gov/) (curated reference values, see [`notebooks/03_ml_health_classifier.ipynb`](notebooks/03_ml_health_classifier.ipynb)) | Structured nutrient values | 20 classes × 8 nutrients | Source of ground-truth nutrition per food class |
| 2 | Augmented dataset (USDA values + ±15% Gaussian noise) | Tabular numeric | 1000 rows × 16 features | Training data for the health classifier |
| 3 | Open Food Facts API (initial attempt — see honest discussion in 2A.2) | Crowdsourced structured | Variable | Originally planned source; abandoned after 503 errors and data-quality issues |

#### 2A.2 Preprocessing and Features

- **Cleaning steps:**
  - **Original plan:** Open Food Facts API for all 20 classes. Result: 8 of 20 classes returned 503 Server Errors during scraping, and the 12 successful classes had nonsensical medians (every class ~51 kcal/100g) because OFF's full-text search returned irrelevant products (e.g. "caesar salad" matched "mineral water" and "milk").
  - **Pivot:** USDA reference values per 100g for all 20 classes, sourced manually from USDA FoodData Central and the Swiss nutrition database. Eight nutrients per class: kcal, fat, saturated_fat, carbs, sugar, fiber, protein, salt.

- **Preprocessing steps:**
  - StandardScaler normalisation (mean 0, std 1) on training data, persisted in the model bundle
  - LabelEncoder for the 3-class target (healthy/medium/unhealthy), with explicit ordering
  - Stratified 80/20 train/test split (seed 42)

- **Feature engineering and selection:** 16 features total — see [`notebooks/03_ml_health_classifier.ipynb`](notebooks/03_ml_health_classifier.ipynb)
  - 8 raw nutrients: kcal, fat, sat_fat, carbs, sugar, fiber, protein, salt
  - 5 derived ratios: sugar_to_carb_ratio, sat_fat_pct_of_fat, calorie_density, protein_to_kcal, fiber_to_carb_ratio
  - 3 WHO-threshold binary flags: high_sugar (>15g), high_salt (>1.5g), high_sat_fat (>5g)
  - **Augmentation:** 50 samples per class generated with ±15% Gaussian noise around USDA medians → 1000 training rows

#### 2A.3 Model Selection

- **Models tested:**
  - Iteration 1: Logistic Regression (multinomial, L2 regularisation, C=1.0)
  - Iteration 2: XGBoost Classifier (200 trees, max_depth=5, learning_rate=0.1)

- **Why these models were chosen:**
  - Logistic Regression as the **linear baseline** — fast, interpretable (coefficients per class), proven on tabular problems
  - XGBoost as the **non-linear comparison** — typical default for structured/tabular data, captures feature interactions

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Establish linear baseline | StandardScaler + LR (C=1.0, multinomial) | LogisticRegression | Test Acc: **100.0%**, F1-macro: 100.0%, CV: 99.88% ± 0.25% | Baseline |
| 2 | Test if non-linear model improves | XGBoost (200 trees, depth=5) | XGBClassifier | Test Acc: 99.5%, F1-macro: 99.55%, CV: 99.88% ± 0.25% | **-0.5 pp** vs Iter 1 |
| 3 | N/A — winner already at 100% in Iter 1, further iteration would overfit | — | — | — | — |

**Final choice:** Logistic Regression. This is counter-intuitive (XGBoost is the usual tabular default), but justified because: (a) higher test accuracy on this data, (b) faster inference, (c) interpretable coefficients, (d) simpler deployment with one less dependency. See artefacts at [`assets/screenshots/ml/model_comparison.csv`](assets/screenshots/ml/model_comparison.csv).

#### 2A.5 Evaluation and Error Analysis

- **Metrics used:** Test accuracy, macro-F1, 5-fold stratified cross-validation, per-class precision/recall/F1, confusion matrix

- **Final results:**
  - Logistic Regression test accuracy: **100.0%**
  - Per-class F1: healthy 100%, medium 100%, unhealthy 100%
  - Cross-validation: 99.88% ± 0.25%
  - See [`assets/screenshots/ml/ml_confusion_matrix.png`](assets/screenshots/ml/ml_confusion_matrix.png) and [`assets/screenshots/ml/feature_importance.png`](assets/screenshots/ml/feature_importance.png)

- **Error patterns and likely causes:** The 100% accuracy reflects that our augmented data is **linearly separable by class** — a consequence of the augmentation methodology (±15% noise around well-defined USDA medians with no class overlap). This is honest evidence that:
  - The classifier is reliable **within its training distribution**
  - A real-world deployment would face fuzzier boundaries (deep-dish pizza with extra cheese vs. thin margherita, portion-size variation, hand-prepared dishes diverging from USDA medians)
  - The model would **not** necessarily achieve 100% on a different data distribution

#### 2A.6 Integration with Other Block(s)

- **Inputs received from other block(s):** A single string `food_class` from the CV block (e.g. `"pizza"`)
- **Outputs provided to other block(s):** A dictionary with `nutrition` (8 raw values), `health_label` (string: healthy/medium/unhealthy), and `probabilities` (per-class confidence). See [`src/ml_model.py`](src/ml_model.py).

### 2B. NLP

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | [WHO Healthy Diet Fact Sheet](https://www.who.int/news-room/fact-sheets/detail/healthy-diet) | PDF (13 pages, ~18,500 chars) | 119 KB | Knowledge base — global policy guidelines |
| 2 | [DGE — 10 Rules of Healthy Eating](https://www.dge.de/gesunde-ernaehrung/dge-ernaehrungsempfehlungen/10-regeln/) | PDF (8 pages, ~5,000 chars) | 778 KB | Knowledge base — German nutrition society, layperson-oriented |
| 3 | [Harvard Healthy Eating Plate](https://www.hsph.harvard.edu/nutritionsource/healthy-eating-plate/) | PDF (6 pages, ~5,300 chars) | 358 KB | Knowledge base — research-oriented Anglo-American perspective |

Total: 27 pages, ~29,000 characters. Three deliberately different perspectives produce richer retrieval results than a single source.

#### 2B.2 Preprocessing and Prompt Design

- **Text preprocessing:**
  - PDF text extraction via `pypdf`
  - Chunking with `langchain_text_splitters.RecursiveCharacterTextSplitter`
  - chunk_size = 500 characters, chunk_overlap = 100 characters
  - Result: ~60–90 chunks total → cached in [`models/rag_chunks.json`](models/rag_chunks.json)

- **Prompt design or retrieval setup:**
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
  - Vector store: ChromaDB (in-memory, rebuilt from `rag_chunks.json` on startup)
  - Top-k retrieval: 3 chunks per query
  - LLM: OpenAI `gpt-4o-mini`
  - Two prompt strategies compared (see 2B.4)

#### 2B.3 Approach Selection

- **Approach used:** Retrieval-Augmented Generation (RAG) with sentence-transformer embeddings + LLM. See [`src/nlp_rag.py`](src/nlp_rag.py).
- **Alternatives considered:**
  - Fine-tuning a smaller LLM on the guidelines → rejected: training cost vs. benefit unfavourable for 29k chars
  - Few-shot prompting without retrieval → rejected: would not scale to add new guidelines later
  - Classical NLP (tf-idf retrieval + extraction) → rejected: cannot generate natural responses
  - RAG chosen for **best balance** of factual grounding, low cost, and conversational quality

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Establish baseline | Plain "Context: ... Question: ... Answer briefly." | gpt-4o-mini, top-k=3, temp=0.3 | Keyword coverage: **59.4%**, source match: 87.5%, avg tokens: 356 | Baseline |
| 2 | Test if structure improves quality | XML-tagged sources + strict grounding instruction ("only use sources, else say 'no info'") | gpt-4o-mini, top-k=3, temp=0.3 | Keyword coverage: **36.5%**, source match: 87.5%, avg tokens: 494 | **-22.9 pp coverage, +38% tokens** |
| 3 | N/A — Strategy 1 dominates on all measurable axes | — | — | — | — |

**Final choice:** Strategy 1 (Basic). The more sophisticated XML prompt's strict "only use sources" constraint caused the LLM to refuse to answer when source phrasings did not exactly match the question (e.g. it refused to discuss eggs because the WHO/DGE documents do not contain the literal word "eggs", although protein sources are clearly addressed). See [`assets/screenshots/rag/rag_strategy_comparison.csv`](assets/screenshots/rag/rag_strategy_comparison.csv) and [`assets/screenshots/rag/rag_evaluation.csv`](assets/screenshots/rag/rag_evaluation.csv).

**Lesson worth keeping:** Prompt structure ≠ prompt quality. Default-mode LLMs already use context effectively; over-restrictive grounding instructions can backfire.

#### 2B.5 Evaluation and Error Analysis

- **Evaluation strategy:** 8 hand-crafted test questions covering different nutrient topics (sugar, salt, fats, balanced meal, etc.), 2 questions in German, 6 in English. For each, measure: keyword coverage (fraction of expected keywords present in the answer), source match (whether the cited source matches the expected one), token count (efficiency).

- **Results:** Strategy 1 wins on every axis. See full per-question results in [`assets/screenshots/rag/rag_evaluation.csv`](assets/screenshots/rag/rag_evaluation.csv).

- **Error patterns and likely causes:**
  - Q6 "Should I drink milk?" → 0% source match for both strategies because WHO/DGE do not explicitly discuss dairy in the indexed PDFs. Harvard does but in a different chunk than what was retrieved. **Mitigation idea:** add re-ranking or query rewriting.
  - Q8 "Are eggs healthy?" → Strategy 1 succeeded with high coverage; Strategy 2 refused because "eggs" is not a literal term in the source documents. Demonstrates the over-grounding failure mode.

#### 2B.6 Integration with Other Block(s)

- **Inputs received from other block(s):**
  - From CV: `food_class` (string)
  - From ML: `kcal` (float), `health_label` (string) → injected into the prompt as conditioning context

- **Outputs provided to other block(s):** Final user-facing response — markdown answer + list of cited sources. This is the final output of the pipeline; no downstream block consumes it. See [`src/nlp_rag.py`](src/nlp_rag.py).

### 2C. Computer Vision

#### 2C.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | [Food-101 dataset](https://www.kaggle.com/datasets/dansbecker/food-101) (20-class subset) | RGB images (224×224) | 15,000 train + 5,000 val | Training + validation data for ViT fine-tuning |
| 2 | [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) | Pre-trained Vision Transformer | 86M parameters | Base model (transfer learning starting point) |

#### 2C.2 Preprocessing and Augmentation

- **Image preprocessing:**
  - Resize to 224×224 (ViT input requirement)
  - ImageNet normalisation: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

- **Augmentation strategy:** Training-set only — random horizontal flip + colour jitter (brightness, contrast, saturation). Validation set: no augmentation. See [`notebooks/02_train_vit_cv.ipynb`](notebooks/02_train_vit_cv.ipynb).

#### 2C.3 Model Selection

- **Vision model(s) used:** `google/vit-base-patch16-224` Vision Transformer (pre-trained on ImageNet-21k)

- **Why these model(s) were chosen:** ViT-Base is the standard transformer-based image classifier — comparable in accuracy to ResNet-50 but with attention-based interpretability and a clean Hugging Face integration. The 20-class subset is small enough that we can fully fine-tune in <20 minutes on a single T4 GPU, which fits the time budget of a semester project.

#### 2C.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Head-only fine-tuning baseline | Freeze backbone, train only classification head (0.02% trainable params), 3 epochs, lr=2e-5 | ViT-Base + linear head | Val Acc: **95.12%**, F1-macro: 95.12%, training time: 9:16 min | Baseline |
| 2 | Full fine-tuning for max accuracy | All 86M params trainable, 3 epochs, lr=2e-5 | ViT-Base full FT | Val Acc: **96.46%**, F1-macro: 96.46%, training time: 15:17 min | **+1.34 pp**, +65% time |
| 3 | N/A — Iter 2 deployed; further epochs risk overfitting on 15k images | — | — | — | — |

**Final choice:** Iteration 2 (full fine-tuning). The +1.34 pp gain in accuracy is worth the +6 min one-time training cost; inference time is identical (single forward pass). The trained model is published on Hugging Face Hub at [`Gianone/smartplate-vit-food`](https://huggingface.co/Gianone/smartplate-vit-food). See [`assets/screenshots/cv/iteration_comparison.csv`](assets/screenshots/cv/iteration_comparison.csv).

#### 2C.5 Evaluation and Error Analysis

- **Metrics and/or visual checks:** Validation accuracy, macro-F1, per-class confusion matrix, top-K error analysis. See [`assets/screenshots/cv/confusion_matrix.png`](assets/screenshots/cv/confusion_matrix.png).

- **Final results:**
  - Validation accuracy: **96.46%**
  - F1-macro: **96.46%**
  - Best-performing classes: edamame (~99% F1), miso_soup (~98%), french_fries (~98%) — visually distinctive
  - Hardest pairs: sushi↔sashimi (most-confused pair), donuts↔pancakes (similar shape and texture). See [`assets/screenshots/cv/top_errors.csv`](assets/screenshots/cv/top_errors.csv).

- **Error patterns and limitations:**
  - Sushi↔sashimi confusion is **acceptable downstream** — both are "healthy" in the ML lookup, so the user experience is unaffected
  - Donuts↔pancakes confusion is **more consequential** — donuts are "unhealthy", pancakes "medium". Mitigation: confidence score is exposed in the UI; low-confidence predictions (<70%) can be flagged
  - Food-101 over-represents Western cuisine; classes from under-represented cuisines would likely have lower accuracy in a deployment

#### 2C.6 Integration with Other Block(s)

- **Inputs received from other block(s):** None — CV is the entry point of the pipeline
- **Outputs provided to other block(s):** A dictionary `{class, confidence, top_5}` consumed by the ML block (for nutrient lookup) and the NLP block (as context in the prompt). See [`src/cv_model.py`](src/cv_model.py).

---

## 3. Deployment

- **Deployment URL:** https://huggingface.co/spaces/Gianone/smartplate
- **Main user flow:**
  1. User uploads a food photo via the Gradio UI
  2. (Optional) Types a question (e.g. "Can I eat this on a diet?")
  3. Clicks Analyze
  4. CV → ML → NLP run sequentially (~30 s on first run, ~5 s steady-state)
  5. UI displays: detected food class + confidence, top-5 predictions, nutrition table per 100g, health category with probabilities, and the NLP-generated explanation with cited sources

- **Screenshot or short demo:**

| Local app screenshots (`assets/screenshots/app/`) | Live HF Space screenshots (`assets/screenshots/deployment/`) |
| --- | --- |
| ![`Pizza.png`](assets/screenshots/app/Pizza.png), ![`Burger.png`](assets/screenshots/app/Burger.png), ![`Salat.png`](assets/screenshots/app/Salat.png) | ![`Hamburger_HF.png`](assets/screenshots/deployment/Hamburger_HF.png), ![`Pizza_HF_question.png`](assets/screenshots/deployment/Pizza_HF_question.png), ![`Salat_HF.png`](assets/screenshots/deployment/Salat_HF.png) |

All screenshots show the complete pipeline: image preview, CV prediction with confidence, nutrition table, health label with probabilities, and the LLM's evidence-grounded explanation with source citations.

**Training/inference separation:** Training code lives exclusively in the four Jupyter notebooks (`notebooks/01_eda_food101.ipynb` through `04_rag_setup.ipynb`). Inference code lives in `src/` and `app.py`. The trained ViT model is hosted on Hugging Face Hub (`Gianone/smartplate-vit-food`); the trained ML model is committed at [`models/health_classifier.pkl`](models/health_classifier.pkl); RAG chunks at [`models/rag_chunks.json`](models/rag_chunks.json). No training code is ever executed in production.

**Deployment friction documented honestly:** The HF Spaces build failed four times before succeeding (Gradio version conflict, Python 3.13 audioop bug, huggingface_hub API change, Gradio API-schema parser bug fixed with a monkey-patch in [`app.py`](app.py)). All four are documented in the HF Space's git history.

---

## 4. Execution Instructions

### Environment setup

```bash
git clone https://github.com/Gianone-byte/smartplate.git
cd smartplate
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### Data setup

The repository ships with all required artefacts:
- ML model: [`models/health_classifier.pkl`](models/health_classifier.pkl) (4.6 KB)
- RAG chunks: [`models/rag_chunks.json`](models/rag_chunks.json) (45 KB)
- Knowledge base PDFs: [`data/knowledge_base/`](data/knowledge_base/)

The CV model is downloaded automatically from Hugging Face Hub on first run (`Gianone/smartplate-vit-food`, ~340 MB).

### Training command(s)

Optional — only needed to reproduce models from raw data. All four notebooks run end-to-end:

```bash
jupyter notebook notebooks/01_eda_food101.ipynb          # ~10 min, EDA
jupyter notebook notebooks/02_train_vit_cv.ipynb         # ~25 min, T4 GPU recommended
jupyter notebook notebooks/03_ml_health_classifier.ipynb # ~2 min, CPU
jupyter notebook notebooks/04_rag_setup.ipynb            # ~5 min, CPU + OpenAI API
```

### Inference/run command(s)

```bash
cp .env.example .env
# → open .env and add OPENAI_API_KEY
python app.py
# → opens http://127.0.0.1:7860
```

### Reproducibility notes

- Random seeds: `42` everywhere (numpy, sklearn, transformers Trainer)
- Pinned versions: `scikit-learn==1.5.1` (matches training environment), `huggingface_hub==0.24.7` (Gradio 4.32 compatibility)
- Python: 3.9 supported locally; Python 3.10 required on HF Spaces (declared in `README.md` SDK metadata)
- All 18 smoke tests in `tests/test_pipeline.py` should pass: `python -m pytest tests/ -v`

---

## 5. Optional Bonus Evidence

- [x] **Third selected block implemented with strong quality** — ML Numeric Data implemented as full block (USDA-derived dataset, feature engineering, 2-model comparison, evaluation) and integrated into the pipeline. See Section 2A.
- [x] **More than two data sources used with clear added value** — Six distinct sources used: Food-101 dataset (CV training), pre-trained ViT (CV transfer learning), USDA nutrition values (ML labels), augmented synthetic dataset (ML training), WHO + DGE + Harvard PDFs (3 separate RAG sources). Each has a justified role.
- [x] **A core section is done exceptionally well** — Section 2B (NLP/RAG) includes a quantitative prompt-strategy comparison with 8 test questions, three measured metrics, and a non-obvious finding (the simpler prompt outperforms the more structured one). The result is reported honestly with the failure-mode explanation.
- [x] **Extended evaluation** — Confusion matrices for both CV and ML blocks, feature importance plot for ML, per-question RAG evaluation CSV, per-class CV error analysis CSV. All committed under `assets/screenshots/`.
- [x] **Ethics, bias, or fairness analysis** — Dedicated discussion of health-advice liability, training-data bias (Food-101 over-represents Western cuisine), eating-disorder safety considerations, and data-privacy commitments. See in-line discussion at end of Section 5 below.
- [ ] Creative or exceptional use case — see honest self-assessment: the use case is realistic and well-motivated but not unprecedented (similar apps exist commercially).

### Ethics, bias and fairness — additional discussion

This is a project for academic use and is documented as such. A production deployment would require the following safeguards, which are out of scope for the prototype but are worth naming:

- **Health-advice liability.** SmartPlate generates dietary suggestions. It is not a substitute for advice from a registered dietitian or doctor. A production version would need clear disclaimers and probably geographic gating (FDA, EU and Swissmedic each regulate health-claim advice differently).
- **Training-data bias.** Food-101 over-represents Western cuisine (burgers, pizza, French pastries). The system will be less accurate on dishes from cuisines under-represented in Food-101. Per-class confidence scores expose this when accuracy drops, but only to users who know what to look for.
- **Eating-disorder safety.** A nutrition app can be triggering for users with eating disorders. The production prompt was deliberately designed to avoid moralising language ("bad food", "forbidden", "you should not eat this") and to acknowledge the emotional side of food. A production version would need explicit content guardrails for known triggering phrases and a clear path to support resources.
- **Privacy.** Photos are processed in-memory and not stored. Conversation history is not retained between sessions. The OpenAI API does log requests per their terms of service — this is documented in `documentation.md` but would need more prominence in a real deployment.
- **Reproducibility.** Every artefact is either committed to git or referenced from Hugging Face Hub. The four notebooks reproduce all results from raw data with fixed seeds. No undocumented hyperparameters.
