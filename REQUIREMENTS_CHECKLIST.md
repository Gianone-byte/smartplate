# SmartPlate — ZHAW Requirements Checklist

**Module:** KI-Anwendungen (FS 2026)
**Submission deadline:** 07.06.2026, 18:00 Uhr
**Repository:** GitHub (add collaborators: `jasminh`, `bkuehnis`)
**Last updated:** 2026-05-24 | Phase: 1 — Scaffold complete

> Legend: `[ ]` TODO · `[~]` IN PROGRESS · `[x]` DONE
>
> Update this file after each phase. Every checkbox that hits DONE needs a
> concrete file + line reference so graders can verify without asking.

---

## Section A — General Project Requirements

| # | Requirement | Status | Where fulfilled | Note |
|---|---|---|---|---|
| A1 | New, self-collected datasets (not course data) | `[x]` DONE | Food-101 (CV), Open Food Facts (ML), WHO/DGE PDFs (NLP) — all external | Confirmed: none of these are ZHAW course datasets |
| A2 | Three AI blocks combined in a pipeline (not isolated) | `[x]` DONE | `src/pipeline.py` — `SmartPlatePipeline.run()` chains all 3 blocks sequentially | Each block's output is the next block's input |
| A3 | Deployed working application | `[ ]` TODO | `app.py` (Gradio) → HF Spaces `Gianone/SmartPlate` | Pending: model training + HF Space publish |
| A4 | Training code separated from inference code | `[x]` DONE | Notebooks = training (Colab), `src/` + `app.py` = inference only | No `model.fit()` or training loops in `app.py` / `src/` |
| A5 | Reproducible experiments (seeds, fixed versions) | `[~]` IN PROGRESS | `requirements.txt` has pinned versions; seeds to be set in each notebook | Add `SEED = 42` + `random.seed` / `torch.manual_seed` in all notebooks |
| A6 | Ethical considerations documented | `[ ]` TODO | `documentation.md` → add sub-section under §6 or §1 | Cover: data privacy, model bias (food class coverage), medical disclaimer |

---

## Section B — Documentation Requirements

The `documentation.md` file follows the Q&A structure. Each sub-section below maps to one of the 6 required sections.

| # | Section | Status | File & Location | Note |
|---|---|---|---|---|
| B1 | **Project Idea & Methodology** — problem, use case, block combination, scope | `[~]` IN PROGRESS | `documentation.md` §1 | All TODO placeholders filled with structure; content to be written per phase |
| B2 | **Data & Preprocessing** — sources, cleaning, EDA, feature engineering per block | `[ ]` TODO | `documentation.md` §2 | Fill after notebooks 01 + 03 are complete |
| B3 | **Modeling & Implementation** — model choice, training, comparisons, iterations | `[ ]` TODO | `documentation.md` §3 | Fill after training notebooks 02 + 03 + 04 |
| B4 | **Evaluation & Analysis** — metrics, error analysis, block-specific results | `[ ]` TODO | `documentation.md` §4 | Fill after full pipeline runs on test data |
| B5 | **Deployment** — live URL, screenshots, training/inference separation explained | `[ ]` TODO | `documentation.md` §5, `assets/screenshots/` | Fill after HF Space is live |
| B6 | **Execution Instructions** — full reproduction steps, seeds, env setup | `[~]` IN PROGRESS | `documentation.md` §6, `README.md` Setup section | Local + Colab steps drafted; Colab notebook-specific steps to fill |

---

## Section C — Assessment Criteria

| # | Criterion | Status | Evidence | Note |
|---|---|---|---|---|
| C1 | Code quality (readable, modular, documented) | `[x]` DONE | All `src/` modules have Google-style docstrings; clear class/function separation | Test with pylint/flake8 before submission |
| C2 | Pipeline integration (end-to-end data flow) | `[x]` DONE | `src/pipeline.py:SmartPlatePipeline.run()` — Block 1→2→3 chained | Visualised in README ASCII diagram |
| C3 | Comparison of at least 2 model variants per block | `[ ]` TODO | Notebooks 02 (ViT LR experiments), 03 (XGBoost vs. LogReg), 04 (chunk size) | Each notebook has placeholder cells for Experiment 1 + 2 |
| C4 | Meaningful evaluation metrics per block | `[ ]` TODO | `documentation.md` §4; notebooks will log metrics | CV: top-1/top-5 acc + F1. ML: acc + macro-F1 + SHAP. NLP: qualitative + retrieval P@3 |
| C5 | Presentation-ready demo (app works, screenshots available) | `[ ]` TODO | `app.py`, `assets/screenshots/` | App UI is built; needs trained models to show real output |

---

## Section D — Submission

| # | Item | Status | Detail |
|---|---|---|---|
| D1 | GitHub repository public | `[ ]` TODO | Create remote on GitHub, set visibility to public |
| D2 | Add collaborator `jasminh` | `[ ]` TODO | GitHub → Settings → Collaborators → Add `jasminh` |
| D3 | Add collaborator `bkuehnis` | `[ ]` TODO | GitHub → Settings → Collaborators → Add `bkuehnis` |
| D4 | Submit link via ZHAW form / ILIAS | `[ ]` TODO | Deadline: **07.06.2026, 18:00 Uhr** — do NOT miss this |
| D5 | HF Space URL working at submission time | `[ ]` TODO | `https://huggingface.co/spaces/Gianone/SmartPlate` |
| D6 | `documentation.md` complete (all TODOs filled) | `[ ]` TODO | All `> TODO:` placeholders replaced with real content |
| D7 | `README.md` updated with live demo URL + screenshots | `[ ]` TODO | Replace "coming soon" placeholder in README |

---

## Section E.1 — ML Block on Numeric Data (XGBoost Health Classifier)

| # | Requirement | Status | Where fulfilled | Note |
|---|---|---|---|---|
| E1.1 | New, real-world numeric dataset (not toy data) | `[x]` DONE | Open Food Facts CSV (~3M products) — `data/raw/` (gitignored) | Not iris, not MNIST — genuinely new data |
| E1.2 | Data cleaning & preprocessing documented | `[ ]` TODO | `notebooks/03_ml_health_classifier.ipynb` Cell 3; `documentation.md` §2 | Missing value handling, outlier removal, median imputation |
| E1.3 | Feature engineering with justification | `[ ]` TODO | `notebooks/03_ml_health_classifier.ipynb` Cell 4; `documentation.md` §2 | Derived: sugar/protein ratio, sat-fat share. Target: from Nutri-Score |
| E1.4 | At least 2 models trained and compared | `[ ]` TODO | `notebooks/03_ml_health_classifier.ipynb` Cells 7–9 | Baseline: Logistic Regression. Main: XGBoost with 2 hyperparameter sets |
| E1.5 | Meaningful evaluation metrics (not just accuracy) | `[ ]` TODO | `notebooks/03_ml_health_classifier.ipynb` Cell 10; `documentation.md` §4 | Macro F1, confusion matrix, SHAP feature importance |
| E1.6 | Model artifact saved and loaded in inference | `[ ]` TODO | `models/xgb_health_classifier.pkl` → `src/ml_model.py:MLModel.load()` | Pickle format; path configured in `MLModel.model_path` |
| E1.7 | Block output feeds next block (pipeline requirement) | `[x]` DONE | `src/ml_model.py:NutritionResult` → consumed by `src/nlp_rag.py:RAGPipeline.generate()` | `NutritionResult` dataclass is the interface contract |

---

## Section E.2 — NLP Block (RAG + LLM)

| # | Requirement | Status | Where fulfilled | Note |
|---|---|---|---|---|
| E2.1 | New text corpus (not course-provided) | `[x]` DONE | WHO Healthy Diet Factsheet + DGE 10 Regeln + Harvard Nutrition Source → `data/knowledge_base/` | Must be committed to repo (PDFs/texts, not gitignored) |
| E2.2 | Document preprocessing & chunking documented | `[ ]` TODO | `notebooks/04_rag_setup.ipynb` Cells 2–4; `documentation.md` §2 (Block 3) | PyPDF extraction, cleaning, chunk_size/overlap experiments |
| E2.3 | Embedding + vector store setup documented | `[ ]` TODO | `notebooks/04_rag_setup.ipynb` Cell 5; `src/nlp_rag.py:RAGPipeline.load()` | ChromaDB + `all-MiniLM-L6-v2`; persist at `chroma_db/` |
| E2.4 | Retrieval quality verified | `[ ]` TODO | `notebooks/04_rag_setup.ipynb` Cell 6; `documentation.md` §4 (Block 3) | Manual precision@3 check on 5 sample queries |
| E2.5 | LLM prompt documented (system prompt + template) | `[x]` DONE | `src/nlp_rag.py:SYSTEM_PROMPT` — constant at module level | Prompt template to be refined in notebook 04 |
| E2.6 | Generated output grounded in retrieved documents | `[ ]` TODO | `documentation.md` §4 (qualitative evaluation) | Verify LLM cites or uses guideline facts, not hallucinations |

---

## Section E.3 — Computer Vision Block (ViT Fine-Tuning)

| # | Requirement | Status | Where fulfilled | Note |
|---|---|---|---|---|
| E3.1 | New image dataset (not course-provided) | `[x]` DONE | Food-101 subset (Bossard et al., 2014) — 20–30 classes — `data/raw/food101/` | Downloaded from HuggingFace Datasets in notebook 01 |
| E3.2 | EDA on image data documented | `[ ]` TODO | `notebooks/01_eda_food101.ipynb`; `documentation.md` §2 (Block 1) | Class distribution, sample grid, size statistics |
| E3.3 | Pre-trained model fine-tuned (transfer learning) | `[ ]` TODO | `notebooks/02_train_vit_cv.ipynb`; `src/cv_model.py` | `google/vit-base-patch16-224` → fine-tune classification head |
| E3.4 | At least 2 training experiments documented | `[ ]` TODO | `notebooks/02_train_vit_cv.ipynb` Cells 6–7; `documentation.md` §3 | Experiment 1: head-only. Experiment 2: full fine-tune or different LR |
| E3.5 | Evaluation with appropriate metrics | `[ ]` TODO | `notebooks/02_train_vit_cv.ipynb` Cell 8; `documentation.md` §4 | Top-1 accuracy, per-class F1, confusion matrix heatmap |
| E3.6 | Model artifact saved and loaded in inference | `[ ]` TODO | `models/vit_food101_finetuned/` → `src/cv_model.py:CVModel.load()` | HuggingFace `save_pretrained` format; id2label JSON also saved |

---

## Bonus Requirements

> These are optional but improve the grade. Attempt in priority order.

| # | Bonus Item | Status | Where | Note |
|---|---|---|---|---|
| BX1 | **All 3 blocks combined in pipeline** | `[x]` DONE | `src/pipeline.py` | This is actually the core requirement for our project — already done |
| BX2 | **Extended evaluation** (beyond basic metrics) | `[ ]` TODO | `documentation.md` §4 (end-to-end section) | Run 20 test images end-to-end; measure error propagation CV→ML→NLP; latency |
| BX3 | **Ethical considerations** | `[ ]` TODO | `documentation.md` §1 or new §7 | Medical disclaimer, bias in Food-101 class selection, data privacy (no user photos stored) |
| BX4 | **Extensive data analysis / EDA** | `[ ]` TODO | `notebooks/01_eda_food101.ipynb`, `notebooks/03_ml_health_classifier.ipynb` | SHAP plots, correlation heatmaps, image quality analysis, class balance report |
| BX5 | **User testing / usability feedback** | `[ ]` TODO | `documentation.md` new §5b or §4 | Let 2–3 people test the app, document their feedback and any resulting UI changes |

---

## Progress Tracker

| Phase | Content | Target | Status |
|---|---|---|---|
| Phase 1 | Scaffold, repo setup, stubs, documentation skeleton | 2026-05-24 | `[x]` DONE |
| Phase 2 | CV block: EDA (nb 01) + ViT training (nb 02) + `cv_model.py` | 2026-05-28 | `[ ]` TODO |
| Phase 3 | ML block: OFF data + XGBoost (nb 03) + `ml_model.py` | 2026-05-31 | `[ ]` TODO |
| Phase 4 | NLP block: KB docs + ChromaDB (nb 04) + `nlp_rag.py` | 2026-06-02 | `[ ]` TODO |
| Phase 5 | End-to-end test, `app.py` live, HF Space deploy | 2026-06-04 | `[ ]` TODO |
| Phase 6 | Documentation fill, evaluation, submission | 2026-06-07 | `[ ]` TODO |

---

*Checklist maintained by Gianpiero Della Quila. Update after each phase by changing `[ ]` → `[~]` → `[x]` and filling in the "Where fulfilled" column.*
