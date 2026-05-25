"""
Smoke tests for SmartPlate.

These tests verify that all src modules are importable and that key classes
are accessible. Model-dependent tests are skipped when artifacts are absent,
so the suite runs cheaply in CI without downloading weights or hitting APIs.

Run with:
    python -m pytest tests/ -v
"""

import importlib
import os
from pathlib import Path

import pytest


# ── Import sanity ──────────────────────────────────────────────────────────────

MODULES_TO_IMPORT = [
    "src",
    "src.cv_model",
    "src.ml_model",
    "src.nlp_rag",
    "src.pipeline",
    "src.data_loader",
]


@pytest.mark.parametrize("module_name", MODULES_TO_IMPORT)
def test_module_importable(module_name: str) -> None:
    """Each src module must be importable without errors."""
    module = importlib.import_module(module_name)
    assert module is not None


def test_cv_model_class_accessible() -> None:
    from src.cv_model import CVModel
    assert CVModel is not None


def test_ml_model_class_accessible() -> None:
    from src.ml_model import MLModel, NutritionResult
    assert MLModel is not None
    assert NutritionResult is not None


def test_rag_pipeline_class_accessible() -> None:
    from src.nlp_rag import RAGPipeline, RAGResult
    assert RAGPipeline is not None
    assert RAGResult is not None


def test_pipeline_class_accessible() -> None:
    from src.pipeline import SmartPlatePipeline, PipelineResult
    assert SmartPlatePipeline is not None
    assert PipelineResult is not None


def test_data_loader_functions_accessible() -> None:
    from src.data_loader import (
        download_food101_subset,
        load_openfoodfacts_sample,
        load_knowledge_base_documents,
    )
    assert callable(download_food101_subset)
    assert callable(load_openfoodfacts_sample)
    assert callable(load_knowledge_base_documents)


# ── Lazy-loading behaviour ─────────────────────────────────────────────────────

def test_pipeline_instantiation_does_not_load_models() -> None:
    """SmartPlatePipeline() must not load any model on __init__."""
    from src.pipeline import SmartPlatePipeline

    pipeline = SmartPlatePipeline()
    # Accessing properties creates the wrapper objects but must NOT load weights
    assert pipeline.cv._model is None
    assert pipeline.ml._classifier is None
    assert pipeline.rag._collection is None


def test_cv_model_does_not_load_on_instantiation() -> None:
    """CVModel() must have _model and _processor as None until predict() is called."""
    from src.cv_model import CVModel

    model = CVModel()
    assert model._model is None
    assert model._processor is None


def test_rag_pipeline_instantiation_does_not_require_api_key() -> None:
    """RAGPipeline() instantiation must succeed without OPENAI_API_KEY."""
    original = os.environ.pop("OPENAI_API_KEY", None)
    try:
        from src.nlp_rag import RAGPipeline

        pipeline = RAGPipeline()
        assert pipeline._collection is None
    finally:
        if original is not None:
            os.environ["OPENAI_API_KEY"] = original


# ── Dataclass correctness ──────────────────────────────────────────────────────

def test_nutrition_result_dataclass() -> None:
    """NutritionResult must be constructable with expected fields."""
    from src.ml_model import NutritionResult

    result = NutritionResult(
        food_label="pizza",
        energy_kcal=266.0,
        fat_g=10.0,
        saturated_fat_g=4.5,
        sugars_g=3.0,
        fiber_g=2.0,
        proteins_g=11.0,
        salt_g=1.2,
        health_score=35.0,
        health_label="unhealthy",
        nutriscore="d",
    )
    assert result.food_label == "pizza"
    assert result.health_label == "unhealthy"


# ── Model-dependent tests (skipped when artifacts are missing) ─────────────────

_ML_MODEL_PATH = Path(__file__).parent.parent / "models" / "health_classifier.pkl"


@pytest.mark.skipif(
    not _ML_MODEL_PATH.exists(),
    reason="health_classifier.pkl not found — run notebook 03 first",
)
def test_ml_predict_pizza_no_crash() -> None:
    """MLModel.predict('pizza') must return a well-formed result dict."""
    from src.ml_model import MLModel

    model = MLModel()
    result = model.predict("pizza")

    assert result["food_class"] == "pizza"
    assert result["health_label"] == "unhealthy"
    assert "nutrition" in result
    assert "probabilities" in result
    assert set(result["probabilities"].keys()) == {"healthy", "medium", "unhealthy"}
    assert result["nutrition"]["kcal"] == 266


@pytest.mark.skipif(
    not _ML_MODEL_PATH.exists(),
    reason="health_classifier.pkl not found — run notebook 03 first",
)
def test_ml_predict_all_classes_no_crash() -> None:
    """Every USDA class in the model must predict without error."""
    from src.ml_model import MLModel
    import joblib

    bundle = joblib.load(_ML_MODEL_PATH)
    classes = list(bundle["usda_nutrition"].keys())

    model = MLModel()
    for cls in classes:
        result = model.predict(cls)
        assert result["health_label"] in {"healthy", "medium", "unhealthy"}, (
            f"Unexpected label for {cls}: {result['health_label']}"
        )


@pytest.mark.skipif(
    not _ML_MODEL_PATH.exists(),
    reason="health_classifier.pkl not found — run notebook 03 first",
)
def test_ml_predict_unknown_class_raises() -> None:
    """MLModel.predict() must raise ValueError for unknown food classes."""
    from src.ml_model import MLModel

    model = MLModel()
    with pytest.raises(ValueError, match="Unknown food class"):
        model.predict("unicorn_steak")
