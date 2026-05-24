"""
Smoke tests for SmartPlate.

These tests verify that all src modules are importable and that the
key classes and functions are accessible. They do NOT test model logic
(models may not be trained yet) — they catch broken imports early.

Run with:
    python -m pytest tests/ -v
"""

import importlib
import pytest


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


def test_pipeline_instantiation_does_not_load_models() -> None:
    """SmartPlatePipeline() must not load any model on __init__."""
    from src.pipeline import SmartPlatePipeline
    pipeline = SmartPlatePipeline()
    # Models are loaded lazily — these should all be None at init time
    assert pipeline.cv._model is None
    assert pipeline.ml._classifier is None
    assert pipeline.rag._collection is None


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
