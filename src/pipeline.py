"""
SmartPlate end-to-end pipeline orchestrator.

Wires together Block 1 (CV), Block 2 (ML), and Block 3 (RAG) into a single
callable. This is the only module that ``app.py`` needs to import.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from src.cv_model import CVModel
from src.ml_model import MLModel, NutritionResult
from src.nlp_rag import RAGPipeline, RAGResult


@dataclass
class PipelineResult:
    """Full result of one SmartPlate pipeline run.

    Attributes:
        food_label: Predicted dish name (from CV block).
        cv_confidence: Confidence of the CV prediction (0.0–1.0).
        nutrition: Nutritional data and health score (from ML block).
        rag: Natural language explanation and tips (from RAG block).
    """

    food_label: str
    cv_confidence: float
    nutrition: NutritionResult
    rag: RAGResult


class SmartPlatePipeline:
    """Orchestrates the three-block SmartPlate pipeline.

    Instantiates and lazily loads all three models. On first call, each
    model loads its artifacts from disk. Subsequent calls reuse the
    loaded models for fast inference.

    Args:
        cv_model_dir: Override path for the ViT model directory.
        ml_model_path: Override path for the XGBoost model file.
        chroma_dir: Override path for the ChromaDB directory.
        openai_model: OpenAI model to use for generation.

    Example:
        >>> pipeline = SmartPlatePipeline()
        >>> from PIL import Image
        >>> img = Image.open("pizza.jpg")
        >>> result = pipeline.run(img)
        >>> print(result.food_label)
        pizza
        >>> print(result.rag.explanation)
        'Pizza is energy-dense with high saturated fat...'
    """

    def __init__(
        self,
        cv_model_dir: str | None = None,
        ml_model_path: str | None = None,
        chroma_dir: str | None = None,
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        self.cv = CVModel(model_dir=cv_model_dir)
        self.ml = MLModel(model_path=ml_model_path)
        self.rag = RAGPipeline(
            chroma_dir=chroma_dir,
            openai_model=openai_model,
        )

    def run(self, image: Image.Image) -> PipelineResult:
        """Run the full three-block pipeline on a single food image.

        Block execution order:
        1. CV: image → food_label + confidence
        2. ML: food_label → nutritional data + health score
        3. RAG: nutritional data → natural language explanation + tips

        Args:
            image: A PIL Image of the food item (any size; will be resized
                internally by the CV preprocessing step).

        Returns:
            A ``PipelineResult`` containing all intermediate and final outputs.

        Raises:
            NotImplementedError: Until all three blocks are implemented.
            ValueError: If CV confidence is below a threshold (configurable,
                default 0.3) — means the model is not confident enough to
                proceed with nutritional analysis.
        """
        # Block 1
        food_label, confidence = self.cv.predict(image)

        # Block 2 — uses Block 1 output
        nutrition: NutritionResult = self.ml.predict(food_label)

        # Block 3 — uses Block 2 output
        rag_result: RAGResult = self.rag.generate(nutrition)

        return PipelineResult(
            food_label=food_label,
            cv_confidence=confidence,
            nutrition=nutrition,
            rag=rag_result,
        )
