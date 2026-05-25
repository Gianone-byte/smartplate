"""
SmartPlate end-to-end pipeline orchestrator.

Wires together Block 1 (CV), Block 2 (ML), and Block 3 (RAG) via lazy-loading
properties. Blocks are instantiated on first access; models load on first call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from PIL import Image

from src.cv_model import CVModel
from src.ml_model import MLModel
from src.nlp_rag import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Full result of one SmartPlate pipeline run — kept for backward compatibility."""

    cv_result: Dict[str, Any]
    ml_result: Dict[str, Any]
    nlp_result: Dict[str, Any]


class SmartPlatePipeline:
    """Orchestrates the three-block SmartPlate pipeline with lazy loading.

    Blocks are instantiated on first property access; model weights load on
    first inference call. This keeps the import fast and cold-start cheap.

    Example:
        >>> pipeline = SmartPlatePipeline()
        >>> result = pipeline.process(image)
        >>> print(result["cv_result"]["class"])
        pizza
    """

    def __init__(self) -> None:
        self._cv: Optional[CVModel] = None
        self._ml: Optional[MLModel] = None
        self._rag: Optional[RAGPipeline] = None

    @property
    def cv(self) -> CVModel:
        if self._cv is None:
            self._cv = CVModel()
        return self._cv

    @property
    def ml(self) -> MLModel:
        if self._ml is None:
            self._ml = MLModel()
        return self._ml

    @property
    def rag(self) -> RAGPipeline:
        if self._rag is None:
            self._rag = RAGPipeline()
        return self._rag

    def process(
        self,
        image: Image.Image,
        user_question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full three-block pipeline on a food image.

        Args:
            image: PIL Image of the food item.
            user_question: Optional follow-up question from the user.

        Returns:
            {
                "image": PIL.Image,
                "cv_result":  {"class": str, "confidence": float, "top_5": list},
                "ml_result":  {"food_class": str, "nutrition": dict,
                               "health_label": str, "probabilities": dict},
                "nlp_result": {"answer": str, "sources": list, "tokens": int}
            }
        """
        logger.info("Running CV block ...")
        cv_result = self.cv.predict(image)
        food_class: str = cv_result["class"]

        logger.info("Running ML block for class: %s", food_class)
        ml_result = self.ml.predict(food_class)

        logger.info("Running RAG block ...")
        nlp_result = self.rag.answer(
            food_class=food_class,
            kcal=float(ml_result["nutrition"]["kcal"]),
            health_label=ml_result["health_label"],
            user_question=user_question,
        )

        return {
            "image": image,
            "cv_result": cv_result,
            "ml_result": ml_result,
            "nlp_result": nlp_result,
        }
