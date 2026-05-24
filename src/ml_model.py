"""
Block 2: ML Health Classifier — Open Food Facts lookup + XGBoost scoring.

Takes a food label from Block 1, fetches nutritional data from Open Food Facts,
and returns a health score and category using the trained XGBoost model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


MODEL_PATH = Path(__file__).parent.parent / "models" / "xgb_health_classifier.pkl"


@dataclass
class NutritionResult:
    """Output of the ML block, passed as input to Block 3 (RAG).

    Attributes:
        food_label: The dish name from Block 1 (e.g. ``"pizza"``).
        energy_kcal: Energy per 100g in kcal.
        fat_g: Total fat per 100g in grams.
        saturated_fat_g: Saturated fat per 100g in grams.
        sugars_g: Total sugars per 100g in grams.
        fiber_g: Dietary fiber per 100g in grams.
        proteins_g: Protein per 100g in grams.
        salt_g: Salt per 100g in grams.
        health_score: Numeric health score 0–100 (higher = healthier).
        health_label: Human-readable category: ``"healthy"``, ``"moderate"``,
            or ``"unhealthy"``.
        nutriscore: Nutri-Score grade (``"a"`` through ``"e"``), or ``None``
            if not available.
    """

    food_label: str
    energy_kcal: float
    fat_g: float
    saturated_fat_g: float
    sugars_g: float
    fiber_g: float
    proteins_g: float
    salt_g: float
    health_score: float
    health_label: str
    nutriscore: Optional[str] = None


class MLModel:
    """Open Food Facts lookup and XGBoost health classifier.

    Args:
        model_path: Path to the saved XGBoost model (``.pkl`` file).
            Defaults to ``models/xgb_health_classifier.pkl``.

    Example:
        >>> model = MLModel()
        >>> result = model.predict("pizza")
        >>> print(result.health_label, result.health_score)
        unhealthy 28.5
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self._classifier = None

    def load(self) -> None:
        """Load the XGBoost classifier from disk.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
        """
        # TODO: Implement after training notebook 03 is complete.
        # import pickle
        # with open(self.model_path, "rb") as f:
        #     self._classifier = pickle.load(f)
        raise NotImplementedError("Train the model first via notebook 03_ml_health_classifier.ipynb.")

    def predict(self, food_label: str) -> NutritionResult:
        """Fetch nutritional data and compute a health score for a given dish.

        Pipeline:
        1. Query Open Food Facts API for ``food_label``
        2. Average nutritional values across top matching products
        3. Run XGBoost classifier to produce health score and label

        Args:
            food_label: Dish name as returned by Block 1 (e.g. ``"pizza"``).

        Returns:
            A ``NutritionResult`` dataclass with nutritional values and
            health classification.

        Raises:
            ValueError: If no matching products are found in Open Food Facts.
            RuntimeError: If the model has not been loaded.
        """
        if self._classifier is None:
            self.load()
        raise NotImplementedError("Implement after training is complete.")
