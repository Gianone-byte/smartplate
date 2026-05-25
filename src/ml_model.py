"""
Block 2: ML Health Classifier — USDA nutrition lookup + Logistic Regression scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "health_classifier.pkl"


@dataclass
class NutritionResult:
    """Nutrition result dataclass — kept for backward compatibility with tests.

    Attributes mirror the USDA-based nutritional values per 100g.
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
    """USDA nutrition lookup + Logistic Regression health classifier.

    The model bundle (``health_classifier.pkl``) contains:
    - ``model``: fitted LogisticRegression
    - ``scaler``: fitted StandardScaler
    - ``label_encoder``: fitted LabelEncoder (healthy / medium / unhealthy)
    - ``feature_cols``: list of 16 feature column names
    - ``usda_nutrition``: dict of curated nutrition data per food class

    Args:
        model_path: Override path to the ``.pkl`` bundle.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._classifier: Optional[Dict[str, Any]] = None

    def _load(self) -> None:
        """Lazy-load the model bundle from disk using joblib."""
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "joblib is required. Run: pip install joblib"
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found: {self.model_path}. "
                "Run notebook 03_ml_health_classifier.ipynb first."
            )

        logger.info("Loading ML model from %s ...", self.model_path)
        self._classifier = joblib.load(self.model_path)
        logger.info(
            "ML model loaded (type: %s, accuracy: %.4f)",
            self._classifier.get("model_type"),
            self._classifier.get("test_accuracy", 0),
        )

    def _build_features(self, nutrition: Dict[str, float]) -> np.ndarray:
        """Compute 16 model features from 8 base USDA nutrients.

        Feature engineering mirrors Notebook 03 (cell 8 / cell 17).
        """
        kcal = nutrition["kcal"]
        fat = nutrition["fat"]
        sat_fat = nutrition["sat_fat"]
        carbs = nutrition["carbs"]
        sugar = nutrition["sugar"]
        fiber = nutrition["fiber"]
        protein = nutrition["protein"]
        salt = nutrition["salt"]

        feature_map: Dict[str, float] = {
            "kcal": kcal,
            "fat": fat,
            "sat_fat": sat_fat,
            "carbs": carbs,
            "sugar": sugar,
            "fiber": fiber,
            "protein": protein,
            "salt": salt,
            "sugar_to_carb_ratio": sugar / (carbs + 1e-6),
            "sat_fat_pct_of_fat": sat_fat / (fat + 1e-6),
            "calorie_density": kcal / 100,
            "protein_to_kcal": protein * 4 / (kcal + 1e-6),
            "fiber_to_carb_ratio": fiber / (carbs + 1e-6),
            "high_sugar": float(sugar > 15),
            "high_salt": float(salt > 1.5),
            "high_sat_fat": float(sat_fat > 5),
        }

        cols = self._classifier["feature_cols"]
        return np.array([[feature_map[c] for c in cols]])

    def predict(self, food_class: str) -> Dict[str, Any]:
        """Look up USDA nutrition and predict health label.

        Args:
            food_class: Food class name as returned by CVModel (e.g. ``"pizza"``).

        Returns:
            {
                "food_class": str,
                "nutrition": {"kcal": float, "fat": float, "sat_fat": float,
                              "carbs": float, "sugar": float, "fiber": float,
                              "protein": float, "salt": float},
                "health_label": str,   # "healthy" | "medium" | "unhealthy"
                "probabilities": {"healthy": float, "medium": float, "unhealthy": float}
            }

        Raises:
            ValueError: If ``food_class`` is not in the USDA nutrition table.
            FileNotFoundError: If the model bundle is missing.
        """
        if self._classifier is None:
            self._load()

        bundle = self._classifier
        usda: Dict[str, Dict[str, float]] = bundle["usda_nutrition"]

        normalized = food_class.lower().replace(" ", "_").replace("-", "_")
        if normalized not in usda:
            raise ValueError(
                f"Unknown food class: '{food_class}'. "
                f"Supported: {sorted(usda.keys())}"
            )

        nutrition = usda[normalized]
        X = self._build_features(nutrition)
        X_scaled = bundle["scaler"].transform(X)

        pred_idx = bundle["model"].predict(X_scaled)[0]
        proba = bundle["model"].predict_proba(X_scaled)[0]

        le = bundle["label_encoder"]
        health_label = str(le.inverse_transform([pred_idx])[0])
        probabilities = {
            str(cls): round(float(p), 4)
            for cls, p in zip(le.classes_, proba)
        }

        return {
            "food_class": normalized,
            "nutrition": nutrition,
            "health_label": health_label,
            "probabilities": probabilities,
        }
