"""
Block 1: Computer Vision — ViT inference wrapper.

Loads the fine-tuned ViT model from HuggingFace Hub and classifies food images.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "Gianone/smartplate-vit-food"


class CVModel:
    """ViT food classifier loaded from HuggingFace Hub.

    Args:
        model_id: HuggingFace Hub model repository ID.
    """

    def __init__(self, model_id: str = _DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None
        self._processor = None

    def _load(self) -> None:
        """Lazy-load model and processor from HuggingFace Hub."""
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for CV inference. "
                "Run: pip install transformers torch"
            ) from exc

        logger.info("Loading CV model from %s ...", self.model_id)
        try:
            self._processor = ViTImageProcessor.from_pretrained(self.model_id)
            self._model = ViTForImageClassification.from_pretrained(self.model_id)
            self._model.eval()
            logger.info(
                "CV model loaded (%d classes)", self._model.config.num_labels
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load CV model '{self.model_id}'. "
                f"Check your internet connection and HuggingFace credentials.\n{exc}"
            ) from exc

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Classify a food image.

        Returns:
            {
                "class": str,         # top-1 food class name
                "confidence": float,  # softmax probability 0..1
                "top_5": [{"class": str, "confidence": float}, ...]
            }
        """
        if self._model is None:
            self._load()

        import torch
        import torch.nn.functional as F

        img = image.convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]

        k = min(5, probs.shape[0])
        top_values, top_indices = probs.topk(k)

        id2label = self._model.config.id2label
        top_5: List[Dict[str, Any]] = [
            {
                "class": id2label[idx.item()],
                "confidence": round(float(val.item()), 4),
            }
            for val, idx in zip(top_values, top_indices)
        ]

        return {
            "class": top_5[0]["class"],
            "confidence": top_5[0]["confidence"],
            "top_5": top_5,
        }
