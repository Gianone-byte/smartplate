"""
Block 1: Computer Vision — ViT inference wrapper.

Loads the fine-tuned ViT model and exposes a single ``predict`` function
that takes a PIL image and returns the predicted food class label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image


MODEL_DIR = Path(__file__).parent.parent / "models" / "vit_food101_finetuned"


class CVModel:
    """Inference wrapper around the fine-tuned ViT food classifier.

    Args:
        model_dir: Path to the directory containing the saved ViT model
            (``config.json``, ``model.safetensors``, ``preprocessor_config.json``).
            Defaults to ``models/vit_food101_finetuned/``.

    Example:
        >>> model = CVModel()
        >>> from PIL import Image
        >>> img = Image.open("pizza.jpg")
        >>> label, confidence = model.predict(img)
        >>> print(label, confidence)
        pizza 0.94
    """

    def __init__(self, model_dir: Optional[str] = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load model and processor from disk into memory.

        Called lazily on first ``predict`` call, or explicitly for warm-up.

        Raises:
            FileNotFoundError: If ``model_dir`` does not exist.
        """
        # TODO: Implement after training notebook 02 is complete.
        # from transformers import ViTForImageClassification, ViTImageProcessor
        # self._processor = ViTImageProcessor.from_pretrained(self.model_dir)
        # self._model = ViTForImageClassification.from_pretrained(self.model_dir)
        # self._model.eval()
        raise NotImplementedError("Train the model first via notebook 02_train_vit_cv.ipynb.")

    def predict(self, image: Image.Image) -> tuple[str, float]:
        """Classify a food image and return the top-1 label with confidence.

        Args:
            image: A PIL Image (any mode; will be converted to RGB internally).

        Returns:
            A tuple ``(label, confidence)`` where ``label`` is the predicted
            Food-101 class name (e.g. ``"pizza"``) and ``confidence`` is the
            softmax probability of the top prediction (0.0–1.0).

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self._model is None:
            self.load()
        raise NotImplementedError("Implement after training is complete.")
