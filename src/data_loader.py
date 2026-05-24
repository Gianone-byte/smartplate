"""
SmartPlate data loading utilities.

This module provides stub functions for loading and preparing data for each
of the three AI blocks. Actual implementation happens in the training notebooks;
these stubs define the interface used by the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def download_food101_subset(
    classes: list[str],
    output_dir: str,
    split: str = "train",
    max_samples_per_class: Optional[int] = None,
) -> dict[str, list[str]]:
    """Download a subset of the Food-101 dataset from HuggingFace Datasets.

    Pulls only the specified classes from Food-101, optionally capping the
    number of samples per class. Images are saved to ``output_dir`` under
    class-named subdirectories (e.g. ``output_dir/pizza/image001.jpg``).

    Args:
        classes: List of Food-101 class names to include.
            Must be valid Food-101 labels (lowercase, underscores for spaces),
            e.g. ``["pizza", "sushi", "caesar_salad"]``.
        output_dir: Root directory where images will be saved.
            Will be created if it does not exist.
        split: Dataset split to download — ``"train"``, ``"validation"``, or
            ``"test"``. Defaults to ``"train"``.
        max_samples_per_class: If set, cap the number of images per class.
            Useful for quick experiments. ``None`` means no cap (all images).

    Returns:
        A dict mapping each class name to a list of absolute file paths of
        the downloaded images, e.g.::

            {
                "pizza": ["/data/raw/pizza/img_001.jpg", ...],
                "sushi": ["/data/raw/sushi/img_001.jpg", ...],
            }

    Raises:
        ValueError: If any class name is not a valid Food-101 label.
        OSError: If ``output_dir`` cannot be created.

    Example:
        >>> paths = download_food101_subset(
        ...     classes=["pizza", "sushi", "salad"],
        ...     output_dir="data/raw/food101",
        ...     split="train",
        ...     max_samples_per_class=100,
        ... )
        >>> len(paths["pizza"])
        100
    """
    # TODO: Implement in notebook 01_eda_food101.ipynb, then port here.
    # Suggested implementation:
    #   from datasets import load_dataset
    #   ds = load_dataset("food101", split=split)
    #   ds_subset = ds.filter(lambda x: x["label"] in class_indices)
    raise NotImplementedError("Implement in notebook 02_train_vit_cv.ipynb first.")


def load_openfoodfacts_sample(
    food_name: str,
    max_results: int = 20,
    country: str = "world",
) -> list[dict]:
    """Query the Open Food Facts API for nutritional data about a food item.

    Searches Open Food Facts by product name and returns a list of matching
    products with their nutritional information per 100g.

    Args:
        food_name: The dish or food item to search for, e.g. ``"pizza margherita"``.
            Accepts free-form text; the API handles partial matches.
        max_results: Maximum number of products to return. Defaults to 20.
        country: Country-specific endpoint to query. Defaults to ``"world"``
            (global database). Use ``"ch"`` for Swiss products.

    Returns:
        A list of product dicts. Each dict contains at minimum::

            {
                "product_name": str,
                "energy_100g": float,        # kcal per 100g
                "fat_100g": float,
                "saturated_fat_100g": float,
                "sugars_100g": float,
                "fiber_100g": float,
                "proteins_100g": float,
                "salt_100g": float,
                "nutriscore_grade": str,     # "a" through "e", or None
            }

        Returns an empty list if no products are found.

    Raises:
        requests.HTTPError: If the API request fails.

    Example:
        >>> products = load_openfoodfacts_sample("margherita pizza", max_results=5)
        >>> products[0]["energy_100g"]
        266.0
    """
    # TODO: Implement in notebook 03_ml_health_classifier.ipynb, then port here.
    # Suggested implementation:
    #   import requests
    #   url = f"https://{country}.openfoodfacts.org/cgi/search.pl"
    #   params = {"search_terms": food_name, "json": 1, "page_size": max_results}
    #   response = requests.get(url, params=params)
    #   return [_extract_nutrients(p) for p in response.json()["products"]]
    raise NotImplementedError("Implement in notebook 03_ml_health_classifier.ipynb first.")


def load_knowledge_base_documents(folder: str) -> list[dict]:
    """Load and parse all documents in the RAG knowledge base folder.

    Reads all ``.pdf``, ``.txt``, and ``.md`` files from the specified folder,
    extracts their text content, and returns a list of document dicts ready
    for chunking and embedding into ChromaDB.

    Args:
        folder: Path to the folder containing knowledge base documents.
            Typically ``data/knowledge_base/``. Subdirectories are not
            traversed — only top-level files are read.

    Returns:
        A list of document dicts, one per file::

            [
                {
                    "source": "who_healthy_diet.pdf",
                    "content": "A healthy diet helps protect against ...",
                    "metadata": {
                        "file_type": "pdf",
                        "num_pages": 4,
                        "file_size_kb": 112,
                    },
                },
                ...
            ]

        Returns an empty list if the folder contains no supported files.

    Raises:
        FileNotFoundError: If ``folder`` does not exist.
        ValueError: If a file cannot be parsed (e.g., encrypted PDF).

    Example:
        >>> docs = load_knowledge_base_documents("data/knowledge_base/")
        >>> len(docs)
        3
        >>> docs[0]["source"]
        'who_healthy_diet.pdf'
    """
    # TODO: Implement in notebook 04_rag_setup.ipynb, then port here.
    # Suggested implementation:
    #   from pypdf import PdfReader
    #   for path in Path(folder).glob("*.pdf"):
    #       reader = PdfReader(path)
    #       text = "\n".join(page.extract_text() for page in reader.pages)
    #       docs.append({"source": path.name, "content": text, ...})
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {folder}")
    raise NotImplementedError("Implement in notebook 04_rag_setup.ipynb first.")
