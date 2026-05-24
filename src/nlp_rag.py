"""
Block 3: NLP/RAG — retrieval-augmented generation for nutrition advice.

Takes the nutritional analysis from Block 2 and generates a personalized,
evidence-grounded explanation using ChromaDB retrieval + OpenAI gpt-4o-mini.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.ml_model import NutritionResult


CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

SYSTEM_PROMPT = """\
You are SmartPlate, a friendly AI nutrition assistant. You explain meal analysis
results in clear, non-judgmental language and give actionable tips based on
WHO and DGE dietary guidelines.

Always:
- Ground your advice in the retrieved guideline excerpts
- Be concise (3-5 sentences max per section)
- Avoid medical advice; recommend consulting a professional for health concerns
- Use simple, everyday language (no jargon)
"""


@dataclass
class RAGResult:
    """Output of the NLP/RAG block — the final pipeline output.

    Attributes:
        explanation: Natural language summary of the meal's nutritional quality.
        tips: Personalized dietary tips based on the analysis.
        sources: List of source document names used for retrieval.
        retrieved_chunks: The raw text chunks retrieved from ChromaDB
            (useful for debugging and transparency).
    """

    explanation: str
    tips: str
    sources: list[str]
    retrieved_chunks: list[str]


class RAGPipeline:
    """ChromaDB-backed RAG pipeline with OpenAI generation.

    Args:
        chroma_dir: Path to the ChromaDB persistence directory.
            Defaults to ``chroma_db/`` in the project root.
        openai_model: OpenAI model ID for generation. Defaults to
            ``"gpt-4o-mini"`` (fast and cost-efficient).
        top_k: Number of chunks to retrieve per query. Defaults to 3.

    Example:
        >>> pipeline = RAGPipeline()
        >>> from src.ml_model import NutritionResult
        >>> result_ml = NutritionResult(
        ...     food_label="pizza", energy_kcal=266, fat_g=10, ...
        ... )
        >>> rag_result = pipeline.generate(result_ml)
        >>> print(rag_result.explanation)
        'Pizza is high in saturated fat and sodium...'
    """

    def __init__(
        self,
        chroma_dir: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        top_k: int = 3,
    ) -> None:
        self.chroma_dir = Path(chroma_dir) if chroma_dir else CHROMA_DIR
        self.openai_model = openai_model
        self.top_k = top_k
        self._collection = None
        self._client = None

    def load(self) -> None:
        """Initialize ChromaDB client and OpenAI client.

        Reads ``OPENAI_API_KEY`` from environment. Loads the existing
        ChromaDB collection — run notebook 04 first to build it.

        Raises:
            EnvironmentError: If ``OPENAI_API_KEY`` is not set.
            FileNotFoundError: If the ChromaDB directory does not exist.
        """
        # TODO: Implement after notebook 04_rag_setup.ipynb is complete.
        # import chromadb
        # from openai import OpenAI
        # api_key = os.getenv("OPENAI_API_KEY")
        # if not api_key:
        #     raise EnvironmentError("OPENAI_API_KEY not set in environment.")
        # self._client = OpenAI(api_key=api_key)
        # chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        # self._collection = chroma_client.get_collection("nutrition_guidelines")
        raise NotImplementedError("Build the knowledge base first via notebook 04_rag_setup.ipynb.")

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve the top-k most relevant chunks from ChromaDB.

        Args:
            query: The search query, typically constructed from the food label
                and health score (e.g. ``"pizza high fat unhealthy tips"``).

        Returns:
            A list of result dicts with keys ``"document"``, ``"source"``,
            and ``"distance"``.
        """
        if self._collection is None:
            self.load()
        raise NotImplementedError("Implement after RAG setup is complete.")

    def generate(self, nutrition: NutritionResult) -> RAGResult:
        """Generate a natural language explanation and tips for a meal.

        Constructs a retrieval query from the nutrition data, fetches relevant
        guideline excerpts, then calls the LLM with a structured prompt that
        includes both the nutritional facts and the retrieved context.

        Args:
            nutrition: The ``NutritionResult`` output from Block 2, containing
                the dish label, nutritional values, and health score.

        Returns:
            A ``RAGResult`` with the LLM-generated explanation and tips,
            plus source attribution for transparency.

        Raises:
            RuntimeError: If the pipeline has not been loaded.
        """
        if self._collection is None:
            self.load()
        raise NotImplementedError("Implement after training and RAG setup are complete.")
