"""
Block 3: NLP/RAG — retrieval-augmented generation for nutrition advice.

Builds or loads a ChromaDB vector store from the pre-computed rag_chunks.json
(or falls back to parsing PDFs) and generates nutrition advice via OpenAI.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent

# SmartPlate Production Prompt — Strategy 1 (Basic) from Notebook 04, cell 18
_SYSTEM_PROMPT = """\
You are SmartPlate, a friendly AI nutrition coach.
- Be evidence-based: use the provided context
- Be balanced: acknowledge enjoyment, explain trade-offs
- Avoid moralizing language ("bad", "forbidden")
- Suggest practical alternatives when appropriate
- Be concise: 3-5 sentences max"""


@dataclass
class RAGResult:
    """RAG result — kept for backward compatibility with tests."""

    answer: str
    sources: List[str]
    tokens: int


class RAGPipeline:
    """ChromaDB-backed RAG pipeline with OpenAI generation.

    On first call, either loads an existing ChromaDB from ``persist_dir`` or
    builds a fresh in-memory collection from ``models/rag_chunks.json``.
    Falls back to parsing PDFs from ``knowledge_base_dir`` if the JSON is absent.

    Args:
        knowledge_base_dir: Path to PDF knowledge base (fallback when JSON missing).
        persist_dir: Path to an existing ChromaDB persistence directory to load.
        embedding_model: sentence-transformers model ID for encoding.
        llm_model: OpenAI model ID for generation.
        top_k: Number of chunks to retrieve per query.
    """

    def __init__(
        self,
        knowledge_base_dir: Optional[str] = None,
        persist_dir: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
        top_k: int = 3,
    ) -> None:
        self.knowledge_base_dir = (
            Path(knowledge_base_dir)
            if knowledge_base_dir
            else _PROJECT_ROOT / "data" / "knowledge_base"
        )
        self.persist_dir = (
            Path(persist_dir) if persist_dir else _PROJECT_ROOT / "chroma_db"
        )
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.top_k = top_k

        self._collection = None
        self._embed_model = None
        self._openai = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Initialize ChromaDB and OpenAI client (called lazily)."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. "
                "Add it to .env or set it as an environment variable."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package required. Run: pip install openai") from exc

        self._openai = OpenAI(api_key=api_key)

        # Try loading from persistent ChromaDB cache first
        if self.persist_dir.exists() and any(self.persist_dir.iterdir()):
            if self._try_load_from_cache():
                return

        # Build fresh from chunks
        self._build_collection()

    def _try_load_from_cache(self) -> bool:
        """Attempt to load existing ChromaDB. Returns True if successful."""
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.persist_dir))
            col = client.get_collection("smartplate_nutrition")
            if col.count() > 0:
                self._collection = col
                logger.info(
                    "Loaded ChromaDB cache from %s (%d chunks)",
                    self.persist_dir,
                    col.count(),
                )
                self._embed_model = self._make_embed_model()
                return True
        except Exception as exc:
            logger.warning("Could not load ChromaDB cache: %s. Rebuilding.", exc)
        return False

    def _make_embed_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers required. Run: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model %s ...", self.embedding_model)
        return SentenceTransformer(self.embedding_model)

    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks from rag_chunks.json, or fall back to parsing PDFs."""
        chunks_path = _PROJECT_ROOT / "models" / "rag_chunks.json"
        if chunks_path.exists():
            logger.info("Loading chunks from %s", chunks_path)
            with open(chunks_path, encoding="utf-8") as f:
                return json.load(f)

        if self.knowledge_base_dir.exists():
            logger.info("Parsing PDFs from %s ...", self.knowledge_base_dir)
            return self._parse_pdfs()

        raise FileNotFoundError(
            f"Neither models/rag_chunks.json nor knowledge base dir '{self.knowledge_base_dir}' "
            "found. Run notebook 04_rag_setup.ipynb first."
        )

    def _parse_pdfs(self) -> List[Dict[str, Any]]:
        """Parse PDFs into chunks — same logic as Notebook 04 cell 7."""
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("pypdf required. Run: pip install pypdf") from exc

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            split_fn = splitter.split_text
        except ImportError:
            def split_fn(text: str) -> List[str]:
                # Simple fallback: split every 500 chars with 100-char overlap
                chunks = []
                step = 400
                for i in range(0, len(text), step):
                    chunks.append(text[i: i + 500])
                return chunks

        chunks = []
        for pdf_path in sorted(self.knowledge_base_dir.glob("*.pdf")):
            try:
                reader = PdfReader(str(pdf_path))
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if not (text and text.strip()):
                        continue
                    for i, chunk_text in enumerate(split_fn(text.strip())):
                        chunks.append({
                            "id": f"{pdf_path.name}_p{page_num}_c{i}",
                            "source": pdf_path.name,
                            "page": page_num,
                            "chunk_idx": i,
                            "text": chunk_text,
                        })
            except Exception as exc:
                logger.warning("Could not parse %s: %s", pdf_path.name, exc)

        return chunks

    def _build_collection(self) -> None:
        """Build an in-memory ChromaDB collection from chunks."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise ImportError("chromadb required. Run: pip install chromadb") from exc

        chunks = self._load_chunks()
        self._embed_model = self._make_embed_model()

        logger.info("Generating embeddings for %d chunks ...", len(chunks))
        texts = [c["text"] for c in chunks]
        embeddings = self._embed_model.encode(
            texts, show_progress_bar=False, batch_size=32
        )

        client = chromadb.Client(Settings(anonymized_telemetry=False))
        try:
            client.delete_collection("smartplate_nutrition")
        except Exception:
            pass

        self._collection = client.create_collection("smartplate_nutrition")
        self._collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[
                {"source": c["source"], "page": c["page"]} for c in chunks
            ],
            ids=[c["id"] for c in chunks],
        )
        logger.info(
            "ChromaDB built in-memory with %d chunks", self._collection.count()
        )

    def _retrieve(self, query: str) -> Dict[str, Any]:
        query_emb = self._embed_model.encode([query]).tolist()
        return self._collection.query(
            query_embeddings=query_emb,
            n_results=self.top_k,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        food_class: str,
        kcal: float,
        health_label: str,
        user_question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate nutrition advice grounded in retrieved guidelines.

        Uses the SmartPlate Production Prompt (Notebook 04, cell 18).

        Args:
            food_class: Food class name from CVModel (e.g. ``"pizza"``).
            kcal: Estimated kcal per 100g from MLModel.
            health_label: Health category from MLModel (``"healthy"`` / ``"medium"`` / ``"unhealthy"``).
            user_question: Optional follow-up question from the user.

        Returns:
            {"answer": str, "sources": [str], "tokens": int}
        """
        if self._collection is None:
            self._load()

        if user_question is None or not user_question.strip():
            user_question = (
                f"Tell me about {food_class.replace('_', ' ')} — is it healthy?"
            )

        results = self._retrieve(user_question)
        context = "\n\n".join(
            f"[Source: {m['source']}, page {m['page']}]\n{doc}"
            for doc, m in zip(
                results["documents"][0], results["metadatas"][0]
            )
        )
        sources = [
            f"{m['source']} (p.{m['page']})"
            for m in results["metadatas"][0]
        ]

        user_prompt = (
            f"The user uploaded a photo of: **{food_class.replace('_', ' ')}**\n\n"
            f"📷 Vision identifies: {food_class}\n"
            f"🍽 Nutrition (per 100g): ~{kcal:.0f} kcal\n"
            f"🏷  Health classification: {health_label}\n\n"
            f"Reference guidelines:\n{context}\n\n"
            f"User asks: {user_question}"
        )

        try:
            response = self._openai.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=300,
                temperature=0.5,
            )
            answer_text = response.choices[0].message.content
            tokens = response.usage.total_tokens
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc)
            answer_text = (
                "I couldn't generate advice right now. "
                "Please check your API key or try again later."
            )
            tokens = 0

        return {"answer": answer_text, "sources": sources, "tokens": tokens}
