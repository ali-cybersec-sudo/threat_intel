"""rag_engine.py
================

Retrieval-Augmented Generation engine for the CTI Multi-Agent System.

Uses ChromaDB as the vector store and sentence-transformers for
embedding generation.  Supports:
* Ingesting raw text and PDF files from ``data/raw_reports/``.
* Semantic retrieval of the top-k most relevant chunks.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGEngine:
    """Retrieval-Augmented Generation engine backed by ChromaDB.

    Parameters
    ----------
    config : dict
        Tool configuration from ``settings.yaml``.  Recognised keys:

        * ``provider`` – currently only ``"chromadb"`` is supported.
        * ``collection_name`` – ChromaDB collection (default ``"cti_reports"``).
        * ``persist_dir`` – directory for persistent storage (default
          ``"./data/chroma_db"``).
        * ``embedding_model`` – HuggingFace model name for embeddings
          (default ``"all-MiniLM-L6-v2"``).
        * ``chunk_size`` – characters per text chunk (default 1000).
        * ``chunk_overlap`` – overlap between chunks (default 200).
        * ``top_k`` – number of results to return (default 5).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.collection_name: str = self.config.get("collection_name", "cti_reports")
        self.persist_dir: str = self.config.get("persist_dir", "./data/chroma_db")
        self.embedding_model: str = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self.chunk_size: int = int(self.config.get("chunk_size", 1000))
        self.chunk_overlap: int = int(self.config.get("chunk_overlap", 200))
        self.top_k: int = int(self.config.get("top_k", 5))

        # Lazy-loaded heavy dependencies
        self._collection: Any = None
        self._embed_fn: Any = None

        logger.info(
            "RAGEngine initialised (collection=%s, persist=%s).",
            self.collection_name, self.persist_dir,
        )

    # =====================================================================
    # Lazy initialisation
    # =====================================================================

    def _init_chroma(self) -> None:
        """Initialise the ChromaDB client and collection on first use."""
        if self._collection is not None:
            return

        try:
            import chromadb
            # Using PersistentClient for modern ChromaDB API
        except ImportError:
            logger.error("chromadb package not installed. Run: pip install chromadb")
            raise

        os.makedirs(self.persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready (%d docs).", self.collection_name, self._collection.count())

    def _get_embed_fn(self) -> Any:
        """Return a SentenceTransformer model, loading it lazily."""
        if self._embed_fn is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
                raise
            self._embed_fn = SentenceTransformer(self.embedding_model)
            logger.info("Loaded embedding model: %s", self.embedding_model)
        return self._embed_fn

    # =====================================================================
    # Text chunking
    # =====================================================================

    def _chunk_text(self, text: str) -> List[str]:
        """Split *text* into overlapping chunks."""
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return [c.strip() for c in chunks if c.strip()]

    # =====================================================================
    # Ingestion
    # =====================================================================

    def ingest_text(self, text: str, metadata: Optional[Dict[str, str]] = None) -> int:
        """Chunk and store *text* in the vector database.

        Returns the number of chunks stored.
        """
        self._init_chroma()
        model = self._get_embed_fn()
        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        embeddings = model.encode(chunks, show_progress_bar=False).tolist()
        ids = [hashlib.sha256(c.encode()).hexdigest()[:16] for c in chunks]
        metas = [metadata or {} for _ in chunks]

        self._collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metas,
        )
        logger.info("Ingested %d chunks.", len(chunks))
        return len(chunks)

    def ingest_pdf(self, pdf_path: str) -> int:
        """Extract text from a PDF and ingest it.

        Requires the ``pypdf`` package.
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.error("pypdf not installed. Run: pip install pypdf")
            return 0

        path = Path(pdf_path)
        if not path.is_file():
            logger.warning("PDF not found: %s", pdf_path)
            return 0

        reader = PdfReader(str(path))
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return self.ingest_text(full_text, metadata={"source": path.name})

    def ingest_directory(self, directory: str = "./data/raw_reports") -> int:
        """Ingest all ``.txt`` and ``.pdf`` files in *directory*."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.warning("Directory not found: %s", directory)
            return 0

        total = 0
        for file_path in sorted(dir_path.iterdir()):
            if file_path.suffix.lower() == ".pdf":
                total += self.ingest_pdf(str(file_path))
            elif file_path.suffix.lower() in (".txt", ".md"):
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                total += self.ingest_text(text, metadata={"source": file_path.name})

        logger.info("Ingested %d total chunks from %s.", total, directory)
        return total

    # =====================================================================
    # Retrieval
    # =====================================================================

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Return the top-k most relevant text chunks for *query*.

        Parameters
        ----------
        query : str
            Natural-language search query.
        top_k : int | None
            Override the default ``top_k`` from config.

        Returns
        -------
        list[str]
            Ranked list of text chunks.
        """
        self._init_chroma()

        if self._collection.count() == 0:
            logger.info("RAG collection is empty; nothing to retrieve.")
            return []

        model = self._get_embed_fn()
        k = top_k or self.top_k
        query_embedding = model.encode([query], show_progress_bar=False).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self._collection.count()),
        )

        documents = results.get("documents", [[]])[0]
        logger.info("RAG retrieved %d chunks for: %s", len(documents), query)
        return documents

    def __repr__(self) -> str:
        return (
            f"RAGEngine(collection={self.collection_name}, "
            f"persist={self.persist_dir}, top_k={self.top_k})"
        )