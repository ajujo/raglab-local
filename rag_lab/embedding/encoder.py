"""Generate embeddings using BGE-M3 model.

Handles both dense and sparse embeddings for all chunks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from rag_lab.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_MODEL,
    STORAGE_DIR,
)
from rag_lab.exceptions import EmbeddingError

logger = logging.getLogger("rag_lab")

# Global model cache
_model_cache: Optional[object] = None


def reset_embedding_cache() -> None:
    """Reset the global embedding model cache.

    Useful for tests that need to load the model on a different device.

    Returns:
        None
    """
    global _model_cache
    _model_cache = None


def load_embedding_model(device: str = "cpu") -> object:
    """Load the BGE-M3 embedding model.

    Args:
        device: Device to run model on ('cuda' or 'cpu').

    Returns:
        The loaded BGE-M3 model.

    Raises:
        EmbeddingError: If model fails to load.
    """
    global _model_cache

    if _model_cache is not None:
        return _model_cache

    try:
        from FlagEmbedding import BGEM3FlagModel
        _model_cache = BGEM3FlagModel(
            EMBEDDING_MODEL,
            use_fp16=True,
            device=device,
        )
        logger.info(f"Loaded BGE-M3 model on {device}")
        return _model_cache
    except Exception as e:
        raise EmbeddingError(f"Failed to load BGE-M3 model: {e}")


def encode_chunks(
    chunks: List[dict],
    batch_size: int = None,
    device: str = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Generate dense and sparse embeddings for all chunks.

    Args:
        chunks: List of chunk dicts with 'text' key.
        batch_size: Batch size for processing.
        device: Device to run model on. If None, uses EMBEDDING_DEVICE from config.

    Returns:
        Tuple of (dense_embeddings, sparse_embeddings) where:
        - dense_embeddings: numpy array of shape (n_chunks, 1024)
        - sparse_embeddings: dict mapping chunk_id to sparse vector

    Raises:
        EmbeddingError: If embedding generation fails.
    """
    if not chunks:
        raise EmbeddingError("No chunks provided for embedding")

    batch_size = batch_size or EMBEDDING_BATCH_SIZE
    device = device or EMBEDDING_DEVICE

    # Extract texts
    texts = [chunk["text"] for chunk in chunks]

    try:
        model = load_embedding_model(device)

        # Process in smaller batches to avoid OOM
        all_dense = []
        all_sparse = {}

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            result = model.encode(
                batch_texts,
                batch_size=batch_size,
                max_length=EMBEDDING_MAX_LENGTH,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )

            # BGE-M3 returns: dense_vecs, lexical_weights, colbert_vecs
            # Use 'in' to check keys instead of direct comparison with arrays
            if "dense_vecs" in result:
                dense_emb = result["dense_vecs"]
                if isinstance(dense_emb, np.ndarray):
                    all_dense.append(dense_emb)
                elif hasattr(dense_emb, "numpy"):
                    all_dense.append(dense_emb.numpy())
                else:
                    all_dense.append(np.array(dense_emb))

            # Sparse embeddings are in lexical_weights (list of dicts)
            if "lexical_weights" in result:
                sparse_data = result["lexical_weights"]
                for j, chunk in enumerate(chunks[i:i + batch_size]):
                    chunk_id = chunk.get("chunk_id", str(i + j))
                    sw = sparse_data[j]
                    if isinstance(sw, dict):
                        all_sparse[chunk_id] = {int(k): float(v) for k, v in sw.items()}
                    elif isinstance(sw, list):
                        all_sparse[chunk_id] = sw
                    else:
                        all_sparse[chunk_id] = {}

        if not all_dense:
            raise EmbeddingError("No dense embeddings generated")

        dense_embeddings = np.vstack(all_dense)

        logger.info(
            f"Generated embeddings for {len(chunks)} chunks"
        )
        return dense_embeddings, all_sparse

    except Exception as e:
        raise EmbeddingError(f"Failed to generate embeddings: {e}")


def save_sparse_index(
    sparse_embeddings: Dict[str, float],
    output_path: Optional[Path] = None,
) -> Path:
    """Save sparse embeddings to a JSON file.

    Args:
        sparse_embeddings: Dict of chunk_id to sparse vector.
        output_path: Path to save the file.

    Returns:
        Path to the saved file.
    """
    output_path = output_path or (STORAGE_DIR / "sparse_index.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sparse_embeddings, f, ensure_ascii=False)

    logger.info(f"Saved sparse index to {output_path}")
    return output_path