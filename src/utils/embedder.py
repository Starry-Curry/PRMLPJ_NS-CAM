from typing import List
from src.utils.logging_setup import get_logger
from src.utils import config
import os

logger = get_logger(__name__)

try:
    # Check if package is installed but don't import yet to save time/memory
    import importlib.util
    spec = importlib.util.find_spec("sentence_transformers")
    S_ENTENCE_AVAILABLE = spec is not None
except Exception:
    S_ENTENCE_AVAILABLE = False


class Embedder:
    def __init__(self, local_path: str = None):
        self.local_path = local_path or config.LOCAL_EMBEDDING_PATH
        self.model = None
        
        # Only try to load if available
        if S_ENTENCE_AVAILABLE and self.local_path and os.path.exists(self.local_path):
            try:
                logger.info(f"Loading local embedding model from {self.local_path}")
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.local_path)
            except Exception as e:
                logger.warning(f"Failed to load local embedding model: {e}")
                self.model = None
        else:
            if not S_ENTENCE_AVAILABLE:
                logger.debug("sentence-transformers not installed; local embedding not available")
            else:
                logger.debug(f"Local embedding path not found or set: {self.local_path}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.model:
            try:
                return self.model.encode(texts, show_progress_bar=False).tolist()
            except Exception as e:
                logger.warning(f"Local embedder failed: {e}")

        # Fallback: simple hash-based mock embeddings (deterministic)
        logger.info("Using fallback mock embeddings")
        out = []
        for t in texts:
            # simple deterministic pseudo-embedding
            h = abs(hash(t)) % (10**8)
            vec = [(h >> (i % 32)) % 100 / 100.0 for i in range(384)]
            out.append(vec)
        return out


_GLOBAL_EMBEDDER = None

def get_global_embedder():
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None:
        _GLOBAL_EMBEDDER = Embedder()
    return _GLOBAL_EMBEDDER

def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_global_embedder().embed(texts)
