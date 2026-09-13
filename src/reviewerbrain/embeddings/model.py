"""Embedding model constants and loader for ReviewerBrain RAG."""

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
# all-MiniLM-L6-v2 truncates inputs at 256 wordpiece tokens; the
# representation layer budgets against this window (see retrieval.representation).


def load_model():
    """Load the frozen embedding model on CPU."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device="cpu")
