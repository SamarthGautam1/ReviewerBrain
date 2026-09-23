"""Local review generation (LLM inference) support for ReviewerBrain.

This package only CONSUMES the validated RAG pipeline (representation,
embeddings, indexes); it never modifies it. It provides:
  - heldout_queries.py: held-out query selection + clean-record join (verified)
  - rag_retrieval.py:   top-3 per-reviewer retrieval via the validated pipeline
  - prompts.py:         versioned prompt loading/rendering
  - llm_client.py:      minimal client for a local Ollama server
  - outputs.py:         validated writers/readers for run artifacts
"""
