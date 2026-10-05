"""Reviewer-specific supervised fine-tuning (LoRA) support.

This package holds the training-side code: SFT dataset construction with
the mandatory leakage guards, the training-configuration loader, and the
Kaggle-oriented training entry point (scripts/training/train_lora.py).
The frozen RAG pipeline is imported, never modified.
"""
