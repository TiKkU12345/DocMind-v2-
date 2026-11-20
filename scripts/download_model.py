"""Run at build time so the embedding model ships with the deploy instead of
being re-downloaded on every cold start."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import embedding  # noqa: E402

if __name__ == "__main__":
    embedding.warmup()
    print("Embedding model cached.")
