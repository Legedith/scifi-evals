"""CLI to compute similarity between two sentences using BGE embeddings.

Usage:
  uv run scripts/bge_similarity.py --s1 "cats sit on mats" --s2 "dogs bark at mailmen"

Notes:
- Uses `BAAI/bge-large-en-v1.5` with normalized embeddings and cosine similarity.
- For retrieval queries, you can enable `--query-instruction` to prepend the
  recommended instruction to sentence 1 (query) before encoding.
"""

from __future__ import annotations

import sys
from typing import List

import numpy as np
import typer


app = typer.Typer(add_completion=False, no_args_is_help=True)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return cosine similarity between two 1-D vectors."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("cosine_similarity expects 1-D vectors")
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def encode_sentences(
    sentences: List[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    normalize: bool = True,
) -> np.ndarray:
    """Encode sentences with BGE and return embeddings as np.ndarray."""
    # Lazy import to avoid heavy dependency during test collection
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(
        sentences,
        batch_size=2,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )


@app.command()
def main(
    s1: str = typer.Option(
        ..., "--s1", help="First sentence (e.g., a query)", prompt=True
    ),
    s2: str = typer.Option(
        ..., "--s2", help="Second sentence (e.g., a document)", prompt=True
    ),
    query_instruction: bool = typer.Option(
        False,
        "--query-instruction/--no-query-instruction",
        help=(
            "Prepend BGE's recommended query instruction to s1 for retrieval: "
            "'Represent this sentence for searching relevant passages: '")
    ),
    model_name: str = typer.Option(
        "BAAI/bge-large-en-v1.5", "--model", help="SentenceTransformer model name"
    ),
) -> None:
    """Compute and print cosine similarity between the two embeddings."""
    if query_instruction:
        prefix = "Represent this sentence for searching relevant passages: "
        s1_used = prefix + s1
    else:
        s1_used = s1

    sentences = [s1_used, s2]
    embeddings = encode_sentences(sentences, model_name=model_name, normalize=True)

    # Cosine is dot product for normalized embeddings; compute both for clarity.
    cos = float(np.dot(embeddings[0], embeddings[1]))
    cos_check = cosine_similarity(embeddings[0], embeddings[1])

    typer.echo(
        f"Model: {model_name}\n"
        f"Sentence 1: {s1}\n"
        f"Sentence 2: {s2}\n"
        f"Cosine similarity: {cos:.4f} (check: {cos_check:.4f})"
    )


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


