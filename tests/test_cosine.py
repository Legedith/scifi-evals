"""Unit tests for cosine similarity helper."""

from __future__ import annotations

import numpy as np

from scripts.bge_similarity import cosine_similarity


def test_cosine_equal_vectors() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6


def test_cosine_orthogonal() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b) - 0.0) < 1e-6


def test_cosine_opposite() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b) + 1.0) < 1e-6


