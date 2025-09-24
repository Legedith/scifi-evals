"""Optional integration test for BGE embeddings.

Enable by setting RUN_BGE_TESTS=1 to avoid large model downloads on CI/local.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pytest

from scripts.bge_similarity import encode_sentences


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_BGE_TESTS") != "1",
    reason="Set RUN_BGE_TESTS=1 to enable BGE integration tests (downloads ~1.3GB)",
)


def test_bge_embeddings_and_similarity() -> None:
    sents: List[str] = [
        "A cat sits on a mat.",
        "A dog chases the mailman.",
    ]
    embeddings = encode_sentences(sents)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0
    cos = float(np.dot(embeddings[0], embeddings[1]))
    assert -1.0 <= cos <= 1.0


