"""End-to-end smoke test: head-tail + CNN identity plugged into a synthetic tracking run."""

from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.runtime.resolver import ResolvedBackend


@pytest.mark.slow
def test_tracking_smoke_headtail_and_cnn_identity(
    tmp_path,
    tiny_flat_headtail,
    tiny_flat_subset,
):
    """Sanity check — head-tail + CNN identity both load + run against a tiny synthetic video."""
    from hydra_suite.core.identity.classification.cnn import (
        CNNIdentityBackend,
        CNNIdentityConfig,
    )
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    headtail = HeadTailAnalyzer(
        model_path=str(tiny_flat_headtail),
        resolved=ResolvedBackend("torch", "cpu", False),
    )
    cnn_cfg = CNNIdentityConfig(
        model_path=str(tiny_flat_subset),
        confidence=0.0,
        scoring_mode="atomic",
    )
    cnn_backend = CNNIdentityBackend(
        cnn_cfg,
        model_path=str(tiny_flat_subset),
        resolved=ResolvedBackend("torch", "cpu", False),
    )

    crops = [np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8) for _ in range(5)]
    ht = headtail.predict_labels(crops)
    cnn = cnn_backend.predict_batch(crops)
    assert len(ht) == 5
    assert len(cnn) == 5

    headtail.close()
    cnn_backend.close()
