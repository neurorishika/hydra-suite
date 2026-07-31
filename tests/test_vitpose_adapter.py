# tests/test_vitpose_adapter.py
from pathlib import Path

import torch

from hydra_suite.core.identity.pose.vitpose.adapter import (
    FinetuneMeta,
    infer_head_from_state,
    load_finetuned_checkpoint,
)
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose


def _save_training_ckpt(tmp_path: Path, variant: str, head: str, k: int) -> Path:
    model = build_vitpose(variant, head, num_keypoints=k)
    ckpt = {
        "model_state": model.state_dict(),
        "optim_state": {},
        "variant": variant,
        "num_keypoints": k,
        "epoch": 3,
        "pck": 0.5,
        "sched_state": {},
    }
    p = tmp_path / "best.pt"
    torch.save(ckpt, p)
    return p


def test_infer_head_classic_vs_simple(tmp_path):
    classic = build_vitpose("B", "classic", num_keypoints=6).state_dict()
    simple = build_vitpose("B", "simple", num_keypoints=6).state_dict()
    assert infer_head_from_state(classic) == "classic"
    assert infer_head_from_state(simple) == "simple"


def test_load_finetuned_roundtrip(tmp_path):
    p = _save_training_ckpt(tmp_path, "B", "classic", 6)
    model, meta = load_finetuned_checkpoint(p)
    assert isinstance(meta, FinetuneMeta)
    assert meta.variant == "B"
    assert meta.head == "classic"
    assert meta.num_keypoints == 6
    # forward produces (1, 6, 64, 48) heatmaps
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 256, 192))
    assert out.shape == (1, 6, 64, 48)


def test_load_plain_state_dict_infers(tmp_path):
    # a bare state_dict (user-supplied, no metadata wrapper)
    model = build_vitpose("S", "simple", num_keypoints=4)
    p = tmp_path / "plain.pt"
    torch.save(model.state_dict(), p)
    loaded, meta = load_finetuned_checkpoint(p)
    assert meta.head == "simple"
    assert meta.num_keypoints == 4
    assert meta.variant == "S"  # inferred from embed_dim
