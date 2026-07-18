import math

from hydra_suite.core.identity.pose.vitpose.training.model_setup import (
    _layer_id_for,
    build_finetune_model,
    build_param_groups,
)


def test_layer_ids_use_plus_two():
    _ = build_finetune_model("B", 6, 0.1)  # depth=12 -> num_layers=14
    num_layers = 12 + 2
    assert _layer_id_for("backbone.pos_embed", num_layers) == 0
    assert _layer_id_for("backbone.patch_embed.proj.weight", num_layers) == 0
    assert _layer_id_for("backbone.blocks.0.attn.qkv.weight", num_layers) == 1
    assert _layer_id_for("backbone.blocks.11.mlp.fc1.weight", num_layers) == 12
    assert (
        _layer_id_for("keypoint_head.final_layer.weight", num_layers) == num_layers - 1
    )
    assert _layer_id_for("backbone.last_norm.weight", num_layers) == num_layers - 1


def test_lr_scales_and_no_decay():
    m = build_finetune_model("B", 6, 0.1)
    decay, base_lr, wd = 0.75, 5e-4, 0.1
    groups = build_param_groups(m, base_lr, decay, wd)
    # every parameter appears exactly once
    n_params = sum(len(g["params"]) for g in groups)
    assert n_params == sum(1 for _ in m.parameters())
    # head group runs at full base_lr (scale ** 0)
    head_lrs = [
        g["lr"]
        for g in groups
        if any(p is m.keypoint_head.final_layer.weight for p in g["params"])
    ]
    assert math.isclose(head_lrs[0], base_lr, rel_tol=1e-6)
    # bias / norm / pos_embed groups carry zero weight decay
    for g in groups:
        if g["weight_decay"] == 0.0:
            continue
        for p in g["params"]:
            assert p.ndim > 1  # decayed params are weight matrices, never biases
