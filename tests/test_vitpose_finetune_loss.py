import torch

from hydra_suite.core.identity.pose.vitpose.training.loss import JointsMSELoss


def test_zero_loss_when_equal():
    out = torch.rand(2, 3, 8, 6)
    w = torch.ones(2, 3, 1)
    loss = JointsMSELoss(True)(out, out.clone(), w)
    assert loss.item() < 1e-8


def test_weight_masks_joint():
    out = torch.zeros(1, 2, 4, 4)
    tgt = torch.zeros(1, 2, 4, 4)
    out[0, 1] = 5.0  # only joint 1 wrong
    w = torch.tensor([[[1.0], [0.0]]])  # joint 1 masked out
    assert JointsMSELoss(True)(out, tgt, w).item() < 1e-8
    w2 = torch.ones(1, 2, 1)
    assert JointsMSELoss(True)(out, tgt, w2).item() > 1.0
