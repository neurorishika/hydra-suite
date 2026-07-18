from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import HEATMAP_SIZE_WH, VARIANTS
from ..decode import decode_udp_cv2
from ..transforms import transform_preds
from .config import RunConfig
from .dataset import CocoKeypointsDataset, load_coco_index
from .loss import JointsMSELoss
from .model_setup import build_finetune_model, build_param_groups, load_finetune_init
from .validate import run_validation


def _split(
    ids: list[int], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    order = ids[:]
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_fraction)))
    return order[n_val:], order[:n_val]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg: RunConfig) -> dict:
    _seed_everything(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out_dir / "run.json")
    device = torch.device(cfg.device)

    all_ids, _ = load_coco_index(Path(cfg.dataset_dir))
    train_ids, val_ids = _split(all_ids, cfg.val_fraction, cfg.seed)
    train_ds = CocoKeypointsDataset(cfg.dataset_dir, train_ids, cfg.sigma, augment=True)
    val_ds = CocoKeypointsDataset(cfg.dataset_dir, val_ids, cfg.sigma, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    model = build_finetune_model(cfg.variant, cfg.num_keypoints, cfg.drop_path)
    load_finetune_init(model, Path(cfg.init_checkpoint))
    model.to(device)

    groups = build_param_groups(
        model, cfg.lr, VARIANTS[cfg.variant].layer_decay, cfg.weight_decay
    )
    opt = torch.optim.AdamW(groups)
    crit = JointsMSELoss(True)

    start_epoch, best_pck, best_epoch = 0, -1.0, 0
    resume_ckpt = None
    if cfg.resume_from:
        resume_ckpt = torch.load(cfg.resume_from, map_location="cpu", weights_only=True)
        model.load_state_dict(resume_ckpt["model_state"])
        opt.load_state_dict(resume_ckpt["optim_state"])
        start_epoch = int(resume_ckpt["epoch"]) + 1
        best_pck = float(resume_ckpt.get("pck", -1.0))

    if resume_ckpt is not None and "sched_state" in resume_ckpt:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        sched.load_state_dict(resume_ckpt["sched_state"])
    else:
        if start_epoch > 0:
            # Older checkpoints don't carry sched_state: seed 'initial_lr' from the
            # freshly-built param groups (their base lr, unaffected by the optimizer
            # state we just loaded) so CosineAnnealingLR can resume at last_epoch>=0.
            for group, base_group in zip(opt.param_groups, groups):
                group.setdefault("initial_lr", base_group["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, last_epoch=start_epoch - 1
        )

    metrics_path = out_dir / "metrics.csv"
    if not metrics_path.exists():
        metrics_path.write_text(
            "epoch,train_loss,val_loss,pck@0.05,pck@0.1\n", encoding="utf-8"
        )

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            img = batch["image"].to(device)
            out = model(img)
            loss = crit(
                out, batch["target"].to(device), batch["target_weight"].to(device)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item() * img.shape[0]
            n += img.shape[0]
        sched.step()
        train_loss = running / max(n, 1)

        val = run_validation(model, val_loader, device)
        p05, p10 = val["pck"][0.05], val["pck"][0.1]
        print(
            f"EPOCH {epoch} train_loss={train_loss:.5f} val_loss={val['val_loss']:.5f} "
            f"pck@0.05={p05:.4f} pck@0.1={p10:.4f}",
            flush=True,
        )
        with metrics_path.open("a", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val['val_loss']:.6f}",
                    f"{p05:.6f}",
                    f"{p10:.6f}",
                ]
            )

        ckpt = {
            "model_state": model.state_dict(),
            "optim_state": opt.state_dict(),
            "variant": cfg.variant,
            "num_keypoints": cfg.num_keypoints,
            "epoch": epoch,
            "pck": p05,
            "sched_state": sched.state_dict(),
        }
        torch.save(ckpt, out_dir / "last.pt")
        if p05 >= best_pck:
            best_pck, best_epoch = p05, epoch
            torch.save(ckpt, out_dir / "best.pt")

    _write_val_overlays(
        model, val_ds, device, out_dir / "val_overlays", cfg.num_keypoints
    )
    return {"best_pck": best_pck, "best_epoch": best_epoch, "output_dir": str(out_dir)}


def _write_val_overlays(
    model, val_ds, device, dst: Path, k: int, limit: int = 6
) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(min(limit, len(val_ds))):
            s = val_ds[i]
            out = model(s["image"].unsqueeze(0).to(device)).cpu().numpy()
            coords, _ = decode_udp_cv2(out, kernel=11)
            pred = transform_preds(
                coords[0], s["center"].numpy(), s["scale"].numpy(), HEATMAP_SIZE_WH
            )
            img = cv2.imread(
                str(val_ds.dir / "images" / val_ds.index[val_ds.ids[i]][1]["file_name"])
            )
            for j in range(k):
                cv2.circle(img, (int(pred[j, 0]), int(pred[j, 1])), 3, (0, 0, 255), -1)
            cv2.imwrite(str(dst / f"val_{i}.png"), img)
