from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import VARIANTS
from ..decode import decode_udp_cv2
from ..geometry import DEFAULT_GEOMETRY, PoseGeometry
from ..transforms import transform_preds
from .augment import AugmentParams
from .config import RunConfig
from .dataset import CocoKeypointsDataset, image_path, load_coco_index, seed_worker
from .ema import ModelEMA
from .loss import JointsMSELoss
from .model_setup import build_finetune_model, build_param_groups, load_finetune_init
from .split import split_ids
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


def _warmup_cosine(total_iters: int, warmup_iters: int):
    """lr multiplier per optimizer step: linear warmup, then cosine to 0."""
    total_iters = max(1, total_iters)
    warmup_iters = max(0, min(warmup_iters, total_iters - 1))

    def factor(step: int) -> float:
        if warmup_iters and step < warmup_iters:
            return 0.001 + 0.999 * step / warmup_iters
        span = max(1, total_iters - warmup_iters)
        t = min(1.0, (step - warmup_iters) / span)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    return factor


def _loader(ds, cfg: RunConfig, shuffle: bool, device) -> DataLoader:
    # Validation is small and unaugmented: load it in-process rather than
    # keeping a second set of (each ~1 GB) spawned workers alive.
    workers = int(cfg.num_workers) if shuffle else 0
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        worker_init_fn=seed_worker if workers else None,
        # Spawned, not forked: by the time the loaders start, the model is on
        # the GPU and torch / OpenMP threads hold locks that forked workers
        # inherit (and then deadlock on).  Persistent, so the spawn cost is paid
        # once per run.
        multiprocessing_context="spawn" if workers else None,
        persistent_workers=workers > 0,
        pin_memory=device.type == "cuda",
        generator=gen,
    )


def _flip_test_pairs(cfg: RunConfig):
    """Pairs for the flip test, or None when it is off (needs a model that
    saw left-right flips in training)."""
    if not cfg.flip_test or cfg.fliplr <= 0:
        return None
    return [tuple(p) for p in (cfg.flip_pairs or [])]


def _append_metrics(path: Path, row: dict) -> None:
    """Append ``row`` under the file's existing header (older runs resume
    into a file without the newer columns)."""
    with path.open(encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh), None) or list(row)
    with path.open("a", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, fieldnames=header, extrasaction="ignore").writerow(row)


def train(cfg: RunConfig) -> dict:
    _seed_everything(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out_dir / "run.json")
    device = torch.device(cfg.device)
    variant = VARIANTS[cfg.variant]

    geom = (
        PoseGeometry.from_hw(cfg.input_size)
        if cfg.input_size is not None
        else DEFAULT_GEOMETRY
    )

    all_ids, index = load_coco_index(Path(cfg.dataset_dir))
    if cfg.split_by == "image":
        train_ids, val_ids = _split(all_ids, cfg.val_fraction, cfg.seed)
    else:
        names = {i: index[i][1]["file_name"] for i in all_ids}
        train_ids, val_ids = split_ids(
            all_ids, names, cfg.val_fraction, cfg.seed, cfg.split_by
        )
    print(
        f"SPLIT by={cfg.split_by} train={len(train_ids)} val={len(val_ids)}",
        flush=True,
    )
    aug = AugmentParams(
        rotation=cfg.rotation,
        rotation_prob=cfg.rotation_prob,
        brightness=cfg.brightness,
        contrast=cfg.contrast,
        saturation=cfg.saturation,
        hue=cfg.hue,
        blur_prob=cfg.blur_prob,
        noise_prob=cfg.noise_prob,
        occlusion_prob=cfg.occlusion_prob,
    )
    train_ds = CocoKeypointsDataset(
        cfg.dataset_dir,
        train_ids,
        cfg.sigma,
        augment=True,
        geom=geom,
        scale_jitter=cfg.scale_jitter,
        aspect_jitter=cfg.aspect_jitter,
        seed=cfg.seed,
        flip_pairs=cfg.flip_pairs,
        fliplr=cfg.fliplr,
        flipud=cfg.flipud,
        aug=aug,
        ignore_occluded=cfg.ignore_occluded,
    )
    val_ds = CocoKeypointsDataset(
        cfg.dataset_dir, val_ids, cfg.sigma, augment=False, geom=geom
    )
    train_loader = _loader(train_ds, cfg, True, device)
    val_loader = _loader(val_ds, cfg, False, device)

    drop_path = cfg.drop_path if cfg.drop_path is not None else variant.drop_path_rate
    model = build_finetune_model(cfg.variant, cfg.num_keypoints, drop_path, geom=geom)
    load_finetune_init(model, Path(cfg.init_checkpoint), geom=geom)
    model.to(device)

    layer_decay = (
        cfg.layer_decay if cfg.layer_decay is not None else variant.layer_decay
    )
    groups = build_param_groups(model, cfg.lr, layer_decay, cfg.weight_decay)
    opt = torch.optim.AdamW(groups)
    crit = JointsMSELoss(True)
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema else None
    use_amp = bool(cfg.amp) and device.type == "cuda"
    flip_pairs = _flip_test_pairs(cfg)

    iters_per_epoch = max(1, len(train_loader))
    total_iters = cfg.epochs * iters_per_epoch
    warmup_iters = int(round(cfg.warmup_epochs * iters_per_epoch))
    for group in opt.param_groups:
        group.setdefault("initial_lr", group["lr"])

    start_epoch, best_pck, best_nme, best_epoch = 0, -1.0, float("inf"), 0
    stale = 0
    resume_ckpt = None
    if cfg.resume_from:
        resume_ckpt = torch.load(cfg.resume_from, map_location="cpu", weights_only=True)
        model.load_state_dict(
            resume_ckpt.get("train_state", resume_ckpt["model_state"])
        )
        opt.load_state_dict(resume_ckpt["optim_state"])
        if ema is not None:
            if "ema_state" in resume_ckpt:
                ema.load_state_dict(resume_ckpt["ema_state"])
            else:
                ema.module.load_state_dict(model.state_dict())
        start_epoch = int(resume_ckpt["epoch"]) + 1
        best_pck = float(resume_ckpt.get("pck", -1.0))
        best_nme = float(resume_ckpt.get("nme", float("inf")))
        best_epoch = int(resume_ckpt.get("best_epoch", resume_ckpt["epoch"]))
        stale = int(resume_ckpt.get("stale", 0))
        if "best_epoch" not in resume_ckpt:
            # Older checkpoints store this epoch's PCK, not the best: take
            # the best from the run's best.pt so it is not overwritten by a
            # worse epoch.
            best_file = Path(cfg.resume_from).with_name("best.pt")
            if best_file.is_file():
                best = torch.load(best_file, map_location="cpu", weights_only=True)
                best_pck = float(best.get("pck", best_pck))
                best_epoch = int(best.get("epoch", best_epoch))
        for group, base_group in zip(opt.param_groups, groups):
            group["initial_lr"] = base_group["lr"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        _warmup_cosine(total_iters, warmup_iters),
        last_epoch=start_epoch * iters_per_epoch - 1,
    )

    metrics_path = out_dir / "metrics.csv"
    if start_epoch == 0 or not metrics_path.exists():
        metrics_path.write_text(
            "epoch,train_loss,val_loss,pck@0.05,pck@0.1,nme,lr\n", encoding="utf-8"
        )

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running, n = 0.0, 0
        lr_sum, steps = 0.0, 0
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(img)
            loss = crit(
                out.float(),
                batch["target"].to(device, non_blocking=True),
                batch["target_weight"].to(device, non_blocking=True),
            )
            lr_sum += opt.param_groups[-1]["lr"]  # the lr this step uses
            steps += 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update(model)
            running += loss.item() * img.shape[0]
            n += img.shape[0]
        train_loss = running / max(n, 1)

        eval_model = ema.module if ema is not None else model
        val = run_validation(
            eval_model,
            val_loader,
            device,
            geom=geom,
            flip_pairs=flip_pairs,
            amp=use_amp,
        )
        p05, p10, nme = val["pck"][0.05], val["pck"][0.1], val["nme"]
        lr_now = lr_sum / max(steps, 1)  # mean lr of the epoch (head group)
        print(
            f"EPOCH {epoch} train_loss={train_loss:.5f} val_loss={val['val_loss']:.5f} "
            f"pck@0.05={p05:.4f} pck@0.1={p10:.4f}",
            flush=True,
        )
        flip_txt = ""
        if "flip_pck" in val:
            flip_txt = (
                f" flip_pck@0.05={val['flip_pck'][0.05]:.4f}"
                f" flip_nme={val['flip_nme']:.5f}"
            )
        print(
            f"METRICS epoch={epoch} nme={nme:.5f} lr={lr_now:.3e}{flip_txt}",
            flush=True,
        )
        _append_metrics(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val['val_loss']:.6f}",
                "pck@0.05": f"{p05:.6f}",
                "pck@0.1": f"{p10:.6f}",
                "nme": f"{nme:.6f}",
                "lr": f"{lr_now:.6e}",
            },
        )

        # Best = highest PCK@0.05; lower normalised error breaks ties.
        better = p05 > best_pck or (p05 == best_pck and nme < best_nme)
        if better:
            best_pck, best_nme, best_epoch = p05, nme, epoch
            stale = 0
        else:
            stale += 1
        ckpt = {
            # inference weights: the EMA copy when EMA is on
            "model_state": eval_model.state_dict(),
            "train_state": model.state_dict(),
            "optim_state": opt.state_dict(),
            "variant": cfg.variant,
            "num_keypoints": cfg.num_keypoints,
            "input_size": geom.to_hw(),
            "epoch": epoch,
            "pck": best_pck,
            "nme": best_nme,
            "best_epoch": best_epoch,
            "stale": stale,
            "sched_state": sched.state_dict(),
        }
        if ema is not None:
            ckpt["ema_state"] = ema.state_dict()
        torch.save(ckpt, out_dir / "last.pt")
        if better:
            ckpt["pck"], ckpt["nme"] = p05, nme
            torch.save(ckpt, out_dir / "best.pt")
        if cfg.patience and stale >= cfg.patience:
            print(
                f"EARLY STOP epoch={epoch}: no better validation for "
                f"{cfg.patience} epochs",
                flush=True,
            )
            break

    final = ema.module if ema is not None else model
    _write_val_overlays(
        final, val_ds, device, out_dir / "val_overlays", cfg.num_keypoints, geom=geom
    )
    return {"best_pck": best_pck, "best_epoch": best_epoch, "output_dir": str(out_dir)}


def _write_val_overlays(
    model,
    val_ds,
    device,
    dst: Path,
    k: int,
    limit: int = 6,
    geom: PoseGeometry = DEFAULT_GEOMETRY,
) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(min(limit, len(val_ds))):
            s = val_ds[i]
            out = model(s["image"].unsqueeze(0).to(device)).cpu().numpy()
            coords, _ = decode_udp_cv2(out, kernel=11)
            pred = transform_preds(
                coords[0], s["center"].numpy(), s["scale"].numpy(), geom.heatmap_size_wh
            )
            img = cv2.imread(
                str(image_path(val_ds.dir, val_ds.index[val_ds.ids[i]][1]["file_name"]))
            )
            for j in range(k):
                cv2.circle(img, (int(pred[j, 0]), int(pred[j, 1])), 3, (0, 0, 255), -1)
            cv2.imwrite(str(dst / f"val_{i}.png"), img)
