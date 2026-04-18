import argparse
import contextlib
import json
import math
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from timm.models import create_model

from datasets.kinetics_sparse_et import VideoClsDataset_sparse
from models import *  # noqa: F401,F403 - registers VideoMamba models with timm


def get_args():
    parser = argparse.ArgumentParser("Clean VideoMamba ET fine-tuning")
    parser.add_argument("--model", default="videomamba_small")
    parser.add_argument("--finetune", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--delete_head", action="store_true")
    parser.add_argument("--model_key", default="model|module")

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--split", default=",")
    parser.add_argument("--train_view1_csv", default="aligned_v01_1.csv")
    parser.add_argument("--train_view2_csv", default="aligned_v02_2.csv")
    parser.add_argument("--val_view_csv", default="v03_val_set.csv")
    parser.add_argument("--nb_classes", default=12, type=int)

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")

    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--clip_grad", default=None, type=float)
    parser.add_argument("--update_freq", default=1, type=int)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--fused_ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--view_ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--debug_overfit_samples", default=0, type=int)

    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--short_side_size", default=224, type=int)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--sampling_rate", default=4, type=int)
    parser.add_argument("--tubelet_size", default=1, type=int)
    parser.add_argument("--drop_path", default=0.0, type=float)
    parser.add_argument("--fc_drop_rate", default=0.0, type=float)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_num", default=0, type=int)
    parser.add_argument("--use_mean_pooling", action="store_true", default=True)
    parser.add_argument("--use_cls", action="store_false", dest="use_mean_pooling")

    parser.add_argument("--aa", default="none")
    parser.add_argument("--train_interpolation", default="bicubic")
    parser.add_argument("--train_crop_min_scale", default=0.5, type=float)
    parser.add_argument("--train_crop_max_scale", default=1.0, type=float)
    parser.add_argument("--train_crop_min_ratio", default=0.75, type=float)
    parser.add_argument("--train_crop_max_ratio", default=1.3333, type=float)
    parser.add_argument("--disable_train_flip", action="store_true", default=True)
    parser.add_argument("--reprob", default=-1.0, type=float)
    parser.add_argument("--remode", default="pixel")
    parser.add_argument("--recount", default=1, type=int)
    parser.add_argument("--num_sample", default=1, type=int)
    parser.add_argument("--data_set", default="Kinetics_sparse_et")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_key(label):
    try:
        return int(label)
    except (TypeError, ValueError):
        return str(label)


def balanced_subset_indices(dataset, sample_count, seed):
    labels = getattr(dataset, "label_array", None)
    if labels is None:
        rng = np.random.default_rng(seed)
        return rng.permutation(len(dataset))[:sample_count].tolist()

    rng = np.random.default_rng(seed)
    class_to_indices = OrderedDict()
    for index, label in enumerate(labels):
        class_to_indices.setdefault(label_key(label), []).append(index)
    for indices in class_to_indices.values():
        rng.shuffle(indices)

    selected = []
    positions = {key: 0 for key in class_to_indices}
    while len(selected) < sample_count:
        added = False
        for key, indices in class_to_indices.items():
            if positions[key] < len(indices):
                selected.append(indices[positions[key]])
                positions[key] += 1
                added = True
                if len(selected) >= sample_count:
                    break
        if not added:
            break
    return selected


def make_dataset(args, mode):
    if mode == "train":
        anno_view1 = os.path.join(args.data_path, args.train_view1_csv)
        anno_view2 = os.path.join(args.data_path, args.train_view2_csv)
    elif mode == "validation":
        anno_view1 = os.path.join(args.data_path, args.val_view_csv)
        anno_view2 = None
    else:
        raise ValueError(f"Unsupported mode for clean training: {mode}")

    return VideoClsDataset_sparse(
        anno_path_view1=anno_view1,
        anno_path_view2=anno_view2,
        prefix=args.prefix,
        split=args.split,
        mode=mode,
        clip_len=args.num_frames,
        frame_sample_rate=args.sampling_rate,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        num_segment=1,
        num_crop=1,
        test_num_segment=1,
        test_num_crop=1,
        keep_aspect_ratio=True,
        new_height=256,
        new_width=320,
        args=args,
    )


def extract_checkpoint_state(checkpoint, model_key):
    for key in model_key.split("|"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            return checkpoint[key], key
    return checkpoint, "<root>"


def strip_prefixes(state):
    stripped = OrderedDict()
    for key, value in state.items():
        if key.startswith("backbone."):
            key = key[len("backbone."):]
        elif key.startswith("encoder."):
            key = key[len("encoder."):]
        elif key.startswith("module."):
            key = key[len("module."):]
        stripped[key] = value
    return stripped


def load_pretrained(model, path, model_key, delete_head):
    if not path:
        return
    checkpoint = torch.load(path, map_location="cpu")
    state, used_key = extract_checkpoint_state(checkpoint, model_key)
    state = strip_prefixes(state)
    if delete_head:
        state.pop("head.weight", None)
        state.pop("head.bias", None)

    model_state = model.state_dict()
    loadable = OrderedDict()
    skipped = []
    for key, value in state.items():
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
            loadable[key] = value
        else:
            skipped.append(key)

    msg = model.load_state_dict(loadable, strict=False)
    print(f"Loaded pretrained checkpoint: {path}")
    print(f"Checkpoint key: {used_key}; loaded={len(loadable)} skipped={len(skipped)}")
    print(f"Missing keys: {len(msg.missing_keys)}; unexpected keys: {len(msg.unexpected_keys)}")
    if skipped[:8]:
        print(f"First skipped keys: {skipped[:8]}")


def save_checkpoint(args, model, optimizer, scheduler, epoch, best_acc1, name):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(args.output_dir) / f"{name}.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_acc1": best_acc1,
            "args": vars(args),
        },
        path,
    )


def load_resume(model, optimizer, scheduler, path, device):
    if not path:
        return 0, 0.0
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=False)
    if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_acc1 = float(checkpoint.get("best_acc1", 0.0))
    print(f"Resumed from {path}; start_epoch={start_epoch}; best_acc1={best_acc1:.2f}")
    return start_epoch, best_acc1


def make_scheduler(optimizer, args):
    def lr_lambda(epoch):
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        denom = max(1, args.epochs - args.warmup_epochs)
        progress = min(1.0, (epoch - args.warmup_epochs) / denom)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = args.min_lr / args.lr if args.lr > 0 else 0.0
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def amp_context(device, args):
    if device.type != "cuda" or (not args.bf16 and not args.fp16):
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    return torch.cuda.amp.autocast(dtype=dtype)


def accuracy(logits, target, topk=(1, 5)):
    maxk = min(max(topk), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    result = []
    for k in topk:
        k = min(k, logits.shape[1])
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / target.numel()).item())
    return result


def new_hist(num_classes):
    return torch.zeros(num_classes, dtype=torch.long)


def update_hist(hist, values):
    hist += torch.bincount(values.detach().cpu().long().view(-1), minlength=hist.numel())[:hist.numel()]


def hist_list(hist):
    return [int(x) for x in hist.tolist()]


def move_train_batch(batch, device):
    view1, view2, target = batch[0], batch[1], batch[2]
    return (
        view1.to(device, non_blocking=True),
        view2.to(device, non_blocking=True),
        target.to(device, non_blocking=True).long(),
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, args):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "loss": 0.0,
        "fused_ce_loss": 0.0,
        "view_ce_loss": 0.0,
        "fused_acc1": 0.0,
        "fused_acc5": 0.0,
        "view1_acc1": 0.0,
        "view2_acc1": 0.0,
        "count": 0,
    }
    hists = {
        "target_hist": new_hist(args.nb_classes),
        "fused_pred_hist": new_hist(args.nb_classes),
        "view1_pred_hist": new_hist(args.nb_classes),
        "view2_pred_hist": new_hist(args.nb_classes),
    }

    start = time.time()
    for step, batch in enumerate(loader):
        view1, view2, target = move_train_batch(batch, device)
        with amp_context(device, args):
            model_outputs = model(view1, view2, return_view_logits=True)
            _, view1_logits, view2_logits = model_outputs[:3]
            fused_logits = 0.5 * (view1_logits + view2_logits)
            fused_ce = criterion(fused_logits, target)
            view_ce = 0.5 * (criterion(view1_logits, target) + criterion(view2_logits, target))
            loss = args.fused_ce_loss_weight * fused_ce + args.view_ce_loss_weight * view_ce
            loss_for_backward = loss / args.update_freq

        if scaler.is_enabled():
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        if (step + 1) % args.update_freq == 0 or (step + 1) == len(loader):
            if scaler.is_enabled():
                if args.clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = target.numel()
        fused_acc1, fused_acc5 = accuracy(fused_logits.detach(), target, topk=(1, 5))
        view1_acc1, _ = accuracy(view1_logits.detach(), target, topk=(1, 5))
        view2_acc1, _ = accuracy(view2_logits.detach(), target, topk=(1, 5))
        totals["loss"] += loss.item() * batch_size
        totals["fused_ce_loss"] += fused_ce.item() * batch_size
        totals["view_ce_loss"] += view_ce.item() * batch_size
        totals["fused_acc1"] += fused_acc1 * batch_size
        totals["fused_acc5"] += fused_acc5 * batch_size
        totals["view1_acc1"] += view1_acc1 * batch_size
        totals["view2_acc1"] += view2_acc1 * batch_size
        totals["count"] += batch_size

        update_hist(hists["target_hist"], target)
        update_hist(hists["fused_pred_hist"], fused_logits.argmax(dim=1))
        update_hist(hists["view1_pred_hist"], view1_logits.argmax(dim=1))
        update_hist(hists["view2_pred_hist"], view2_logits.argmax(dim=1))

        if step % max(1, args.print_freq) == 0:
            print(
                f"Epoch {epoch} [{step}/{len(loader)}] "
                f"loss={loss.item():.4f} acc1={fused_acc1:.2f} "
                f"view1={view1_acc1:.2f} view2={view2_acc1:.2f}"
            )

    count = max(1, totals.pop("count"))
    stats = {f"train_{key}": value / count for key, value in totals.items()}
    stats["train_time"] = time.time() - start
    stats.update({f"train_{key}": hist_list(value) for key, value in hists.items()})
    return stats


@torch.no_grad()
def validate(model, loader, criterion, device, args):
    model.eval()
    totals = {"loss": 0.0, "acc1": 0.0, "acc5": 0.0, "count": 0}
    hists = {"target_hist": new_hist(args.nb_classes), "pred_hist": new_hist(args.nb_classes)}

    for batch in loader:
        video, target = batch[0], batch[1]
        video = video.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        with amp_context(device, args):
            logits = model(video)
            loss = criterion(logits, target)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        batch_size = target.numel()
        totals["loss"] += loss.item() * batch_size
        totals["acc1"] += acc1 * batch_size
        totals["acc5"] += acc5 * batch_size
        totals["count"] += batch_size
        update_hist(hists["target_hist"], target)
        update_hist(hists["pred_hist"], logits.argmax(dim=1))

    count = max(1, totals.pop("count"))
    stats = {f"val_{key}": value / count for key, value in totals.items()}
    stats.update({f"val_{key}": hist_list(value) for key, value in hists.items()})
    return stats


def write_log(args, stats):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "log.txt", "a", encoding="utf-8") as f:
        f.write(json.dumps(stats, ensure_ascii=False) + "\n")


def main():
    args = get_args()
    if args.fused_ce_loss_weight <= 0 and args.view_ce_loss_weight <= 0:
        raise ValueError("At least one of fused_ce_loss_weight or view_ce_loss_weight must be > 0.")
    if args.bf16 and args.fp16:
        raise ValueError("Use only one mixed precision mode: --bf16 or --fp16.")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print("Clean ET training configuration:")
    for key in sorted(vars(args)):
        print(f"  {key}: {getattr(args, key)}")

    train_dataset = make_dataset(args, "train")
    if args.debug_overfit_samples > 0:
        sample_count = min(len(train_dataset), max(args.debug_overfit_samples, args.batch_size))
        indices = balanced_subset_indices(train_dataset, sample_count, args.seed)
        labels = [label_key(train_dataset.label_array[index]) for index in indices]
        label_hist = OrderedDict()
        for label in labels:
            label_hist[label] = label_hist.get(label, 0) + 1
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Debug balanced subset: {len(indices)} samples; class histogram: {dict(label_hist)}")
    val_dataset = make_dataset(args, "validation")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=max(1, int(1.5 * args.batch_size)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    model = create_model(
        args.model,
        img_size=args.input_size,
        pretrained=False if args.finetune else True,
        num_classes=args.nb_classes,
        fc_drop_rate=args.fc_drop_rate,
        drop_path_rate=args.drop_path,
        kernel_size=args.tubelet_size,
        num_frames=args.num_frames,
        use_checkpoint=args.use_checkpoint,
        checkpoint_num=args.checkpoint_num,
        use_mean_pooling=args.use_mean_pooling,
    )
    load_pretrained(model, args.finetune, args.model_key, args.delete_head)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    if args.resume:
        args.start_epoch, best_acc1 = load_resume(model, optimizer, scheduler, args.resume, device)
    else:
        best_acc1 = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: lr={current_lr:.8g}")
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, args)
        val_stats = validate(model, val_loader, criterion, device, args)
        scheduler.step()

        stats = {
            "epoch": epoch,
            "lr": current_lr,
            **train_stats,
            **val_stats,
        }
        write_log(args, stats)
        print(
            f"Epoch {epoch} summary: "
            f"train_acc1={stats['train_fused_acc1']:.2f} "
            f"val_acc1={stats['val_acc1']:.2f} "
            f"train_loss={stats['train_loss']:.4f} "
            f"val_loss={stats['val_loss']:.4f}"
        )

        save_checkpoint(args, model, optimizer, scheduler, epoch, best_acc1, "latest")
        if stats["val_acc1"] >= best_acc1:
            best_acc1 = stats["val_acc1"]
            save_checkpoint(args, model, optimizer, scheduler, epoch, best_acc1, "best")


if __name__ == "__main__":
    main()
