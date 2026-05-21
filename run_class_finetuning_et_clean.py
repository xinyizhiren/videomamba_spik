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
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from datasets.multiview_action_clean import CrossViewTrainDataset, SingleViewDataset, VideoTransform
from models.videomamba_clean import create_videomamba_small_clean


MODEL_ALIASES = {
    "videomamba_small": "videomamba_small_clean",
    "videomamba_small_clean": "videomamba_small_clean",
    "spikmamba": "spikmamba_fixed",
    "spikmamba_fixed": "spikmamba_fixed",
    "videomamba_small_trainable_snn": "videomamba_small_trainable_snn",
    "trainable_snn": "videomamba_small_trainable_snn",
}


def init_distributed_mode(args):
    args.rank = 0
    args.world_size = 1
    args.local_rank = 0
    args.distributed = False

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return

    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.distributed = args.world_size > 1
    if not args.distributed:
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def main_print(*values, **kwargs):
    if is_main_process():
        print(*values, **kwargs)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def reset_stateful_modules(model):
    target = unwrap_model(model)
    if hasattr(target, "reset_spike_state"):
        target.reset_spike_state()
    elif hasattr(target, "reset_states"):
        target.reset_states()


def active_spike_module_names(model):
    target = unwrap_model(model)
    if hasattr(target, "active_spike_module_names"):
        return list(target.active_spike_module_names())
    return []


def active_spike_parameter_names(model):
    target = unwrap_model(model)
    if hasattr(target, "active_spike_parameter_names"):
        return set(target.active_spike_parameter_names())
    return set()


class DistributedEvalSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        if len(self.dataset) <= self.rank:
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.num_replicas + 1


def get_args():
    parser = argparse.ArgumentParser("Clean VideoMamba ET fine-tuning")
    parser.add_argument(
        "--model",
        default="videomamba_small_clean",
        help="Model to train: videomamba_small_clean/videomamba_small or spikmamba_fixed/spikmamba.",
    )
    parser.add_argument("--finetune", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--model_key", default="model|module")
    parser.add_argument(
        "--min_pretrained_load_ratio",
        default=0.05,
        type=float,
        help="Skip --finetune loading when matched tensor ratio is below this value.",
    )

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--csv_delimiter", default=",")
    parser.add_argument("--train_view1_csv", default="aligned_v01_1.csv")
    parser.add_argument("--train_view2_csv", default="aligned_v02_2.csv")
    parser.add_argument("--val_view_csv", default="v03_val_set.csv")
    parser.add_argument("--test_view_csv", default="v03_test_set.csv")
    parser.add_argument("--nb_classes", default=12, type=int)

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.add_argument("--disable_cudnn_benchmark", action="store_true", default=False)

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

    parser.add_argument("--fused_ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--view_ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--debug_overfit_samples", default=0, type=int)
    parser.add_argument("--eval", action="store_true", help="Run evaluation only.")
    parser.add_argument("--eval_split", default="test", choices=("validation", "test"))
    parser.add_argument("--eval_checkpoint", default="", help="Checkpoint path for eval-only mode.")
    parser.add_argument("--skip_initial_eval", action="store_true", default=False)
    parser.add_argument("--skip_initial_best_checkpoint", action="store_true", default=False)
    parser.add_argument("--dump_model_summary", action="store_true", default=False)
    parser.add_argument("--summary_depth", default=5, type=int)

    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--short_side_size", default=224, type=int)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--sampling_rate", default=4, type=int)
    parser.add_argument("--tubelet_size", default=1, type=int)
    parser.add_argument("--drop_path", default=0.0, type=float)
    parser.add_argument("--fc_drop_rate", default=0.0, type=float)
    parser.add_argument("--use_mean_pooling", action="store_true", default=True)
    parser.add_argument("--use_cls", action="store_false", dest="use_mean_pooling")
    parser.add_argument("--spik_patch_size", default=14, type=int)
    parser.add_argument("--spik_embed_dims", default=384, type=int)
    parser.add_argument("--spik_time_steps", default=1, type=int)
    parser.add_argument("--spik_bimamba", action="store_true", default=True)
    parser.add_argument("--no_spik_bimamba", action="store_false", dest="spik_bimamba")
    parser.add_argument("--snn_block_indices", default="0")
    parser.add_argument("--snn_spike_patch", action="store_true", default=False)
    parser.add_argument("--snn_spike_position", default="post", choices=("pre", "post", "prepost"))
    parser.add_argument("--snn_unsigned_spikes", action="store_false", dest="snn_signed_spikes")
    parser.add_argument("--snn_timesteps", default=4, type=int)
    parser.add_argument("--snn_threshold_init", default=1.0, type=float)
    parser.add_argument("--snn_threshold_percentile", default=0.99, type=float)
    parser.add_argument("--snn_fixed_threshold", action="store_false", dest="snn_train_threshold")
    parser.add_argument("--snn_surrogate_alpha", default=4.0, type=float)
    parser.add_argument("--spike_lr_multiplier", default=5.0, type=float)
    parser.add_argument("--distill_weight", default=0.0, type=float)
    parser.add_argument("--distill_temperature", default=2.0, type=float)
    parser.add_argument("--teacher_checkpoint", default="")
    parser.set_defaults(snn_signed_spikes=True, snn_train_threshold=True)

    parser.add_argument("--train_crop_min_scale", default=0.5, type=float)
    parser.add_argument("--train_crop_max_scale", default=1.0, type=float)
    parser.add_argument("--train_crop_min_ratio", default=0.75, type=float)
    parser.add_argument("--train_crop_max_ratio", default=1.3333, type=float)
    parser.add_argument("--disable_train_flip", action="store_true", default=True)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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
    deterministic_debug = args.debug_overfit_samples > 0
    if mode == "train":
        anno_view1 = os.path.join(args.data_path, args.train_view1_csv)
        anno_view2 = os.path.join(args.data_path, args.train_view2_csv)
        transform = VideoTransform(
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            train=True,
            deterministic=deterministic_debug,
            crop_scale=(args.train_crop_min_scale, args.train_crop_max_scale),
            crop_ratio=(args.train_crop_min_ratio, args.train_crop_max_ratio),
            horizontal_flip=not args.disable_train_flip,
        )
        return CrossViewTrainDataset(
            view1_csv=anno_view1,
            view2_csv=anno_view2,
            prefix=args.prefix,
            delimiter=args.csv_delimiter,
            clip_len=args.num_frames,
            sampling_rate=args.sampling_rate,
            transform=transform,
            random_temporal=not deterministic_debug,
        )
    elif mode in ("validation", "test"):
        csv_name = args.val_view_csv if mode == "validation" else args.test_view_csv
        anno_view1 = os.path.join(args.data_path, csv_name)
        transform = VideoTransform(
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            train=False,
        )
        return SingleViewDataset(
            csv_path=anno_view1,
            prefix=args.prefix,
            delimiter=args.csv_delimiter,
            clip_len=args.num_frames,
            sampling_rate=args.sampling_rate,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported mode for clean training: {mode}")


def normalize_model_name(model_name):
    normalized = MODEL_ALIASES.get(model_name)
    if normalized is None:
        supported = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(f"Unsupported --model '{model_name}'. Supported values: {supported}")
    return normalized


def parse_int_tuple(text):
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return tuple(values)


def build_model(args):
    model_name = normalize_model_name(args.model)
    args.model = model_name
    if model_name == "videomamba_small_clean":
        return create_videomamba_small_clean(
            img_size=args.input_size,
            num_classes=args.nb_classes,
            num_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            fc_drop_rate=args.fc_drop_rate,
            drop_path=args.drop_path,
            use_mean_pooling=args.use_mean_pooling,
        )

    if model_name == "videomamba_small_trainable_snn":
        from models.videomamba_trainable_snn import create_videomamba_small_trainable_snn

        return create_videomamba_small_trainable_snn(
            img_size=args.input_size,
            num_classes=args.nb_classes,
            num_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            fc_drop_rate=args.fc_drop_rate,
            drop_path=args.drop_path,
            use_mean_pooling=args.use_mean_pooling,
            spike_patch=args.snn_spike_patch,
            spike_block_indices=parse_int_tuple(args.snn_block_indices),
            spike_position=args.snn_spike_position,
            snn_timesteps=args.snn_timesteps,
            signed_spikes=args.snn_signed_spikes,
            threshold_init=args.snn_threshold_init,
            threshold_percentile=args.snn_threshold_percentile,
            train_threshold=args.snn_train_threshold,
            surrogate_alpha=args.snn_surrogate_alpha,
        )

    if model_name == "spikmamba_fixed":
        from models.videomamba_spik_baseline_1_fixed import spikmamba_fixed

        return spikmamba_fixed(
            pretrained=False,
            img_size_h=args.input_size,
            img_size_w=args.input_size,
            patch_size=args.spik_patch_size,
            embed_dims=args.spik_embed_dims,
            num_classes=args.nb_classes,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            kernel_size=args.tubelet_size,
            drop_path_rate=args.drop_path,
            T=args.spik_time_steps,
            bimamba=args.spik_bimamba,
        )

    raise AssertionError(f"Unhandled normalized model: {model_name}")


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


def is_bimamba_reverse_key(key):
    reverse_markers = (".A_b_log", ".D_b", ".conv1d_b.", ".x_proj_b.", ".dt_proj_b.")
    return any(marker in key for marker in reverse_markers)


def print_checkpoint_structure(checkpoint, state, model_state):
    if isinstance(checkpoint, dict):
        top_keys = list(checkpoint.keys())
        main_print(f"Checkpoint top-level keys ({len(top_keys)}): {top_keys[:12]}")
    ckpt_tensor_keys = [key for key, value in state.items() if torch.is_tensor(value)]
    model_tensor_keys = [key for key, value in model_state.items() if torch.is_tensor(value)]
    ckpt_reverse_keys = [key for key in ckpt_tensor_keys if is_bimamba_reverse_key(key)]
    model_reverse_keys = [key for key in model_tensor_keys if is_bimamba_reverse_key(key)]
    main_print(
        "Checkpoint tensors: "
        f"{len(ckpt_tensor_keys)} total, {len(ckpt_reverse_keys)} BiMamba reverse-branch tensors")
    main_print(
        "Model tensors: "
        f"{len(model_tensor_keys)} total, {len(model_reverse_keys)} BiMamba reverse-branch tensors")


def load_pretrained(model, path, model_key, min_load_ratio=0.05):
    if not path:
        return
    checkpoint = torch.load(path, map_location="cpu")
    state, used_key = extract_checkpoint_state(checkpoint, model_key)
    state = strip_prefixes(state)

    model_state = model.state_dict()
    print_checkpoint_structure(checkpoint, state, model_state)
    loadable = OrderedDict()
    skipped = []
    for key, value in state.items():
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
            loadable[key] = value
        else:
            skipped.append(key)

    load_ratio = len(loadable) / max(1, len(model_state))
    if load_ratio < min_load_ratio:
        main_print(
            f"Skipped pretrained checkpoint: only {len(loadable)}/{len(model_state)} "
            f"model tensors matched ({load_ratio:.2%}), below min_pretrained_load_ratio={min_load_ratio:.2%}.")
        main_print("Set --min_pretrained_load_ratio 0 to force this partial load.")
        return

    msg = model.load_state_dict(loadable, strict=False)
    main_print(f"Loaded pretrained checkpoint: {path}")
    main_print(f"Checkpoint key: {used_key}; loaded={len(loadable)} skipped={len(skipped)}")
    main_print(f"Missing keys: {len(msg.missing_keys)}; unexpected keys: {len(msg.unexpected_keys)}")
    if skipped[:8]:
        main_print(f"First skipped keys: {skipped[:8]}")


def build_teacher(args, device):
    if args.distill_weight <= 0:
        return None
    checkpoint_path = args.teacher_checkpoint or args.finetune
    if not checkpoint_path:
        raise ValueError("--distill_weight > 0 requires --teacher_checkpoint or --finetune.")
    teacher = create_videomamba_small_clean(
        img_size=args.input_size,
        num_classes=args.nb_classes,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        fc_drop_rate=0.0,
        drop_path=0.0,
        use_mean_pooling=args.use_mean_pooling,
    )
    load_pretrained(teacher, checkpoint_path, args.model_key, min_load_ratio=0.05)
    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    main_print(f"Using ANN teacher checkpoint: {checkpoint_path}")
    return teacher


def save_checkpoint(args, model, optimizer, scheduler, epoch, best_acc1, name):
    if not is_main_process():
        return
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(args.output_dir) / f"{name}.pth"
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
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
    unwrap_model(model).load_state_dict(state, strict=False)
    if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_acc1 = float(checkpoint.get("best_acc1", 0.0))
    main_print(f"Resumed from {path}; start_epoch={start_epoch}; best_acc1={best_acc1:.2f}")
    return start_epoch, best_acc1


def load_eval_checkpoint(model, path, device):
    if not path:
        raise ValueError("--eval requires --eval_checkpoint or --resume")
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    msg = unwrap_model(model).load_state_dict(state, strict=False)
    epoch = checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown"
    best_acc1 = checkpoint.get("best_acc1", None) if isinstance(checkpoint, dict) else None
    main_print(f"Loaded eval checkpoint: {path}")
    main_print(f"Eval checkpoint epoch: {epoch}")
    if best_acc1 is not None:
        main_print(f"Eval checkpoint best_acc1: {float(best_acc1):.2f}")
    main_print(f"Eval missing keys: {len(msg.missing_keys)}; unexpected keys: {len(msg.unexpected_keys)}")


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


def make_optimizer(model, args):
    base_params = []
    spike_params = []
    active_spike_params = active_spike_parameter_names(model)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        normalized_name = name[len("module."):] if name.startswith("module.") else name
        is_spike_like = "spike" in name or "log_threshold" in name
        if normalized_name in active_spike_params or (not active_spike_params and is_spike_like):
            spike_params.append(param)
        elif active_spike_params and is_spike_like:
            continue
        else:
            base_params.append(param)

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if spike_params:
        param_groups.append(
            {
                "params": spike_params,
                "lr": args.lr * args.spike_lr_multiplier,
                "weight_decay": 0.0,
            }
        )
        main_print(
            f"Spike parameter group: {len(spike_params)} tensors, "
            f"lr={args.lr * args.spike_lr_multiplier:.8g}"
        )
    return torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)


def amp_context(device, args):
    if device.type != "cuda" or not args.bf16:
        return contextlib.nullcontext()
    return torch.cuda.amp.autocast(dtype=torch.bfloat16)


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


def forward_train_logits(model, view1, view2):
    model_outputs = model(view1, view2, return_view_logits=True)
    if not isinstance(model_outputs, (tuple, list)) or len(model_outputs) < 3:
        raise RuntimeError("Model must return fused, view1, and view2 logits when return_view_logits=True.")
    return model_outputs[:3]


def distillation_kl(student_logits, teacher_logits, temperature):
    temperature = max(float(temperature), 1e-6)
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)


def reduce_metrics(totals, hists, device, prefix, elapsed=None):
    total_keys = list(totals)
    total_tensor = torch.tensor([float(totals[key]) for key in total_keys], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    reduced_totals = {key: total_tensor[index].item() for index, key in enumerate(total_keys)}

    count = max(1.0, reduced_totals.pop("count"))
    stats = {f"{prefix}_{key}": value / count for key, value in reduced_totals.items()}

    if elapsed is not None:
        elapsed_tensor = torch.tensor(float(elapsed), dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        stats[f"{prefix}_time"] = elapsed_tensor.item()

    for key, hist in hists.items():
        hist_tensor = hist.to(device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(hist_tensor, op=dist.ReduceOp.SUM)
        stats[f"{prefix}_{key}"] = hist_list(hist_tensor.cpu())
    return stats


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, args, teacher_model=None):
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "loss": 0.0,
        "fused_ce_loss": 0.0,
        "view_ce_loss": 0.0,
        "distill_loss": 0.0,
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
        sync_grad = (step + 1) % args.update_freq == 0 or (step + 1) == len(loader)
        with amp_context(device, args):
            reset_stateful_modules(model)
            fused_logits, view1_logits, view2_logits = forward_train_logits(model, view1, view2)
            fused_ce = criterion(fused_logits, target)
            view_ce = 0.5 * (criterion(view1_logits, target) + criterion(view2_logits, target))
            distill_loss = fused_logits.new_zeros(())
            if teacher_model is not None and args.distill_weight > 0:
                with torch.no_grad():
                    reset_stateful_modules(teacher_model)
                    teacher_fused, teacher_view1, teacher_view2 = forward_train_logits(teacher_model, view1, view2)
                distill_loss = (
                    distillation_kl(fused_logits, teacher_fused, args.distill_temperature)
                    + 0.5 * distillation_kl(view1_logits, teacher_view1, args.distill_temperature)
                    + 0.5 * distillation_kl(view2_logits, teacher_view2, args.distill_temperature)
                ) / 2.0
            loss = (
                args.fused_ce_loss_weight * fused_ce
                + args.view_ce_loss_weight * view_ce
                + args.distill_weight * distill_loss
            )
            loss_for_backward = loss / args.update_freq

        backward_context = (
            model.no_sync()
            if args.distributed and hasattr(model, "no_sync") and not sync_grad
            else contextlib.nullcontext()
        )
        with backward_context:
            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

        if sync_grad:
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
        totals["distill_loss"] += distill_loss.item() * batch_size
        totals["fused_acc1"] += fused_acc1 * batch_size
        totals["fused_acc5"] += fused_acc5 * batch_size
        totals["view1_acc1"] += view1_acc1 * batch_size
        totals["view2_acc1"] += view2_acc1 * batch_size
        totals["count"] += batch_size

        update_hist(hists["target_hist"], target)
        update_hist(hists["fused_pred_hist"], fused_logits.argmax(dim=1))
        update_hist(hists["view1_pred_hist"], view1_logits.argmax(dim=1))
        update_hist(hists["view2_pred_hist"], view2_logits.argmax(dim=1))

        if is_main_process() and step % max(1, args.print_freq) == 0:
            main_print(
                f"Epoch {epoch} [{step}/{len(loader)}] "
                f"loss={loss.item():.4f} acc1={fused_acc1:.2f} "
                f"view1={view1_acc1:.2f} view2={view2_acc1:.2f}"
            )

    return reduce_metrics(totals, hists, device, "train", elapsed=time.time() - start)


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
            reset_stateful_modules(model)
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

    return reduce_metrics(totals, hists, device, "val")


@torch.no_grad()
def initialize_spikes_from_train_batch(model, loader, device, args):
    if not active_spike_module_names(model):
        return
    was_training = model.training
    model.eval()
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
    view1, view2, _ = move_train_batch(batch, device)
    with amp_context(device, args):
        reset_stateful_modules(model)
        _ = forward_train_logits(model, view1, view2)
        reset_stateful_modules(model)
    model.train(was_training)
    main_print("Initialized spike thresholds from one training batch before initial validation.")


@torch.no_grad()
def dump_model_summary(args, model, device):
    if not is_main_process() or not args.dump_model_summary:
        return
    output_path = Path(args.output_dir) / "model_summary.txt"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    try:
        from torchinfo import summary
    except ImportError as exc:
        message = (
            "torchinfo is not installed; cannot write model summary.\n"
            "Install it with: pip install torchinfo\n"
            f"ImportError: {exc}\n"
        )
        output_path.write_text(message, encoding="utf-8")
        main_print(message.strip())
        return

    target = unwrap_model(model)
    was_training = target.training
    target.eval()
    reset_stateful_modules(target)
    dummy = torch.zeros(
        1,
        3,
        args.num_frames,
        args.input_size,
        args.input_size,
        device=device,
    )
    try:
        with amp_context(device, args):
            model_summary = summary(
                target,
                input_data=dummy,
                depth=args.summary_depth,
                col_names=("input_size", "output_size", "num_params", "trainable"),
                verbose=0,
            )
        output_path.write_text(str(model_summary) + "\n", encoding="utf-8")
        main_print(f"Saved torchinfo model summary to {output_path}")
    finally:
        reset_stateful_modules(target)
        target.train(was_training)


def write_log(args, stats, filename="log.txt"):
    if not is_main_process():
        return
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(stats, ensure_ascii=False) + "\n")


def write_run_metadata(args, model, teacher_model=None):
    if not is_main_process():
        return
    target = unwrap_model(model)
    teacher = unwrap_model(teacher_model) if teacher_model is not None else None
    metadata = {
        "args": vars(args),
        "model_class": target.__class__.__name__,
        "teacher_class": teacher.__class__.__name__ if teacher is not None else None,
        "active_spike_modules": active_spike_module_names(target),
        "active_spike_parameter_names": sorted(active_spike_parameter_names(target)),
        "spike_patch": bool(getattr(target, "spike_patch", False)),
        "spike_position": getattr(target, "spike_position", None),
        "spike_block_indices": list(getattr(target, "spike_block_indices", [])),
        "snn_timesteps": getattr(target, "snn_timesteps", None),
        "signed_spikes": getattr(target, "signed_spikes", None),
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)


def run(args):
    if args.fused_ce_loss_weight <= 0 and args.view_ce_loss_weight <= 0:
        raise ValueError("At least one of fused_ce_loss_weight or view_ce_loss_weight must be > 0.")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed + args.rank)
    if args.distributed and torch.cuda.is_available():
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = not args.disable_cudnn_benchmark

    main_print("Clean ET training configuration:")
    for key in sorted(vars(args)):
        main_print(f"  {key}: {getattr(args, key)}")

    if args.eval:
        train_dataset = None
        val_dataset = make_dataset(args, args.eval_split)
    else:
        train_dataset = make_dataset(args, "train")
        if args.debug_overfit_samples > 0:
            sample_count = min(len(train_dataset), max(args.debug_overfit_samples, args.batch_size))
            indices = balanced_subset_indices(train_dataset, sample_count, args.seed)
            labels = [label_key(train_dataset.label_array[index]) for index in indices]
            label_hist = OrderedDict()
            for label in labels:
                label_hist[label] = label_hist.get(label, 0) + 1
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
            main_print(f"Debug balanced subset: {len(indices)} samples; class histogram: {dict(label_hist)}")
        val_dataset = make_dataset(args, "validation")
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_sampler = None
    if not args.eval and args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            drop_last=True,
        )
    val_sampler = DistributedEvalSampler(val_dataset) if args.distributed else None

    if not args.eval:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=loader_generator if train_sampler is None else None,
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=max(1, int(1.5 * args.batch_size)),
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
    )

    model = build_model(args)
    main_print(f"Built model: {args.model} ({model.__class__.__name__})")
    spike_modules = active_spike_module_names(model)
    if spike_modules:
        main_print(f"Active spike modules ({len(spike_modules)}): {', '.join(spike_modules)}")
    else:
        main_print("Active spike modules: none")
    eval_checkpoint = args.eval_checkpoint or args.resume
    if not (args.eval and eval_checkpoint):
        load_pretrained(model, args.finetune, args.model_key, args.min_pretrained_load_ratio)
    model.to(device)
    teacher_model = None if args.eval else build_teacher(args, device)
    write_run_metadata(args, model, teacher_model)
    if args.distributed:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[args.local_rank], output_device=args.local_rank)
        model = DistributedDataParallel(model, **ddp_kwargs)

    criterion = torch.nn.CrossEntropyLoss()
    if args.eval:
        if eval_checkpoint:
            load_eval_checkpoint(model, eval_checkpoint, device)
        elif not args.finetune:
            raise ValueError("--eval requires either --eval_checkpoint, --resume, or --finetune")
        eval_stats = validate(model, val_loader, criterion, device, args)
        stats = {
            "mode": "eval",
            "split": args.eval_split,
            "checkpoint": eval_checkpoint or args.finetune,
            **eval_stats,
        }
        write_log(args, stats, filename=f"{args.eval_split}_log.txt")
        main_print(
            f"Eval {args.eval_split}: "
            f"acc1={stats['val_acc1']:.2f} "
            f"acc5={stats['val_acc5']:.2f} "
            f"loss={stats['val_loss']:.4f}"
        )
        return

    optimizer = make_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    if args.resume:
        args.start_epoch, best_acc1 = load_resume(model, optimizer, scheduler, args.resume, device)
    else:
        best_acc1 = 0.0

    if not args.skip_initial_eval or args.dump_model_summary:
        initialize_spikes_from_train_batch(model, train_loader, device, args)

    dump_model_summary(args, model, device)

    if not args.skip_initial_eval:
        initial_val_stats = validate(model, val_loader, criterion, device, args)
        initial_stats = {
            "mode": "initial_eval",
            "epoch": args.start_epoch - 1,
            "lr": optimizer.param_groups[0]["lr"],
            "checkpoint": args.resume or args.finetune,
            **initial_val_stats,
        }
        write_log(args, initial_stats)
        main_print(
            f"Initial validation before training: "
            f"val_acc1={initial_stats['val_acc1']:.2f} "
            f"val_acc5={initial_stats['val_acc5']:.2f} "
            f"val_loss={initial_stats['val_loss']:.4f}"
        )
        if not args.skip_initial_best_checkpoint and initial_stats["val_acc1"] >= best_acc1:
            best_acc1 = initial_stats["val_acc1"]
            save_checkpoint(args, model, optimizer, scheduler, args.start_epoch - 1, best_acc1, "best")

    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        main_print(f"Epoch {epoch}: lr={current_lr:.8g}")
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            args,
            teacher_model=teacher_model,
        )
        val_stats = validate(model, val_loader, criterion, device, args)
        scheduler.step()

        stats = {
            "epoch": epoch,
            "lr": current_lr,
            **train_stats,
            **val_stats,
        }
        write_log(args, stats)
        main_print(
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


def main():
    args = get_args()
    init_distributed_mode(args)
    try:
        run(args)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
