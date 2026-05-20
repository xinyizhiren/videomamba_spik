import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ann2snn.spike_utils import (
    cal_delay_time_video,
    dump_model_layer_order,
    evaluate_ann_classifier,
    evaluate_snn_classifier,
    load_clean_checkpoint,
    weight_scaling_iter_video,
)
from ann2snn.videomamba_ann2snn import create_videomamba_small_ann2snn
from datasets.multiview_action_clean import SingleViewDataset, VideoTransform
from models.videomamba_clean import create_videomamba_small_clean


def parse_args():
    parser = argparse.ArgumentParser("Convert clean VideoMamba ANN checkpoint to SNN")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="./outputs/ann2snn_videomamba")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--csv_delimiter", default=",")
    parser.add_argument("--calib_view_csv", default="aligned_v01_1.csv")
    parser.add_argument("--val_view_csv", default="v03_val_set.csv")
    parser.add_argument("--test_view_csv", default="v03_test_set.csv")
    parser.add_argument("--calibration_samples", type=int, default=256)
    parser.add_argument("--calibration_steps", type=int, default=200)

    parser.add_argument("--nb_classes", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--short_side_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--sampling_rate", type=int, default=4)
    parser.add_argument("--tubelet_size", type=int, default=1)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--fc_drop_rate", type=float, default=0.0)
    parser.add_argument("--use_cls", action="store_false", dest="use_mean_pooling")
    parser.set_defaults(use_mean_pooling=True)

    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--delay", type=int, default=-1)
    parser.add_argument("--spike_block_indices", default="0,1")
    parser.add_argument("--spike_patch", action="store_true")
    parser.add_argument("--no_spike_patch", action="store_false", dest="spike_patch")
    parser.add_argument("--unsigned_spikes", action="store_false", dest="signed_spikes")
    parser.add_argument("--threshold_scale", type=float, default=1.0)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--dump_layer_order", action="store_true")
    parser.set_defaults(spike_patch=True, signed_spikes=True)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_block_indices(text):
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return tuple(values)


def make_loader(args, csv_name, batch_size, limit=None):
    transform = VideoTransform(
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        train=False,
    )
    dataset = SingleViewDataset(
        csv_path=str(Path(args.data_path) / csv_name),
        prefix=args.prefix,
        delimiter=args.csv_delimiter,
        clip_len=args.num_frames,
        sampling_rate=args.sampling_rate,
        transform=transform,
    )
    if limit is not None and limit > 0 and len(dataset) > limit:
        dataset = Subset(dataset, list(range(limit)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )


def summarize_load(msg, title):
    print(f"{title}: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys[:10]:
        print(f"{title} first missing: {msg.missing_keys[:10]}")
    if msg.unexpected_keys[:10]:
        print(f"{title} first unexpected: {msg.unexpected_keys[:10]}")


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spike_block_indices = parse_block_indices(args.spike_block_indices)

    ann_model = create_videomamba_small_clean(
        num_classes=args.nb_classes,
        img_size=args.input_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        drop_path=args.drop_path,
        fc_drop_rate=args.fc_drop_rate,
        use_mean_pooling=args.use_mean_pooling,
    )
    snn_model = create_videomamba_small_ann2snn(
        num_classes=args.nb_classes,
        img_size=args.input_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        drop_path=args.drop_path,
        fc_drop_rate=args.fc_drop_rate,
        use_mean_pooling=args.use_mean_pooling,
        spike_patch=args.spike_patch,
        spike_block_indices=spike_block_indices,
        signed_spikes=args.signed_spikes,
        threshold_scale=args.threshold_scale,
    )

    _, ann_msg = load_clean_checkpoint(ann_model, args.checkpoint, device="cpu")
    checkpoint, snn_msg = load_clean_checkpoint(snn_model, args.checkpoint, device="cpu")
    summarize_load(ann_msg, "ANN load")
    summarize_load(snn_msg, "SNN load")
    ann_model.to(device)
    snn_model.to(device)

    calib_loader = make_loader(
        args,
        args.calib_view_csv,
        batch_size=args.batch_size,
        limit=args.calibration_samples,
    )
    val_loader = make_loader(args, args.val_view_csv, batch_size=args.batch_size)
    test_loader = make_loader(args, args.test_view_csv, batch_size=args.batch_size)

    if args.dump_layer_order:
        first_batch = next(iter(calib_loader))
        sample = first_batch[0][:1].to(device)
        ann_order_path = output_dir / "videomamba_ann_layer_order.txt"
        snn_order_path = output_dir / "videomamba_snn_layer_order.txt"
        dump_model_layer_order(ann_model, sample, ann_order_path)
        dump_model_layer_order(snn_model, sample, snn_order_path)
        print(f"Saved ANN layer order to {ann_order_path}")
        print(f"Saved SNN layer order to {snn_order_path}")

    calibration_stats = weight_scaling_iter_video(
        calib_loader,
        ann_model,
        snn_model,
        device,
        args.calibration_steps,
    )
    delay = args.delay if args.delay >= 0 else cal_delay_time_video(calib_loader, snn_model, device)
    print(f"Estimated delay: {delay}")

    metrics = {
        "checkpoint": args.checkpoint,
        "timesteps": args.timesteps,
        "delay": delay,
        "spike_patch": args.spike_patch,
        "spike_block_indices": list(spike_block_indices),
        "signed_spikes": args.signed_spikes,
        "threshold_scale": args.threshold_scale,
        "calibration_samples": args.calibration_samples,
        "calibration_steps": args.calibration_steps,
        **calibration_stats,
    }

    if not args.skip_eval:
        metrics.update(evaluate_ann_classifier(val_loader, ann_model, device))
        metrics.update(evaluate_snn_classifier(val_loader, snn_model, device, args.timesteps, delay))
        test_metrics = evaluate_snn_classifier(test_loader, snn_model, device, args.timesteps, delay)
        metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    save_path = output_dir / "videomamba_ann2snn.pth"
    torch.save(
        {
            "model": snn_model.state_dict(),
            "ann_checkpoint": args.checkpoint,
            "timesteps": args.timesteps,
            "delay": delay,
            "metrics": metrics,
            "args": vars(args),
        },
        save_path,
    )
    print(f"Saved SNN checkpoint to {save_path}")

    summary_path = output_dir / "conversion_summary.json"
    summary_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}")

    if isinstance(checkpoint, dict) and "best_acc1" in checkpoint:
        print(f"Source ANN checkpoint best_acc1: {float(checkpoint['best_acc1']):.4f}")


if __name__ == "__main__":
    main()
