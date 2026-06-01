import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ann2snn.spike_utils import dump_model_layer_order, load_clean_checkpoint
from ann2snn.videomamba_ann2snn import create_videomamba_small_ann2snn
from models.videomamba_clean import create_videomamba_small_clean


def parse_args():
    parser = argparse.ArgumentParser("Dump VideoMamba module execution order")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", default="./videomamba_layer_order.txt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nb_classes", type=int, default=12)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--tubelet_size", type=int, default=1)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--fc_drop_rate", type=float, default=0.0)
    parser.add_argument("--model_variant", default="ann", choices=("ann", "snn"))
    parser.add_argument("--spike_block_indices", default="0,1")
    parser.add_argument("--spike_patch", action="store_true")
    parser.add_argument("--no_spike_patch", action="store_false", dest="spike_patch")
    parser.add_argument("--unsigned_spikes", action="store_false", dest="signed_spikes")
    parser.add_argument("--threshold_scale", type=float, default=1.0)
    parser.add_argument("--use_cls", action="store_false", dest="use_mean_pooling")
    parser.set_defaults(use_mean_pooling=True, spike_patch=True, signed_spikes=True)
    return parser.parse_args()


def parse_block_indices(text):
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return tuple(values)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.model_variant == "snn":
        model = create_videomamba_small_ann2snn(
            num_classes=args.nb_classes,
            img_size=args.input_size,
            num_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            drop_path=args.drop_path,
            fc_drop_rate=args.fc_drop_rate,
            use_mean_pooling=args.use_mean_pooling,
            spike_patch=args.spike_patch,
            spike_block_indices=parse_block_indices(args.spike_block_indices),
            signed_spikes=args.signed_spikes,
            threshold_scale=args.threshold_scale,
        )
    else:
        model = create_videomamba_small_clean(
            num_classes=args.nb_classes,
            img_size=args.input_size,
            num_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            drop_path=args.drop_path,
            fc_drop_rate=args.fc_drop_rate,
            use_mean_pooling=args.use_mean_pooling,
        )
    if args.checkpoint:
        _, msg = load_clean_checkpoint(model, args.checkpoint, device="cpu")
        print(f"Loaded checkpoint: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.to(device).eval()
    sample = torch.zeros(1, 3, args.num_frames, args.input_size, args.input_size, device=device)
    output_path = Path(args.output)
    dump_model_layer_order(model, sample, output_path)
    print(f"Saved layer order to {output_path}")


if __name__ == "__main__":
    main()
