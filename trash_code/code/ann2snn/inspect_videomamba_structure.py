import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from ann2snn.spike_utils import dump_model_layer_order, load_clean_checkpoint
from ann2snn.videomamba_ann2snn import create_videomamba_small_ann2snn
from models.videomamba_clean import create_videomamba_small_clean


ACTIVATION_TYPES = (
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.GELU,
    nn.SiLU,
    nn.Hardswish,
    nn.Sigmoid,
    nn.Tanh,
    nn.Softplus,
)
RELU_TYPES = (nn.ReLU, nn.ReLU6)
STRUCTURE_TYPES = {"CleanVideoMamba", "VisionMamba", "PatchEmbed", "Block", "Mamba"}


def parse_args():
    parser = argparse.ArgumentParser("Inspect clean VideoMamba ANN/SNN structure")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", default="./outputs/ann2snn_structure/videomamba_structure.txt")
    parser.add_argument("--device", default="cpu")
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
    parser.add_argument(
        "--max_depth",
        type=int,
        default=-1,
        help="Limit printed module-tree depth. Use -1 for full tree.",
    )
    parser.add_argument(
        "--dump_forward_order",
        action="store_true",
        help="Also run a dummy forward pass and dump leaf execution order with tensor shapes.",
    )
    parser.set_defaults(use_mean_pooling=True, spike_patch=True, signed_spikes=True)
    return parser.parse_args()


def parse_block_indices(text):
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return tuple(values)


def build_model(args):
    common_kwargs = dict(
        num_classes=args.nb_classes,
        img_size=args.input_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        drop_path=args.drop_path,
        fc_drop_rate=args.fc_drop_rate,
        use_mean_pooling=args.use_mean_pooling,
    )
    if args.model_variant == "snn":
        return create_videomamba_small_ann2snn(
            **common_kwargs,
            spike_patch=args.spike_patch,
            spike_block_indices=parse_block_indices(args.spike_block_indices),
            signed_spikes=args.signed_spikes,
            threshold_scale=args.threshold_scale,
        )
    return create_videomamba_small_clean(**common_kwargs)


def direct_param_count(module):
    return sum(param.numel() for param in module.parameters(recurse=False))


def direct_buffer_count(module):
    return sum(buffer.numel() for buffer in module.buffers(recurse=False))


def module_depth(name):
    return 0 if not name else name.count(".") + 1


def describe_previous_module(module):
    if module is None:
        return "none", "unknown", "unknown"
    class_name = module.__class__.__name__
    if isinstance(module, (nn.BatchNorm1d, nn.Linear)):
        feature_count = getattr(module, "num_features", getattr(module, "out_features", "unknown"))
        return class_name, str(feature_count), "SpikingNeuron2d"
    if isinstance(module, (nn.BatchNorm2d, nn.Conv2d)):
        feature_count = getattr(module, "num_features", getattr(module, "out_channels", "unknown"))
        return class_name, str(feature_count), "SpikingNeuron4d/reference-style"
    if isinstance(module, (nn.BatchNorm3d, nn.Conv3d)):
        feature_count = getattr(module, "num_features", getattr(module, "out_channels", "unknown"))
        return class_name, str(feature_count), "SpikingNeuron5d"
    if isinstance(module, nn.Sequential) and len(module) > 0:
        return describe_previous_module(module[-1])
    return class_name, "unknown", "manual-check"


def collect_relu_candidates(model):
    candidates = []
    for parent_name, parent in model.named_modules():
        children = list(parent.named_children())
        for index, (child_name, child) in enumerate(children):
            if not isinstance(child, RELU_TYPES):
                continue
            prev_name = children[index - 1][0] if index > 0 else ""
            prev_module = children[index - 1][1] if index > 0 else None
            prev_class, feature_count, spike_hint = describe_previous_module(prev_module)
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            prev_full_name = f"{parent_name}.{prev_name}" if parent_name and prev_name else prev_name
            candidates.append(
                {
                    "name": full_name,
                    "class": child.__class__.__name__,
                    "parent": parent_name or "<root>",
                    "previous": prev_full_name or "none",
                    "previous_class": prev_class,
                    "features": feature_count,
                    "spike_hint": spike_hint,
                }
            )
    return candidates


def collect_activation_modules(model):
    records = []
    counts = Counter()
    for name, module in model.named_modules():
        if isinstance(module, ACTIVATION_TYPES):
            class_name = module.__class__.__name__
            records.append((name, class_name))
            counts[class_name] += 1
    return counts, records


def collect_structure_modules(model):
    records = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if class_name in STRUCTURE_TYPES:
            records.append((name or "<root>", class_name, direct_param_count(module)))
    return records


def format_module_tree(model, max_depth):
    lines = []
    for name, module in model.named_modules():
        depth = module_depth(name)
        if max_depth >= 0 and depth > max_depth:
            continue
        label = name or "<root>"
        child_count = len(list(module.children()))
        indent = "  " * depth
        lines.append(
            f"{indent}{label}: {module.__class__.__name__} "
            f"direct_params={direct_param_count(module)} "
            f"direct_buffers={direct_buffer_count(module)} "
            f"children={child_count}"
        )
    return lines


def write_structure_report(model, args, output_path, load_message=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    activation_counts, activation_records = collect_activation_modules(model)
    relu_candidates = collect_relu_candidates(model)
    structure_records = collect_structure_modules(model)
    total_params = sum(param.numel() for param in model.parameters())
    total_buffers = sum(buffer.numel() for buffer in model.buffers())
    total_modules = sum(1 for _ in model.modules())

    lines = [
        "VideoMamba structure inspection",
        "================================",
        f"model_variant: {args.model_variant}",
        f"checkpoint: {args.checkpoint or '<none>'}",
        f"total_modules: {total_modules}",
        f"total_params: {total_params}",
        f"total_buffers: {total_buffers}",
    ]
    if load_message is not None:
        lines.extend(
            [
                f"checkpoint_missing_keys: {len(load_message.missing_keys)}",
                f"checkpoint_unexpected_keys: {len(load_message.unexpected_keys)}",
            ]
        )

    lines.extend(["", "Main structure modules", "----------------------"])
    for name, class_name, direct_params in structure_records:
        lines.append(f"{name}\t{class_name}\tdirect_params={direct_params}")

    lines.extend(["", "Activation module summary", "-------------------------"])
    if activation_counts:
        for class_name, count in sorted(activation_counts.items()):
            lines.append(f"{class_name}: {count}")
    else:
        lines.append("No torch.nn activation modules found.")

    lines.extend(["", "Activation module list", "----------------------"])
    if activation_records:
        for name, class_name in activation_records:
            lines.append(f"{name}\t{class_name}")
    else:
        lines.append("none")

    lines.extend(["", "ReLU/ReLU6 replacement candidates", "--------------------------------"])
    if relu_candidates:
        for candidate in relu_candidates:
            lines.append(
                "{name}\t{class}\tparent={parent}\tprevious={previous}"
                "\tprevious_class={previous_class}\tfeatures={features}"
                "\tspike_hint={spike_hint}".format(**candidate)
            )
    else:
        lines.append("none")
        lines.append("Verdict: pure ReLU/ReLU6-to-spike replacement would convert 0 layers.")

    lines.extend(["", "Module tree", "-----------"])
    lines.extend(format_module_tree(model, args.max_depth))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_dump_forward_order(model, args, output_path):
    if not args.dump_forward_order:
        return None
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model.to(device).eval()
    sample = torch.zeros(1, 3, args.num_frames, args.input_size, args.input_size, device=device)
    forward_path = output_path.with_name(f"{output_path.stem}_forward_order.txt")
    dump_model_layer_order(model, sample, forward_path)
    return forward_path


def main():
    args = parse_args()
    output_path = Path(args.output)
    model = build_model(args)

    load_message = None
    if args.checkpoint:
        _, load_message = load_clean_checkpoint(model, args.checkpoint, device="cpu")

    write_structure_report(model, args, output_path, load_message=load_message)
    print(f"Saved structure report to {output_path}")

    try:
        forward_path = maybe_dump_forward_order(model, args, output_path)
    except Exception as exc:
        forward_path = None
        print(f"Forward-order dump failed: {exc.__class__.__name__}: {exc}")
    if forward_path is not None:
        print(f"Saved forward order to {forward_path}")


if __name__ == "__main__":
    main()
