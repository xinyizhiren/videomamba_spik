import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from ann2snn.slayers import SpikingNeuron2d, SpikingNeuron3dSeq, SpikingNeuron5d


SPIKE_LAYER_TYPES = (SpikingNeuron2d, SpikingNeuron3dSeq, SpikingNeuron5d)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(1, self.count)

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += int(n)


def topk_accuracy(logits, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = min(max(topk), logits.shape[1])
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        results = []
        for k in topk:
            kk = min(k, logits.shape[1])
            correct_k = correct[:kk].reshape(-1).float().sum(0)
            results.append(correct_k.mul_(100.0 / target.numel()).item())
        return results


def get_video_tensor_from_batch(batch, device):
    video = batch[0].to(device, non_blocking=True)
    target = batch[1].to(device, non_blocking=True).long() if len(batch) > 1 else None
    return video, target


def iter_spike_layers(model):
    for module in model.modules():
        if isinstance(module, SPIKE_LAYER_TYPES):
            yield module


def set_spike_mode(model, mode):
    for module in iter_spike_layers(model):
        module.mode = mode


def reset_model(model):
    for module in iter_spike_layers(model):
        module.reset()


def load_clean_checkpoint(model, checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    msg = model.load_state_dict(state, strict=False)
    return checkpoint, msg


def swap_adjacent_bn_maxpool(module):
    if isinstance(module, nn.Sequential):
        layers = list(module.children())
        for idx in range(len(layers) - 1):
            is_bn = isinstance(layers[idx], (nn.BatchNorm2d, nn.BatchNorm3d))
            is_pool = isinstance(layers[idx + 1], (nn.MaxPool2d, nn.MaxPool3d))
            if is_bn and is_pool:
                layers[idx], layers[idx + 1] = layers[idx + 1], layers[idx]
                module.__init__(*layers)
    for child in module.children():
        swap_adjacent_bn_maxpool(child)


def weight_scaling_iter_video(dataloader, ann_model, snn_model, device, iter_steps):
    ann_model.eval().to(device)
    snn_model.eval().to(device)
    set_spike_mode(snn_model, "ann")

    global_step = 0
    pbar = tqdm(total=iter_steps, desc="ANN2SNN calibrating", leave=False)
    mse_meter = AverageMeter()
    delta_meter = AverageMeter()

    with torch.no_grad():
        while global_step < iter_steps:
            for batch in dataloader:
                if global_step >= iter_steps:
                    break
                video, _ = get_video_tensor_from_batch(batch, device)
                ann_logits = ann_model(video)
                snn_logits = snn_model(video)
                mse = F.mse_loss(snn_logits.float(), ann_logits.float()).item()
                deltas = [layer.delta for layer in iter_spike_layers(snn_model)]
                avg_delta = sum(deltas) / max(1, len(deltas))

                mse_meter.update(mse, video.size(0))
                delta_meter.update(avg_delta, 1)
                global_step += 1
                pbar.update(1)
                pbar.set_postfix(mse=f"{mse_meter.avg:.6f}", delta=f"{delta_meter.avg:.6f}")

    pbar.close()
    reset_model(snn_model)
    return {"calibration_mse": mse_meter.avg, "calibration_delta": delta_meter.avg}


def cal_delay_time_video(dataloader, model, device, max_batches=1):
    model.eval().to(device)
    set_spike_mode(model, "clip")
    seen = 0
    with torch.no_grad():
        for batch in dataloader:
            video, _ = get_video_tensor_from_batch(batch, device)
            _ = model(video)
            seen += 1
            if seen >= max_batches:
                break
    delay = int(sum(float(getattr(layer, "r", 0.0)) for layer in iter_spike_layers(model)))
    reset_model(model)
    return delay


def evaluate_ann_classifier(dataloader, model, device):
    model.eval().to(device)
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    with torch.no_grad():
        for batch in dataloader:
            video, target = get_video_tensor_from_batch(batch, device)
            logits = model(video)
            acc1, acc5 = topk_accuracy(logits, target, topk=(1, 5))
            acc1_meter.update(acc1, video.size(0))
            acc5_meter.update(acc5, video.size(0))
    return {"ann_acc1": acc1_meter.avg, "ann_acc5": acc5_meter.avg}


def evaluate_snn_classifier(dataloader, model, device, timesteps, delay=0):
    model.eval().to(device)
    set_spike_mode(model, "snn")
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    with torch.no_grad():
        for batch in dataloader:
            video, target = get_video_tensor_from_batch(batch, device)
            reset_model(model)
            total_logits = None
            valid_steps = 0
            for step in range(int(timesteps)):
                logits = model(video)
                if step >= int(delay):
                    total_logits = logits if total_logits is None else total_logits + logits
                    valid_steps += 1
            if total_logits is None:
                total_logits = logits
                valid_steps = 1
            total_logits = total_logits / float(valid_steps)
            acc1, acc5 = topk_accuracy(total_logits, target, topk=(1, 5))
            acc1_meter.update(acc1, video.size(0))
            acc5_meter.update(acc5, video.size(0))
            reset_model(model)

    return {"snn_acc1": acc1_meter.avg, "snn_acc5": acc5_meter.avg}


def dump_model_layer_order(model, sample, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    hooks = []
    counter = {"idx": 0}

    def make_hook(name):
        def hook(_module, _inputs, outputs):
            counter["idx"] += 1
            if isinstance(outputs, (tuple, list)):
                shape = [tuple(out.shape) for out in outputs if hasattr(out, "shape")]
            elif hasattr(outputs, "shape"):
                shape = tuple(outputs.shape)
            else:
                shape = str(type(outputs))
            records.append(
                {
                    "order": counter["idx"],
                    "name": name,
                    "class": _module.__class__.__name__,
                    "shape": shape,
                }
            )
        return hook

    for name, module in model.named_modules():
        if name and len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        if isinstance(sample, (tuple, list)):
            _ = model(*sample)
        else:
            _ = model(sample)

    for hook in hooks:
        hook.remove()

    with output_path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(
                f"{item['order']:04d}\t{item['name']}\t{item['class']}\t{item['shape']}\n"
            )
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records
