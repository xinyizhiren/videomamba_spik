import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from decord import VideoReader, cpu
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def read_video_csv(path, delimiter=","):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if not row or len(row) < 2:
                continue
            video_path = row[0].strip()
            if not video_path:
                continue
            rows.append((video_path, int(row[1])))
    if not rows:
        raise ValueError(f"No valid video rows found in {path}")
    return rows


def make_sample_id(sample):
    sample = str(sample).replace("\\", "/").strip().lstrip("./")
    return os.path.splitext(sample)[0]


def resolve_path(prefix, sample):
    sample_path = Path(str(sample))
    if sample_path.is_absolute():
        return str(sample_path)
    return str(Path(prefix) / sample_path)


def sample_frame_indices(video_len, clip_len, sampling_rate, random_sample):
    if video_len <= 0:
        raise ValueError("Cannot sample frames from an empty video")
    span = clip_len * max(1, sampling_rate)
    if video_len >= span:
        start = random.randint(0, video_len - span) if random_sample else (video_len - span) // 2
        indices = start + np.arange(clip_len) * max(1, sampling_rate)
    else:
        indices = np.linspace(0, video_len - 1, clip_len)
    return np.clip(indices, 0, video_len - 1).astype(np.int64).tolist()


def load_video(path, clip_len, sampling_rate, random_sample):
    reader = VideoReader(path, num_threads=1, ctx=cpu(0))
    indices = sample_frame_indices(len(reader), clip_len, sampling_rate, random_sample)
    return reader.get_batch(indices).asnumpy()


def to_tensor(frames):
    frames = torch.from_numpy(frames).float() / 255.0
    return frames.permute(0, 3, 1, 2).contiguous()


def resize_short_side(frames, short_side):
    _, _, height, width = frames.shape
    if height == short_side and width == short_side:
        return frames
    if height < width:
        new_height = short_side
        new_width = int(round(width * short_side / height))
    else:
        new_width = short_side
        new_height = int(round(height * short_side / width))
    return torch_f.interpolate(frames, size=(new_height, new_width), mode="bilinear", align_corners=False)


def center_crop(frames, crop_size):
    _, _, height, width = frames.shape
    top = max(0, (height - crop_size) // 2)
    left = max(0, (width - crop_size) // 2)
    return frames[:, :, top:top + crop_size, left:left + crop_size]


def random_resized_crop_params(height, width, scale, ratio):
    area = height * width
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * random.uniform(scale[0], scale[1])
        aspect_ratio = math.exp(random.uniform(log_ratio[0], log_ratio[1]))
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = random.randint(0, height - crop_height)
            left = random.randint(0, width - crop_width)
            return top, left, crop_height, crop_width

    in_ratio = width / height
    if in_ratio < ratio[0]:
        crop_width = width
        crop_height = int(round(crop_width / ratio[0]))
    elif in_ratio > ratio[1]:
        crop_height = height
        crop_width = int(round(crop_height * ratio[1]))
    else:
        crop_width = width
        crop_height = height
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return top, left, crop_height, crop_width


def random_resized_crop(frames, crop_size, scale, ratio):
    _, _, height, width = frames.shape
    top, left, crop_height, crop_width = random_resized_crop_params(height, width, scale, ratio)
    frames = frames[:, :, top:top + crop_height, left:left + crop_width]
    return torch_f.interpolate(frames, size=(crop_size, crop_size), mode="bilinear", align_corners=False)


def normalize(frames):
    return (frames - IMAGENET_MEAN.to(frames.device)) / IMAGENET_STD.to(frames.device)


class VideoTransform:
    def __init__(
            self,
            crop_size=224,
            short_side_size=224,
            train=True,
            deterministic=False,
            crop_scale=(0.5, 1.0),
            crop_ratio=(0.75, 1.3333),
            horizontal_flip=False):
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.train = train
        self.deterministic = deterministic
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.horizontal_flip = horizontal_flip

    def __call__(self, frames):
        frames = to_tensor(frames)
        frames = resize_short_side(frames, self.short_side_size)
        if self.train and not self.deterministic:
            frames = random_resized_crop(frames, self.crop_size, self.crop_scale, self.crop_ratio)
            if self.horizontal_flip and random.random() < 0.5:
                frames = torch.flip(frames, dims=[-1])
        else:
            frames = center_crop(frames, self.crop_size)
            if frames.shape[-2:] != (self.crop_size, self.crop_size):
                frames = torch_f.interpolate(
                    frames, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)
        frames = normalize(frames)
        return frames.permute(1, 0, 2, 3).contiguous()


class CrossViewTrainDataset(Dataset):
    def __init__(
            self,
            view1_csv,
            view2_csv,
            prefix="",
            delimiter=",",
            clip_len=16,
            sampling_rate=4,
            transform=None,
            random_temporal=True):
        self.samples_view1 = read_video_csv(view1_csv, delimiter)
        self.samples_view2 = read_video_csv(view2_csv, delimiter)
        if len(self.samples_view1) != len(self.samples_view2):
            raise ValueError(
                f"View CSV length mismatch: {len(self.samples_view1)} vs {len(self.samples_view2)}")

        self.prefix = prefix
        self.clip_len = clip_len
        self.sampling_rate = sampling_rate
        self.transform = transform or VideoTransform(train=True)
        self.random_temporal = random_temporal
        self.label_array = []
        for index, ((_, label1), (_, label2)) in enumerate(zip(self.samples_view1, self.samples_view2)):
            if label1 != label2:
                raise ValueError(f"Label mismatch at row {index}: view1={label1}, view2={label2}")
            self.label_array.append(label1)

    def __len__(self):
        return len(self.samples_view1)

    def __getitem__(self, index):
        sample1, label = self.samples_view1[index]
        sample2, _ = self.samples_view2[index]
        path1 = resolve_path(self.prefix, sample1)
        path2 = resolve_path(self.prefix, sample2)
        frames1 = load_video(path1, self.clip_len, self.sampling_rate, self.random_temporal)
        frames2 = load_video(path2, self.clip_len, self.sampling_rate, self.random_temporal)
        return self.transform(frames1), self.transform(frames2), torch.tensor(label, dtype=torch.long)


class SingleViewDataset(Dataset):
    def __init__(
            self,
            csv_path,
            prefix="",
            delimiter=",",
            clip_len=16,
            sampling_rate=4,
            transform=None):
        self.samples = read_video_csv(csv_path, delimiter)
        self.prefix = prefix
        self.clip_len = clip_len
        self.sampling_rate = sampling_rate
        self.transform = transform or VideoTransform(train=False)
        self.label_array = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample, label = self.samples[index]
        path = resolve_path(self.prefix, sample)
        frames = load_video(path, self.clip_len, self.sampling_rate, random_sample=False)
        return self.transform(frames), torch.tensor(label, dtype=torch.long), make_sample_id(sample)
