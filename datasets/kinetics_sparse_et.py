import os
import os
import io
import random
import numpy as np
from numpy.lib.function_base import disp
import torch
from torchvision import transforms
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from .random_erasing import RandomErasing
from .video_transforms import (
    Compose, Resize, CenterCrop, Normalize,
    create_random_augment, random_short_side_scale_jitter,
    random_crop, random_resized_crop_with_shift, random_resized_crop,
    horizontal_flip, random_short_side_scale_jitter, uniform_crop,
)
from .volume_transforms import ClipToTensor
import cv2  # 添加cv2导入用于图像处理

try:
    from petrel_client.client import Client

    has_client = True
except ImportError:
    has_client = False


# 添加视频扰动函数
def add_gaussian_blur(buffer, kernel_size=5, sigma=0):
    """添加高斯模糊"""
    # 计算适当的sigma值，较小的sigma会减弱模糊效果
    if sigma <= 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8  # 比OpenCV默认值小

    # 检查输入维度，支持单个视频或批量视频
    if len(buffer.shape) == 4:  # 单个视频 (T, H, W, C)
        blurred_buffer = np.zeros_like(buffer)
        for i in range(buffer.shape[0]):  # 遍历每一帧
            blurred_buffer[i] = cv2.GaussianBlur(buffer[i], (kernel_size, kernel_size), sigma)
        return blurred_buffer
    elif len(buffer.shape) == 5:  # 批量视频 (B, T, H, W, C)
        blurred_buffer = np.zeros_like(buffer)
        for b in range(buffer.shape[0]):  # 遍历每个视频
            for i in range(buffer.shape[1]):  # 遍历每一帧
                blurred_buffer[b, i] = cv2.GaussianBlur(buffer[b, i], (kernel_size, kernel_size), sigma)
        return blurred_buffer
    else:
        raise ValueError(f"Unsupported buffer shape: {buffer.shape}")


def add_rain(buffer, rain_density=0.01, rain_length=20, rain_width=1, rain_color=(200, 200, 200)):
    """添加雨点效果"""
    # 检查输入维度，支持单个视频或批量视频
    if len(buffer.shape) == 4:  # 单个视频 (T, H, W, C)
        rain_buffer = buffer.copy()
        T, H, W, C = buffer.shape

        # 对每一帧添加雨点
        for t in range(T):
            # 随机生成雨点的起始位置
            num_drops = int(H * W * rain_density)
            for _ in range(num_drops):
                # 随机选择雨点的起始位置
                x = np.random.randint(0, W)
                y = np.random.randint(0, H - rain_length)

                # 绘制雨点（垂直线）
                for i in range(rain_length):
                    if y + i < H:
                        for j in range(rain_width):
                            if x + j < W:
                                rain_buffer[t, y + i, x + j] = rain_color

        return rain_buffer
    elif len(buffer.shape) == 5:  # 批量视频 (B, T, H, W, C)
        rain_buffer = buffer.copy()
        B, T, H, W, C = buffer.shape

        # 对每个视频的每一帧添加雨点
        for b in range(B):
            for t in range(T):
                # 随机生成雨点的起始位置
                num_drops = int(H * W * rain_density)
                for _ in range(num_drops):
                    # 随机选择雨点的起始位置
                    x = np.random.randint(0, W)
                    y = np.random.randint(0, H - rain_length)

                    # 绘制雨点（垂直线）
                    for i in range(rain_length):
                        if y + i < H:
                            for j in range(rain_width):
                                if x + j < W:
                                    rain_buffer[b, t, y + i, x + j] = rain_color

        return rain_buffer
    else:
        raise ValueError(f"Unsupported buffer shape: {buffer.shape}")


def add_fog(buffer, fog_density=0.3):
    """添加雾效果"""
    # 检查输入维度，支持单个视频或批量视频
    if len(buffer.shape) == 4:  # 单个视频 (T, H, W, C)
        fog_buffer = buffer.copy()
        T, H, W, C = buffer.shape

        # 创建雾效果（白色半透明层）
        for t in range(T):
            # 创建白色雾层
            fog = np.ones_like(buffer[t]) * 255

            # 添加随机性，使雾看起来更自然
            noise = np.random.normal(0, 20, (H, W, C)).astype(np.float32)
            fog = np.clip(fog + noise, 0, 255).astype(np.uint8)

            # 将雾与原始图像混合
            fog_buffer[t] = cv2.addWeighted(buffer[t], 1 - fog_density, fog, fog_density, 0)

        return fog_buffer
    elif len(buffer.shape) == 5:  # 批量视频 (B, T, H, W, C)
        fog_buffer = buffer.copy()
        B, T, H, W, C = buffer.shape

        # 对每个视频的每一帧添加雾效果
        for b in range(B):
            for t in range(T):
                # 创建白色雾层
                fog = np.ones_like(buffer[b, t]) * 255

                # 添加随机性，使雾看起来更自然
                noise = np.random.normal(0, 20, (H, W, C)).astype(np.float32)
                fog = np.clip(fog + noise, 0, 255).astype(np.uint8)

                # 将雾与原始图像混合
                fog_buffer[b, t] = cv2.addWeighted(buffer[b, t], 1 - fog_density, fog, fog_density, 0)

        return fog_buffer
    else:
        raise ValueError(f"Unsupported buffer shape: {buffer.shape}")


def add_gaussian_noise(buffer, mean=0, std=25):
    """添加高斯噪声"""
    # 检查输入维度，支持单个视频或批量视频
    if len(buffer.shape) == 4 or len(buffer.shape) == 5:  # 单个视频或批量视频
        noise = np.random.normal(mean, std, buffer.shape).astype(np.float32)
        noisy_buffer = buffer.astype(np.float32) + noise
        return np.clip(noisy_buffer, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported buffer shape: {buffer.shape}")


def adjust_contrast(buffer, factor=1.5):
    """调整对比度"""
    # 检查输入维度，支持单个视频或批量视频
    if len(buffer.shape) == 4:  # 单个视频 (T, H, W, C)
        contrast_buffer = np.zeros_like(buffer)
        for i in range(buffer.shape[0]):
            # 计算平均亮度
            mean = np.mean(buffer[i].astype(np.float32))

            # 调整对比度: 新像素 = (原像素 - 平均值) * 因子 + 平均值
            contrast_buffer[i] = np.clip((buffer[i].astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

        return contrast_buffer
    elif len(buffer.shape) == 5:  # 批量视频 (B, T, H, W, C)
        contrast_buffer = np.zeros_like(buffer)
        for b in range(buffer.shape[0]):
            for i in range(buffer.shape[1]):
                # 计算平均亮度
                mean = np.mean(buffer[b, i].astype(np.float32))

                # 调整对比度: 新像素 = (原像素 - 平均值) * 因子 + 平均值
                contrast_buffer[b, i] = np.clip((buffer[b, i].astype(np.float32) - mean) * factor + mean, 0,
                                                255).astype(np.uint8)

        return contrast_buffer
    else:
        raise ValueError(f"Unsupported buffer shape: {buffer.shape}")


def apply_perturbation(buffer, perturbation_type, severity=2):
    """根据扰动类型应用相应的扰动"""
    if perturbation_type is None or perturbation_type == 'none':
        return buffer

    # 输出调试信息
    print(f"应用扰动: {perturbation_type}, 严重程度: {severity}, 输入形状: {buffer.shape}, 数据类型: {buffer.dtype}")

    # 根据严重程度调整参数，适当增强扰动强度
    if perturbation_type == 'gaussian_blur':
        # 高斯模糊核大小必须是奇数且大于1
        kernel_size = 2+1* severity  # 最小有效核大小
        sigma = 1.5   # 0.5, 1.0 - 适度的模糊
        print(f"高斯模糊核大小: {kernel_size}, sigma: {sigma}")
        return add_gaussian_blur(buffer, kernel_size=kernel_size, sigma=sigma)

    elif perturbation_type == 'rain':
        if severity==1:
            rain_density=0.0015
        if severity==3:
            rain_density=0.005
        # rain_density = 0.005 * severity  # 0.0005, 0.001 - 适度的雨点密度
        print(f"雨点密度: {rain_density}")
        return add_rain(buffer, rain_density=rain_density)

    elif perturbation_type == 'fog':
        if severity==1:
            fog_density=0.03
        if severity==3:
            fog_density=0.1
        # fog_density = 0.1 * severity  # 0.01, 0.02 - 适度的雾密度
        print(f"雾密度: {fog_density}")
        return add_fog(buffer, fog_density=fog_density)

    elif perturbation_type == 'gaussian_noise':
        if severity==1:
            std = 6
        if severity==3:
            std = 15
        # std = 5 * severity  # 2, 4 - 适度的噪声标准差
        print(f"高斯噪声标准差: {std}")
        return add_gaussian_noise(buffer, std=std)

    elif perturbation_type == 'contrast':
        # 对比度变化：适度变化
        if severity==1:
            contrast =1.1
        if severity==3:
            contrast =0.75
        # contrast = 0.5 + severity * 0.25  # 1.0, 1.1 - 适度的对比度变化
        print(f"对比度因子: {contrast}")
        return adjust_contrast(buffer, factor=contrast)

    return buffer


class VideoClsDataset_sparse(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path_view1, anno_path_view2, prefix='', split=' ', mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,
                 args=None, perturbation_type=None, perturbation_severity=2):
        self.anno_path_view1 = anno_path_view1
        self.anno_path_view2 = anno_path_view2
        self.prefix = prefix
        self.split = split
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False  # 随机擦除
        # 新增：扰动类型和严重程度
        self.perturbation_type = perturbation_type
        self.perturbation_severity = perturbation_severity

        assert num_segment == 1
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        import pandas as pd

        # 读取两个视角的注释文件
        cleaned_view1 = pd.read_csv(self.anno_path_view1, header=None, delimiter=self.split)
        # cleaned_view2 = pd.read_csv(self.anno_path_view2, header=None, delimiter=self.split)
        self.dataset_samples_view1 = list(cleaned_view1.values[:, 0])
        # self.dataset_samples_view2 = list(cleaned_view2.values[:, 0])
        self.label_array = list(cleaned_view1.values[:, 1])  # 视角的标签相同

        # cleaned = pd.read_csv(self.anno_path, header=None, delimiter=self.split)
        # self.dataset_samples = list(cleaned.values[:, 0])
        # self.label_array = list(cleaned.values[:, 1])

        self.client = None
        if has_client:
            self.client = Client('~/petreloss.conf')

        if mode == 'train':
            cleaned_view2 = pd.read_csv(self.anno_path_view2, header=None, delimiter=self.split)
            self.dataset_samples_view2 = list(cleaned_view2.values[:, 0])
            labels_view2 = list(cleaned_view2.values[:, 1])

            # 验证标签一致性
            for idx, (lv1, lv2) in enumerate(zip(self.label_array, labels_view2)):
                assert str(lv1) == str(lv2), f"样本索引 {idx} 的标签不匹配：view1={lv1}, view2={lv2}"
            # print("视角一致")

        elif mode == 'validation':
            self.data_transform = Compose([
                Resize(self.short_side_size, interpolation='bilinear'),
                CenterCrop(size=(self.crop_size, self.crop_size)),
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = Compose([
                Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = Compose([
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples_view1[idx])
                        self.test_seg.append((ck, cp))

    @staticmethod
    def _build_sample_id(sample):
        sample = str(sample).replace("\\", "/").strip()
        marker = "multiview_action_videos/"
        if marker in sample:
            sample = sample.split(marker, 1)[1]
        sample = sample.lstrip("./")
        sample_id = os.path.splitext(sample)[0]
        if sample_id:
            return sample_id
        return os.path.splitext(os.path.basename(sample))[0]

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args

            sample_view1 = self.dataset_samples_view1[index]
            sample_view2 = self.dataset_samples_view2[index]
            buffer_view1 = self.loadvideo_decord(sample_view1, chunk_nb=-1)
            buffer_view2 = self.loadvideo_decord(sample_view2, chunk_nb=-1)
            # sample = self.dataset_samples[index]
            # buffer = self.loadvideo_decord(sample, chunk_nb=-1) # T H W C
            # print(f'加载后buffer_view1 shape（THWC）: {buffer_view1.shape}')
            # buffer_view1 shape（THWC）: (16, 480, 640, 3)

            if len(buffer_view1) == 0 or len(buffer_view2) == 0:
                while len(buffer_view1) == 0 or len(buffer_view2) == 0:
                    warnings.warn(
                        "video {} or {} not correctly loaded during training".format(sample_view1, sample_view2))
                    index = np.random.randint(self.__len__())
                    sample_view1 = self.dataset_samples_view1[index]
                    sample_view2 = self.dataset_samples_view2[index]
                    buffer_view1 = self.loadvideo_decord(sample_view1, chunk_nb=-1)
                    buffer_view2 = self.loadvideo_decord(sample_view2, chunk_nb=-1)

            # print("index:", index)
            if args.num_sample > 1:
                # print("num_sample=2")
                frame_list_view1 = []
                frame_list_view2 = []
                label_list = []
                index_list = []

                for _ in range(args.num_sample):
                    new_frames_view1 = self._aug_frame(buffer_view1, args)  # (C T H W)torch.Size([3, 16, 224, 224])
                    new_frames_view2 = self._aug_frame(buffer_view2, args)

                    label = self.label_array[index]
                    frame_list_view1.append(new_frames_view1)
                    frame_list_view2.append(new_frames_view2)
                    label_list.append(label)
                    index_list.append(index)

                return frame_list_view1, frame_list_view2, label_list, label_list, label_list, {}
            else:

                buffer_view1 = self._aug_frame(buffer_view1, args)
                buffer_view2 = self._aug_frame(buffer_view2, args)

            return buffer_view1, buffer_view2, self.label_array[index], index, index, {}


        elif self.mode == 'validation':
            # sample = self.dataset_samples[index]
            # buffer = self.loadvideo_decord(sample, chunk_nb=0)
            sample = self.dataset_samples_view1[index]
            # sample_view2 = self.dataset_samples_view2[index]
            buffer = self.loadvideo_decord(sample, chunk_nb=0)
            # buffer_view2 = self.loadvideo_decord(sample_view2, chunk_nb=0)

            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples_view1[index]
                    buffer = self.loadvideo_decord(sample, chunk_nb=0)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], self._build_sample_id(sample)
            # if len(buffer_view1) == 0 or len(buffer_view2) == 0:
            #     while len(buffer_view1) == 0 or len(buffer_view2) == 0:
            #         warnings.warn(
            #             "video {} or {} not correctly loaded during validation".format(sample_view1, sample_view2))
            #         index = np.random.randint(self.__len__())
            #         sample_view1 = self.dataset_samples_view1[index]
            #         sample_view2 = self.dataset_samples_view2[index]
            #         buffer_view1 = self.loadvideo_decord(sample_view1, chunk_nb=0)
            #         buffer_view2 = self.loadvideo_decord(sample_view2, chunk_nb=0)
            #
            #     # 进行数据预处理
            # buffer_view1 = self.data_transform(buffer_view1)
            # buffer_view2 = self.data_transform(buffer_view2)
            # return buffer_view1, buffer_view2, self.label_array[index], sample_view1.split("/")[-1].split(".")[0]


        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample, chunk_nb=chunk_nb)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format( \
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample, chunk_nb=chunk_nb)

            # 注释掉这部分，因为扰动将在final_test函数中应用
            # 应用扰动（如果指定）
            # if self.perturbation_type is not None:
            #     buffer = apply_perturbation(buffer, self.perturbation_type, self.perturbation_severity)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)
            if self.test_num_crop == 1:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) / 2
                spatial_start = int(spatial_step)
            else:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                               / (self.test_num_crop - 1)
                spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[:, spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[:, :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)
            # print(f'buffer shape: {buffer.shape}')
            return buffer, self.test_label_array[index], self._build_sample_id(sample), chunk_nb, split_nb
            # return buffer, self.test_label_array[index], sample.split("ixmas_avi/")[-1].split(".")[0], \
            #     chunk_nb, split_nb
            # return buffer, self.test_label_array[index], sample.split("Interaction/")[-1].split(".")[0], \
            #         chunk_nb, split_nb
            # return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
            #     chunk_nb, split_nb

        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
            self,
            buffer,
            args,
    ):

        aug_transform = create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        # 去掉随机变换
        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C

        # 移除归一化
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def _get_seq_frames(self, video_size, num_frames, clip_idx=-1):
        seg_size = max(0., float(video_size - 1) / num_frames)
        max_frame = int(video_size) - 1
        seq = []
        # index from 1, must add 1
        if clip_idx == -1:
            for i in range(num_frames):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                idx = min(random.randint(start, end), max_frame)
                seq.append(idx)
        else:
            num_segment = 1
            if self.mode == 'test':
                num_segment = self.test_num_segment
            duration = seg_size / (num_segment + 1)
            for i in range(num_frames):
                start = int(np.round(seg_size * i))
                frame_index = start + int(duration * (clip_idx + 1))
                idx = min(frame_index, max_frame)
                seq.append(idx)
        return seq

    def loadvideo_decord(self, sample, chunk_nb=0):
        """Load video content using Decord"""
        fname = sample
        fname = os.path.join(self.prefix, fname)

        try:
            if self.keep_aspect_ratio:
                if "s3://" in fname:
                    video_bytes = self.client.get(fname)
                    vr = VideoReader(io.BytesIO(video_bytes),
                                     num_threads=1,
                                     ctx=cpu(0))
                else:
                    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                if "s3://" in fname:
                    video_bytes = self.client.get(fname)
                    vr = VideoReader(io.BytesIO(video_bytes),
                                     width=self.new_width,
                                     height=self.new_height,
                                     num_threads=1,
                                     ctx=cpu(0))
                else:
                    vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                     num_threads=1, ctx=cpu(0))

            all_index = self._get_seq_frames(len(vr), self.clip_len, clip_idx=chunk_nb)
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        #     # 打印视频信息
        #     print(f"Loading video: {fname}")
        #     print(f"Total frames: {len(vr)}")
        #     frame = vr[0].asnumpy()  # 读取第一帧以获取分辨率
        #     height, width = frame.shape[:2]
        #     print(f"Resolution: {width}x{height}")
        #
        #     all_index = self._get_seq_frames(len(vr), self.clip_len, clip_idx=chunk_nb)
        #     print(f"Selected frame indices: {all_index}")
        #     vr.seek(0)
        #     buffer = vr.get_batch(all_index).asnumpy()
        #     print(f"Loaded buffer shape: {buffer.shape}")
        #     return buffer
        # except Exception as e:
        #     print(f"Error loading video {fname}: {str(e)}")
        #     print(f"Exception type: {type(e).__name__}")
        #     import traceback
        #     traceback.print_exc()  # 打印完整的堆栈跟踪
        #     return []

    def __len__(self):
        if self.mode != 'test':
            print(f"Dataset_view1 length (mode: {self.mode}): {len(self.dataset_samples_view1)}")
            # return len(self.dataset_samples_view1 + self.dataset_samples_view2)
            return len(self.dataset_samples_view1)
        else:
            print(f"view1 Dataset length (mode: {self.mode}): {len(self.test_dataset)}")
            return len(self.test_dataset)


def spatial_sampling(
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        random_horizontal_flip=True,
        inverse_uniform_sampling=False,
        aspect_ratio=None,
        scale=None,
        motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = random_crop(frames, crop_size)
        else:
            transform_func = (
                random_resized_crop_with_shift
                if motion_shift
                else random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor
