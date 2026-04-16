import os
import numpy as np
import sys
from typing import Iterable, Optional
import torch
import torch.distributed as dist
from datasets.mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
import torch.nn.functional as F
from engines.losses import entropy_loss, compute_gradient_penalty
from engines.stillmix import StillMixRandomBlending
import random
import imageio
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from datetime import datetime
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图工具
import cv2
import glob
from fvcore.nn import FlopCountAnalysis

import torch.nn as nn

class EDL_Loss(nn.Module):
    """
    evidence deep learning loss
    """
    def __init__(self, num_classes):
        super(EDL_Loss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, labels):
        alpha = torch.exp(logits)+10./self.num_classes
        total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]

        one_hot_y = torch.eye(logits.shape[1]).cuda()
        one_hot_y = one_hot_y[labels]
        one_hot_y.requires_grad = False
        loss_nll = torch.sum(one_hot_y * (total_alpha.log() - alpha.log()), dim=1) # / logits.shape[0]

        uniform_bata = torch.ones((1, logits.shape[1])).cuda()
        uniform_bata.requires_grad = False
        total_uniform_beta = torch.sum(uniform_bata, dim=1)
        new_alpha = one_hot_y + (1.0 - one_hot_y) * (self.num_classes / 10.) * alpha
        new_total_alpha = torch.sum(new_alpha, dim=1)  # new_total_alpha.shape: [B]
        loss_kl = torch.lgamma(new_total_alpha) - torch.lgamma(total_uniform_beta) - torch.sum(torch.lgamma(new_alpha), dim=1) \
            + torch.sum((new_alpha - 1) * (torch.digamma(new_alpha) - torch.digamma(new_total_alpha.unsqueeze(1))), dim=1)
        loss_kl = 0.5*loss_kl / self.num_classes

        return loss_nll, loss_kl

def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    # **转换为 float32，防止 lgamma 报错**
    S_alpha = S_alpha.to(torch.float32)
    alpha = alpha.to(torch.float32)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=-1, keepdim=True) + 1e-10
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    S = S.to(torch.float32)
    alpha = alpha.to(torch.float32)

    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A + B))


def train_class_batch(model, samples_view1, samples_view2, target, criterion, cur_epoch):
    # 打印 samples_view1 和 samples_view2 的形状，确保输入数据的形状正确
    # print(f'samples_view1 shape: {samples_view1.shape}')

    # outputs, loss_hsic = model(samples_view1, samples_view2, hsic=True)
    # print("hsic_loss:", loss_hsic.item())
    # loss = criterion(outputs, target) + loss_hsic * 0.5
    # print(f"Current epoch: {cur_epoch}")

    outputs, alpha1, alpha2 = model(samples_view1, samples_view2)
    loss = criterion(outputs, target)
    #
    lossnds1 = ce_loss(target, alpha1 + 1, 10, cur_epoch, 10) + \
               ce_loss(target, alpha2 + 1, 10, cur_epoch, 10)
    # c是类别
    # print(f"crossloss: {loss}")
    # print(f"5*lossnds1: {5 * lossnds1}")
    total_loss = 5* lossnds1 + loss


    # edl_criterion = EDL_Loss(8)  # edl_criterion = EDL_Loss(num_classes)类别
    # x1, x11 = edl_criterion(alpha1, target)
    # x2, x22 = edl_criterion(alpha2, target)
    # loss1 = (x1 + x11).mean() + (x2 + x22).mean()
    # print(f"0.5edlloss value: {0.5*loss1}")
    # total_loss = 0.5*loss1 + loss
    # print(f"edlloss value: {loss1}")
    # total_loss = loss1 + loss

    return total_loss, outputs


# 定义一个函数生成随机边界框
def rand_bbox(frame_size, thumbnail_size):
    """
    生成随机的边界框坐标，用于嵌入缩略图
    Args:
        frame_size (int): 视频帧的高度或宽度（假设为正方形）
        thumbnail_size (int): 缩略图的大小
    Returns:
        tuple: (x1, y1, x2, y2) 边界框坐标
    """
    x1 = random.randint(0, frame_size - thumbnail_size)
    y1 = random.randint(0, frame_size - thumbnail_size)
    x2 = x1 + thumbnail_size
    y2 = y1 + thumbnail_size
    return x1, y1, x2, y2


def save_video_frames(video_tensor, output_path, fps=16):
    frames = []
    num_frames = video_tensor.size(1)  # T，即帧数
    for t in range(num_frames):
        frame = video_tensor[:, t, :, :].permute(1, 2, 0).cpu().numpy()  # 转为 (H, W, C)
        frame = (frame * 255).astype('uint8')  # 转换为 0-255 的 uint8 格式
        frames.append(frame)
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264')


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    try:
        return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale
    except Exception:
        return 0


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, no_amp=False, bf16=False):
    model.train(True)
    
    # 手动计算GFLOPs (只在第一个epoch执行一次)
    if epoch == -1:
        for samples_view1, samples_view2, _, _, _, _ in data_loader:
            try:
                # 获取输入样本形状
                B, C, T, H, W = samples_view1.shape
                
                # 获取模型信息
                embed_dim = model.embed_dim if hasattr(model, "embed_dim") else 384
                depth = len(model.layers) if hasattr(model, "layers") else 24
                num_classes = model.num_classes if hasattr(model, "num_classes") else 10  # 获取实际类别数，默认为10
                
                # 计算参数量
                params = sum(p.numel() for p in model.parameters()) / 1e6
                
                # 手动计算GFLOPs (基于Mamba论文和VideoMamba结构)
                # 1. Patch Embedding
                patch_size = 16  # 默认patch大小
                flops_patch_embed = B * C * (H//patch_size) * (W//patch_size) * T * embed_dim * (patch_size*patch_size) * 2
                
                # 2. Mamba Blocks
                seq_len = (H//patch_size) * (W//patch_size) * T
                flops_per_token = 8 * embed_dim * embed_dim  # 估计值
                flops_blocks = B * seq_len * flops_per_token * depth
                
                # 3. 分类头
                flops_head = B * embed_dim * num_classes  # 使用正确的类别数
                
                # 总计FLOPs
                total_flops = flops_patch_embed + flops_blocks + flops_head
                gflops = total_flops / 1e9
                
                # 打印信息
                print("\n" + "="*50)
                print(f"模型信息:")
                print(f"参数量: {params:.2f}M")
                print(f"输入形状: {samples_view1.shape}")
                print(f"嵌入维度: {embed_dim}")
                print(f"模型深度: {depth}")
                print(f"估计GFLOPs: {gflops:.2f}G")
                print("="*50 + "\n")
                
            except Exception as e:
                print(f"计算信息时出错: {e}")
                print("\n" + "="*50)
                print(f"模型信息:")
                print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
                if hasattr(model, "embed_dim"):
                    if model.embed_dim <= 192:
                        print(f"估计GFLOPs: ~5-8G (基于模型大小)")
                    elif model.embed_dim <= 384:
                        print(f"估计GFLOPs: ~18-25G (基于模型大小)")
                    elif model.embed_dim <= 576:
                        print(f"估计GFLOPs: ~40-50G (基于模型大小)")
                    else:
                        print(f"估计GFLOPs: ~80-100G (基于模型大小)")
                print("="*50 + "\n")
            
            # 只计算一次就跳出循环
            break
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    # 新增：定义缩略图大小和应用概率
    # thumbnail_size = 112  # 可根据需要调整缩略图大小
    participation_rate = 0  # 100% 的概率应用缩略图嵌入
    # print("participation_rate:",participation_rate)
    # 新增：定义缩略图参数
    thumbnail_scale = 1 / 2  # 缩略图为原图1/9面积，边长为1/3

    past_frames_buffer = []  # 存储过去3帧的buffer

    for data_iter_step, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        samples_view1 = batch[0]
        samples_view2 = batch[1]
        targets = batch[2]
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration

        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    if "lr_scale" in param_group:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    else:
                        param_group["lr"] = lr_schedule_values[it]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples_view1 = samples_view1.to(device, non_blocking=True)
        samples_view2 = samples_view2.to(device, non_blocking=True)  # (B , C , T , H , W)
        if epoch == 0:
            print("devive:",device)
        # still_view1 = still_view1.to(device, non_blocking=True)
        # still_view2 = still_view2.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # print(f'samples_view1 shape: {samples_view1.shape}')
        # print(f'targets shape: {targets.shape}') (B , C , T , H , W)
        # 缩略图嵌入逻辑
        if random.random() < participation_rate:
            print("应用缩略图图，概率：",participation_rate)
            batch_size, channels, num_frames, height, width = samples_view1.size()
            thumbnail_size = int(height * thumbnail_scale)  # 缩略图边长为原图的1/3
            total_thumbnail_height = thumbnail_size * 3  # 三个缩略图纵向排列的总高度

            thumbnail_positions = []  # 为每个视频生成三个固定的随机位置
            # thumbnail_positions_2 = []  # 为每个视频生成三个固定的随机位置
            for b in range(batch_size):
                positions = []
                for _ in range(3):  # 为三个缩略图生成位置
                    x1 = random.randint(0, width - thumbnail_size)
                    y1 = random.randint(0, height - thumbnail_size)
                    positions.append((x1, y1))
                thumbnail_positions.append(positions)

            # 处理每个batch
            for b in range(batch_size):
                rand_x = random.randint(0, width - thumbnail_size)  # 随机宽度方向位置
                # print("视频内固定位置：",rand_x)

                for t in range(num_frames):
                    # 创建当前帧的canvas
                    canvas = samples_view1[b, :, t, :, :].clone()
                    canvas_2 = samples_view2[b, :, t, :, :].clone()
                    # 只有当有足够的过去帧时才嵌入缩略图
                    if t >= 2:  # 需要至少3帧历史（t-1, t-2, t-3）
                        thumbnails = []
                        # 获取过去3帧
                        for past_offset in range(1, 4):
                            past_frame_idx = t - past_offset
                            if past_frame_idx >= 0:  # 确保索引有效
                                thumbnail = F.interpolate(
                                    samples_view2[b, :, past_frame_idx, :, :].unsqueeze(0),
                                    size=(thumbnail_size, thumbnail_size),
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze(0)
                                thumbnails.append(thumbnail)

                        thumbnails_2 = []
                        for past_offset_2 in range(1, 4):
                            past_frame_idx_2 = t - past_offset_2
                            if past_frame_idx_2 >= 0:  # 确保索引有效
                                thumbnail = F.interpolate(
                                    samples_view1[b, :, past_frame_idx_2, :, :].unsqueeze(0),
                                    size=(thumbnail_size, thumbnail_size),
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze(0)
                                thumbnails_2.append(thumbnail)

                        # 方案一
                        # 缩略图并列，位置随机（视频内/视频外
                        # if thumbnails:
                        #     stacked_thumbnails = torch.cat(thumbnails, dim=-2)  # 纵向拼接
                        #     # 确保纵向空间足够
                        #     # rand_x = random.randint(0, width - thumbnail_size)  # 随机宽度方向位置
                        #     # 使用随机的rand_x
                        #     if stacked_thumbnails.shape[-2] <= height:
                        #         canvas[:, :stacked_thumbnails.shape[-2],
                        #         rand_x:rand_x + thumbnail_size] = stacked_thumbnails

                        # 方案二
                        # 三个都随机，视频内固定
                        if thumbnails:
                            for i, thumbnail in enumerate(thumbnails):
                                x1, y1 = thumbnail_positions[b][i]  # 获取该缩略图的固定位置
                                x2 = x1 + thumbnail_size
                                y2 = y1 + thumbnail_size
                                if x2 <= width and y2 <= height:  # 确保不超出边界
                                    canvas[:, y1:y2, x1:x2] = thumbnail
                        if thumbnails_2:
                            for i, thumbnail in enumerate(thumbnails_2):
                                x1, y1 = thumbnail_positions[b][i]  # 获取该缩略图的固定位置
                                x2 = x1 + thumbnail_size
                                y2 = y1 + thumbnail_size
                                if x2 <= width and y2 <= height:  # 确保不超出边界
                                    canvas_2[:, y1:y2, x1:x2] = thumbnail

                    # 更新当前帧
                    samples_view1[b, :, t, :, :] = canvas
                    samples_view2[b, :, t, :, :] = canvas_2

        # # 缩略图嵌入逻辑
        # if random.random() < participation_rate:  # 随机决定是否应用缩略图增强
        #     batch_size, _, num_frames, height, width = samples_view1.size()
        #
        #     # 随机选择视角2的一个帧索引
        #     # t = random.randint(0, num_frames - 1)
        #     # 固定为第8帧
        #     t = 8
        #

        #     # 计算右下角的插入位置
        #     # x1 = width - thumbnail_size  # 224 - 112 = 112
        #     # y1 = height - thumbnail_size  # 224 - 112 = 112
        #     # x2 = width  # 224
        #     # y2 = height  # 224
        #
        #     # 对每个批次样本进行处理
        #     for b in range(batch_size):
        #         # 提取视角2的第 t 帧并缩放为缩略图
        #         thumbnail = F.interpolate(samples_view2[b, :, t, :, :].unsqueeze(0),
        #                                   size=(thumbnail_size, thumbnail_size),
        #                                   mode='bilinear',
        #                                   align_corners=False).squeeze(0)
        #         # 生成随机边界框（对所有帧使用相同位置）
        #         # bbx1, bby1, bbx2, bby2 = rand_bbox(height, thumbnail_size)
        #         # 将缩略图嵌入到视角1的每一帧的相同位置
        #         for frame in range(num_frames):
        #             samples_view1[b, :, frame, bbx1:bbx2, bby1:bby2] = thumbnail
        #             # samples_view1[b, :, frame, y1:y2, x1:x2] = thumbnail

        # 在训练循环中保存所有视频
        # bsize = samples_view1.size(0)  # 获取批量大小
        # for b in range(bsize):
        #     output_path = f"/HOME/scw6fgn/run/VideoMamba-main/videomamba/video_sm/exp/k400/videomamba_small/video25/epoch_{epoch}_batch_{data_iter_step}_sample_{b}.mp4"  # 为每个样本生成唯一的文件名
        #     save_video_frames(samples_view1[b], output_path)
        #     output_path2 = f"/HOME/scw6fgn/run/VideoMamba-main/videomamba/video_sm/exp/k400/videomamba_small/video25_2/epoch_{epoch}_batch_{data_iter_step}_sample_{b}.mp4"  # 为每个样本生成唯一的文件名
        #     save_video_frames(samples_view2[b], output_path2)

        if mixup_fn is not None:
            samples_view1, targets = mixup_fn(samples_view1, targets)
            samples_view2, _ = mixup_fn(samples_view2, targets)

        if loss_scaler is None:
            if not no_amp:
                print("not no_amp")
                samples_view1 = samples_view1.bfloat16() if bf16 else samples_view1.half()
                samples_view2 = samples_view2.bfloat16() if bf16 else samples_view2.half()
            print("loss_scaler is None")
            # 方法二：
            # loss, output = train_class_batch(model, samples_view1, samples_view2, targets, criterion, epoch)

            # loss_total1, output = train_class_batch(model, mix_view1, mix_view2, targets, criterion, epoch)
            # baseline：
            output = model(samples_view1, samples_view2)
            loss = criterion(output, targets)
            print(f"?crossloss value: {loss}")

        else:
            # print("loss_scaler is ")
            # 这条线
            with amp_autocast:
                # 方法二三：
                loss, output = train_class_batch(model, samples_view1, samples_view2, targets, criterion, epoch)
                # print(f"edl total loss: {loss}")
                # print(f"ce total loss: {loss}")

                # baseline：
                # output = model(samples_view1, None)
                # output = model(samples_view1, samples_view2)
                # loss = criterion(output, targets)
                # print(f"crossloss value: {loss}")


        loss_value = loss.item()

        # 分布式训练环境中检测损失值中的 NaN（非数字）和 Inf（无穷大）
        loss_list = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, loss)
        loss_list = torch.tensor(loss_list)
        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()

        if loss_list_isnan or loss_list_isinf:
            print(" ========== loss_isnan = {},  loss_isinf = {} ========== ".format(loss_list_isnan, loss_list_isinf))
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            if loss_scaler != 'none':
                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss /= update_freq
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.parameters(), create_graph=is_second_order,
                                        update_grad=(data_iter_step + 1) % update_freq == 0)
                if (data_iter_step + 1) % update_freq == 0:
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
                loss_scale_value = loss_scaler.state_dict()["scale"]
            else:
                loss /= update_freq
                loss.backward()
                if (data_iter_step + 1) % update_freq == 0:
                    if max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
                loss_scale_value = 0

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, amp_autocast, ds=True, no_amp=False, bf16=False, maxk=5):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'


    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        # videos = batch[0]
        # target = batch[1]
        # videos = videos.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        # 获取双视角数据
        videos_view1 = batch[0]  # 第一个视角的视频
        # videos_view2 = batch[1]  # 第二个视角的视频
        target = batch[1]  # 真实标签
        videos_view1 = videos_view1.to(device, non_blocking=True)
        # videos_view2 = videos_view2.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # if ds:
        #     if not no_amp:
        #         videos = videos.bfloat16() if bf16 else videos.half()
        #     output = model(videos)
        #     loss = criterion(output, target)
        # else:
        #     with amp_autocast:
        #         output = model(videos)
        #         loss = criterion(output, target)
        # 计算输出
        if ds:
            if not no_amp:
                videos_view1 = videos_view1.bfloat16() if bf16 else videos_view1.half()
                # videos_view2 = videos_view2.bfloat16() if bf16 else videos_view2.half()
            # 将双视角数据输入模型
            output = model(videos_view1)
            loss = criterion(output, target)
        else:
            with amp_autocast:
                output = model(videos_view1)
                loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, maxk))

        batch_size = videos_view1.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# 模块级别（保持不变）
features = None
gradients = None

@torch.no_grad()
def final_test(data_loader, model, device, file, amp_autocast, ds=True, no_amp=False, bf16=False, maxk=5):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    # 收集预测、真实标签和特征
    all_preds = []
    all_labels = []
    all_features = []
    # 初始化全局变量

    # # 注册钩子以获取中间层特征和梯度
    # def get_features_hook(module, input, output):
    #     global features
    #     features = output
    #
    # def get_gradients_hook(module, grad_in, grad_out):
    #     global gradients
    #     gradients = grad_out[0]
    # 注册钩子以获取中间层特征和梯度
    def get_features_hook(module, input, output):
        global features
        features = output
        print(f"Forward hook triggered! Features shape: {output.shape}")

    def get_gradients_hook(module, grad_in, grad_out):
        global gradients
        gradients = grad_out[0] if grad_out and grad_out[0] is not None else None
        print(f"Backward hook triggered! Gradients shape: {gradients.shape if gradients is not None else 'None'}")



    # 目标层：patch_embed
    # 如果 model 是 DDP 包装的，需要通过 model.module 访问原始模型
    base_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    target_layer = base_model.patch_embed.proj  # 选择 patch_embed 的卷积层
    # target_layer = base_model.layers[11].mixer  # 选择最后一个 Mamba 块的输出投影层
    # target_layer = base_model.layers[-1].norm  #
    # print(f"norm_f exists: {hasattr(base_model, 'norm_f')}")
    # if hasattr(base_model, 'norm_f'):
    #     target_layer = base_model.norm_f
    #     print("Using target_layer: norm_f")
    # elif hasattr(base_model, 'layers') and len(base_model.layers) > 0:
    #     target_layer = base_model.layers[-1].norm
    #     print("Using target_layer: layers[-1].norm")
    # else:
    #     target_layer = base_model.patch_embed.proj
    #     print("Using target_layer: patch_embed.proj")

    target_layer.register_forward_hook(get_features_hook)
    target_layer.register_full_backward_hook(get_gradients_hook)



    # for batch in metric_logger.log_every(data_loader, 10, header):
    # 使用 enumerate 获取 batch_idx
    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        videos_view1 = batch[0]

        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos_view1 = videos_view1.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)


        # compute output
        if ds:
            if not no_amp:
                videos_view1 = videos_view1.bfloat16() if bf16 else videos_view1.half()

            output = model(videos_view1)
            loss = criterion(output, target)
        else:
            with amp_autocast:
                output = model(videos_view1)
                loss = criterion(output, target)
        # 收集预测、真实标签和特征
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        all_features.extend(output.float().cpu().numpy())  # 提取模型输出作为特征

        # --- Grad-CAM 热力图生成 ---
        # --- Grad-CAM 热力图生成（只对前 5 个 batch 生成，减少开销） ---
        if batch_idx < 0:
            for i in range(output.size(0)):
                # 选择目标类别（预测类别）
                target_class = preds[i].item()
                # 临时启用梯度计算
                with torch.enable_grad():
                    model.zero_grad(set_to_none=True)
                    # 重新计算 output，确保可以跟踪梯度
                    videos_view1_i = videos_view1[i:i + 1]  # 取第 i 个样本，保持 batch 维度
                    if ds:
                        if not no_amp:
                            videos_view1_i = videos_view1_i.bfloat16() if bf16 else videos_view1_i.half()
                        output_i = model(videos_view1_i)
                    else:
                        with amp_autocast:
                            output_i = model(videos_view1_i)
                    output_i = output_i[0, target_class]  # 目标类别的分数

                    # 清空梯度
                    model.zero_grad()
                    output_i.backward()

                # # 获取特征和梯度
                # global features, gradients
                # # 方案一热力图
                # 特征形状：(batch_size, channels, frames, height/16, width/16)
                feature = features[0]  # (channels, frames, height/16, width/16)
                gradient = gradients[0]  # (channels, frames, height/16, width/16)

                # 获取特征和梯度
                if features is None or gradients is None:
                    print(f"Warning: Features or gradients not captured for batch {batch_idx}, sample {i}")
                    continue

                try:
                    feature = features[0]  # [channels, frames, height/16, width/16]
                    gradient = gradients[0]  # [channels, frames, height/16, width/16]
                    if feature.shape != gradient.shape:
                        print(f"Shape mismatch: features {feature.shape}, gradients {gradient.shape}")
                        continue
                except (TypeError, IndexError) as e:
                    print(f"Error accessing features or gradients: {e}, batch {batch_idx}, sample {i}")
                    continue

                # 计算梯度的全局平均池化，得到每个通道的权重
                weights = torch.mean(gradient, dim=(1, 2, 3))  # (channels,)
                heatmap = torch.zeros_like(feature[0])  # (frames, height/16, width/16)

                # # 方案二
                # # 重塑特征和梯度
                # T, H, W = videos_view1.shape[2], videos_view1.shape[3], videos_view1.shape[4]
                # H_patch, W_patch = H // 16, W // 16
                # # features: (B, T*H/16*W/16 + 1, 384)
                # feature = features[0, 1:]  # 跳过 cls_token，(T*H/16*W/16, 384)
                # gradient = gradients[0, 1:]  # (T*H/16*W/16, 384)
                # feature = feature.view(T, H_patch, W_patch, 384).permute(3, 0, 1, 2)  # (384, T, H/16, W/16)
                # gradient = gradient.view(T, H_patch, W_patch, 384).permute(3, 0, 1, 2)  # (384, T, H/16, W/16)
                # # 计算热力图
                # weights = torch.mean(gradient, dim=(1, 2, 3))  # (384,)
                # heatmap = torch.zeros(T, H_patch, W_patch, device=feature.device)


                # 计算热力图
                for c in range(feature.shape[0]):
                    heatmap += weights[c] * feature[c]
                heatmap = torch.relu(heatmap)  # 取正值
                heatmap = heatmap / (torch.max(heatmap) + 1e-6)  # 归一化到 [0, 1]

                # 将热力图调整到视频帧大小
                heatmap = heatmap.float().cpu().numpy()  # (frames, height/16, width/16)
                video_frames = videos_view1[i].float().cpu().numpy()  # (channels, frames, height, width)
                video_frames = video_frames.transpose(1, 2, 3, 0)  # (frames, height, width, channels)

                # 保存热力图（每帧一张图片）
                save_dir = os.path.dirname(file)
                heatmap_dir = os.path.join(save_dir, f'heatmap_{ids[i]}_{batch_idx}_{i}')
                os.makedirs(heatmap_dir, exist_ok=True)

                for t in range(video_frames.shape[0]):
                    frame = video_frames[t]  # (height, width, channels)
                    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)  # 归一化到 [0, 1]
                    frame = (frame * 255).astype(np.uint8)  # 转换为 0-255

                    # 调整热力图大小
                    heatmap_t = heatmap[t]  # (height/16, width/16)
                    heatmap_t = cv2.resize(heatmap_t, (frame.shape[1], frame.shape[0]))  # 调整到原始帧大小
                    heatmap_t = np.uint8(255 * heatmap_t)  # 转换为 0-255
                    heatmap_t = cv2.applyColorMap(heatmap_t, cv2.COLORMAP_JET)  # 应用颜色映射

                    # 叠加热力图到原始帧
                    superimposed_img = heatmap_t * 0.4 + frame * 0.6
                    superimposed_img = np.uint8(superimposed_img)

                    # 保存热力图
                    cv2.imwrite(os.path.join(heatmap_dir, f'frame_{t}.png'), superimposed_img)
                # --- 将热力图帧拼接成视频 ---
                # 获取所有热力图帧
                frame_files = sorted(glob.glob(os.path.join(heatmap_dir, 'frame_*.png')),
                                     key=lambda x: int(x.split('_')[-1].split('.png')[0]))
                if not frame_files:
                    print(f"No frames found in {heatmap_dir}, skipping video creation.")
                    continue

                # 读取第一帧以获取视频分辨率
                first_frame = cv2.imread(frame_files[0])
                height, width, _ = first_frame.shape

                # 创建视频写入器
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                video_path = os.path.join(heatmap_dir, f'heatmap_video_{timestamp}.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
                fps = 16  # 帧率，可以根据需要调整
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                # 将所有帧写入视频
                for frame_file in frame_files:
                    frame = cv2.imread(frame_file)
                    video_writer.write(frame)

                # 释放视频写入器
                video_writer.release()
                # print(f"Heatmap video saved to {video_path}")

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                               str(output.data[i].float().cpu().numpy().tolist()), \
                                               str(int(target[i].cpu().numpy())), \
                                               str(int(chunk_nb[i].cpu().numpy())), \
                                               str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, maxk))
        print("acc1:", acc1)

        batch_size = videos_view1.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # --- 可视化部分 ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.dirname(file)

    # 1. 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=cm_percent, fmt='.1f', cmap='Blues',
                xticklabels=sorted(set(all_labels)), yticklabels=sorted(set(all_labels)))
    plt.title(
        f'Confusion Matrix - Test Phase\nAcc@1: {metric_logger.acc1.global_avg:.2f}%, Acc@5: {metric_logger.acc5.global_avg:.2f}%')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_test_{timestamp}.png'))
    plt.close()

    # # 2. 2D t-SNE 可视化
    # features = np.array(all_features)
    # tsne_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    # tsne_results_2d = tsne_2d.fit_transform(features)
    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(tsne_results_2d[:, 0], tsne_results_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.6)
    # plt.colorbar(scatter, label='Class')
    # plt.title('2D t-SNE Visualization - Test Phase')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.savefig(os.path.join(save_dir, f'tsne_2d_test_{timestamp}.png'))
    # plt.close()
    #
    # # 3. 3D t-SNE 可视化
    # tsne_3d = TSNE(n_components=3, random_state=42, perplexity=min(30, len(features) - 1))
    # tsne_results_3d = tsne_3d.fit_transform(features)
    # fig = plt.figure(figsize=(12, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(tsne_results_3d[:, 0], tsne_results_3d[:, 1], tsne_results_3d[:, 2],
    #                      c=all_labels, cmap='tab10', s=50, alpha=0.6)
    # plt.colorbar(scatter, label='Class')
    # ax.set_title('3D t-SNE Visualization - Test Phase')
    # ax.set_xlabel('t-SNE Dimension 1')
    # ax.set_ylabel('t-SNE Dimension 2')
    # ax.set_zlabel('t-SNE Dimension 3')
    # plt.savefig(os.path.join(save_dir, f'tsne_3d_test_{timestamp}.png'))
    # plt.close()

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float32, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                print("重复？")
                continue

            # dict_feats[name] = []
            # dict_label[name] = 0
            # dict_pos[name] = []
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)

    # 计算混淆矩阵
    cm = confusion_matrix(label, pred)
    print("Confusion Matrix:")
    print(cm)
    # --- 可视化部分 ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(max(label) + 1), yticklabels=range(max(label) + 1))
    plt.title('Confusion Matrix - Merge Results')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # 修改保存路径为 eval_path 目录下
    save_path = os.path.join(eval_path, f'confusion_matrix_{timestamp}.png')
    plt.savefig(save_path)
    plt.close()  # 关闭图表，避免显示

    # # 2. t-SNE 可视化
    # # 准备特征数据（每个视频的平均特征）
    # features = np.array([np.mean(feat_list, axis=0) for feat_list in dict_feats.values()])
    # labels = np.array([int(dict_label[name]) for name in dict_feats.keys()])
    #
    # # 运行 t-SNE
    # tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    # tsne_results = tsne.fit_transform(features)
    #
    # # 绘制 t-SNE 散点图
    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.6)
    # plt.colorbar(scatter, label='Class')
    # plt.title('t-SNE Visualization of Video Features')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.savefig(os.path.join(eval_path, f'tsne_visualization_{timestamp}.png'))
    # plt.close()

    # # 运行 3D t-SNE
    # tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(features) - 1))
    # tsne_results = tsne.fit_transform(features)
    #
    # # 绘制 3D 散点图
    # fig = plt.figure(figsize=(12, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2],
    #                      c=labels, cmap='tab10', s=50, alpha=0.6)
    # plt.colorbar(scatter, label='Class')
    # ax.set_title('3D t-SNE Visualization of Video Features')
    # ax.set_xlabel('t-SNE Dimension 1')
    # ax.set_ylabel('t-SNE Dimension 2')
    # ax.set_zlabel('t-SNE Dimension 3')
    # plt.savefig(os.path.join(eval_path, f'tsne_3d_{timestamp}.png'))
    # plt.close()

    return final_top1 * 100, final_top5 * 100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
