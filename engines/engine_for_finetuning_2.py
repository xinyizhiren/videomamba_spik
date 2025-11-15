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


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    # **转换为 float32，防止 lgamma 报错**
    # S_alpha = S_alpha.to(torch.float32)
    # alpha = alpha.to(torch.float32)
    # 防止 lgamma 和 digamma 计算 NaN
    # S_alpha = torch.clamp(S_alpha, min=1e-6).to(torch.float32)
    # alpha = torch.clamp(alpha, min=1e-6).to(torch.float32)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=-1, keepdim=True)
    # S = torch.clamp(S, min=1e-6)
    # 确保 S 不为负数
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    # S = S.to(torch.float32)
    # alpha = alpha.to(torch.float32)

    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    print(f"annealing_coef : {annealing_coef}")
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A + B))

def train_class_batch(model, samples_view1, samples_view2, target, criterion,cur_epoch):
    # 打印 samples_view1 和 samples_view2 的形状，确保输入数据的形状正确
    # print(f'samples_view1 shape: {samples_view1.shape}')


    # outputs, loss_hsic = model(samples_view1, samples_view2, hsic=True)
    # print("hsic_loss:", loss_hsic.item())
    # loss = criterion(outputs, target) + loss_hsic * 0.5
    print(f"Current epoch: {cur_epoch}")

    outputs,alpha1,alpha2 = model(samples_view1, samples_view2, hsic=False)
    lossnds1 = ce_loss(target, alpha1+1, 10, cur_epoch, 10) + \
               ce_loss(target, alpha2+1, 10, cur_epoch, 10)

    loss = criterion(outputs, target)
    print(f"crossloss value: {loss}")
    print(f"lossnds1 value: {lossnds1}")

    total_loss = lossnds1 + loss
    return total_loss, outputs

# def train_class_batch(model, samples_view1, samples_view2, target, criterion, cur_epoch):
#     """
#     训练一个批次，包含 ALBAR 的训练策略
#     """
#     # 确保输入需要梯度计算
#     samples_view1.requires_grad_(True)
#     samples_view2.requires_grad_(True)
#
#     # 前向传播
#     outputs, alpha1, alpha2 = model(samples_view1, samples_view2, hsic=False)
#
#     # 1. 计算基础分类损失
#     base_loss = criterion(outputs, target)
#
#     # 2. 计算静态片段分类损失（对抗损失）
#     lossnds = ce_loss(target, alpha1+1, 9, cur_epoch, 10) + \
#               ce_loss(target, alpha2+1, 9, cur_epoch, 10)
#
#     # 3. 计算熵最大化损失
#     ent_loss = entropy_loss(outputs)
#
#     # 4. 计算梯度惩罚
#     grad_penalty = compute_gradient_penalty(outputs, samples_view1)
#
#     # 组合所有损失
#     total_loss = base_loss + 0.1 * lossnds - 0.1 * ent_loss + 0.1 * grad_penalty

#     return total_loss, outputs


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

    # 初始化 StillMix
    stillmix1 = StillMixRandomBlending(prob_aug=0.5)  # Adjust the probability as needed
    stillmix2 = StillMixRandomBlending(prob_aug=0.5)


    for data_iter_step, (samples_view1, samples_view2, targets,still_view1,still_view2,_) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
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

        # print(
        #     f"[训练循环] samples_view1 类型: {type(samples_view1)}",
        #     f"samples_view2 类型: {type(samples_view2)}",
        #     f"still_view1 类型: {type(still_view1)}",
        #     f"still_view2 类型: {type(still_view2)}",
        #     sep="\n"
        # )
        # print(
        #     f"[训练循环] samples_view1 形状: {samples_view1.shape}",
        #     f"samples_view2 形状: {samples_view2.shape}",
        #     f"still_view1 形状: {still_view1.shape}",
        #     f"still_view2 形状: {still_view2.shape}",
        #     sep="\n"
        # )

        targets = targets.to(device, non_blocking=True)
        # stillmix
        # mix_view1 = samples_view1.permute(0,2,1,3,4) # (B,T,C,H,W)
        # mix_view1, order1 = stillmix1(mix_view1) # (B,T,C,H,W)
        # print(f'order1 shape: {order1.shape}')
        # print(order1)
        # mix_view1 = mix_view1.permute(0, 2, 1, 3, 4) #混合视频  (B,C,T,H,W)
        # mix_output_1 = model(mix_view1)  # 动态特征
        # still_view1 = still_view1[order1]
        # targets_view1 = targets[order1]
        # print("targets_view1:", targets_view1)


        still_view1 =still_view1.to(device, non_blocking=True)#  (B , C , T , H , W)
        still_view1.requires_grad = True
        spatial_output_1 = model(still_view1) # 静态特征
        # print(f'经过model后的still_view1 shape: {spatial_output_1.shape}') #(B,num)
        gradpen_1 = compute_gradient_penalty(spatial_output_1, still_view1) # 梯度惩罚损失
        still_view1.requires_grad = False
        spatial_entropy_1 = entropy_loss(spatial_output_1) # 熵最大化损失
        # spatial_loss_1 = criterion(spatial_output_1, targets_view1)  # 静态交叉熵，远离
        spatial_loss_1 = criterion(spatial_output_1, targets)  # 静态交叉熵，远离
        # mix_loss_1 = criterion(mix_output_1, targets_view1)  # 动态交叉熵，靠近
        print(f"梯度惩罚损失gradpen_1 value: {gradpen_1}")
        print(f"香农熵损失spatial_entropy_1 value: {spatial_entropy_1}")
        print(f"静态交叉熵损失spatial_loss_1 value: {spatial_loss_1}")
        # print(f"动态交叉熵损失mix_loss_1 value: {mix_loss_1}")


        # mix_view2 = samples_view2.permute(0,2,1,3,4)
        # mix_view2, order2 = stillmix2(mix_view2)
        # print(f'order2 shape: {order2.shape}')
        # print(order2)
        # mix_view2 = mix_view2.permute(0,2,1,3,4)
        # mix_output_2 =model(mix_view2) # 动态特征
        # still_view2 = still_view2[order2]
        # targets_view2 = targets[order2]
        # print("targets_view2", targets_view2)

        still_view2 = still_view2.to(device, non_blocking=True)#  (B , C , T , H , W)
        still_view2.requires_grad = True
        spatial_output_2 = model(still_view2)  # 静态特征
        gradpen_2 = compute_gradient_penalty(spatial_output_2, still_view2)  # 梯度惩罚损失
        still_view2.requires_grad = False
        spatial_entropy_2 = entropy_loss(spatial_output_2)  # 熵最大化损失
        # spatial_loss_2 = criterion(spatial_output_2, targets_view2)  # 静态交叉熵，远离
        spatial_loss_2 = criterion(spatial_output_2, targets)  # 静态交叉熵，远离
        # mix_loss_2 = criterion(mix_output_2, targets_view2)  # 动态交叉熵，靠近
        print(f"梯度惩罚损失gradpen_2 value: {gradpen_2}")
        print(f"香农熵损失spatial_entropy_2 value: {spatial_entropy_2}")
        print(f"静态交叉熵损失spatial_loss_2 value: {spatial_loss_2}")
        # print(f"动态交叉熵损失mix_loss_2 value: {mix_loss_2}")



        samples_view1 = samples_view1.to(device, non_blocking=True)#  (B , C , T , H , W)
        samples_view2 = samples_view2.to(device, non_blocking=True)#  (B , C , T , H , W)


        if mixup_fn is not None:
            samples_view1, targets = mixup_fn(samples_view1, targets)
            samples_view2, _ = mixup_fn(samples_view2, targets)

        if loss_scaler is None:
            if not no_amp:
                print("not no_amp")
                samples_view1 = samples_view1.bfloat16() if bf16 else samples_view1.half()
                samples_view2 = samples_view2.bfloat16() if bf16 else samples_view2.half()
            print("loss_scaler is None")
            loss_total1, output = train_class_batch(model, samples_view1, samples_view2, targets, criterion, epoch)
            # loss_total1, output = train_class_batch(model, mix_view1, mix_view2, targets, criterion, epoch)
        else:
            # 这条线
            # print("loss_scaler is ")
            with amp_autocast:
                loss_total1, output = train_class_batch(model, samples_view1, samples_view2, targets, criterion, epoch)

        max_weight = min(1, epoch / 10.0)
        print(f"max_weight : {max_weight}")
        # loss = loss_total1 - spatial_loss_1 -spatial_loss_2 - spatial_entropy_1 - spatial_entropy_2 + gradpen_1 +gradpen_2 +mix_loss_2 +mix_loss_1
        loss = loss_total1 +max_weight*(10*gradpen_1 + 10*gradpen_2-0.5* spatial_loss_1 - 0.5*spatial_loss_2 - 0.5*spatial_entropy_1 - 0.5*spatial_entropy_2)
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


@torch.no_grad()
def final_test(data_loader, model, device, file, amp_autocast, ds=True, no_amp=False, bf16=False, maxk=5):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
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

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                               str(output.data[i].float().cpu().numpy().tolist()), \
                                               str(int(target[i].cpu().numpy())), \
                                               str(int(chunk_nb[i].cpu().numpy())), \
                                               str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, maxk))

        batch_size = videos_view1.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

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
                continue
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
    return final_top1 * 100, final_top5 * 100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
