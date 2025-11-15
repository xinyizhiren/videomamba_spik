#!/bin/bash

# 取消服务器/HPC依赖
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

JOB_NAME='videomamba_small_f16_res224_local'

OUTPUT_DIR="/home/oyys/code/VIT_4090/videomamba/video_sm/outputs/$JOB_NAME"
LOG_DIR="/home/oyys/code/VIT_4090/videomamba/video_sm/logs/$JOB_NAME"
PREFIX='/data_hdd/oyys/VIT_4090/dataset/data/multiview_action_videos/'
DATA_PATH='/data_hdd/oyys/VIT_4090/dataset/data/multiview_action_videos/'

# 直接运行Python命令，无需srun集群参数
python /home/oyys/code/VIT_4090/videomamba/video_sm/run_class_finetuning.py \
        --model videomamba_small \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'Kinetics_sparse' \
        --split ',' \
        --nb_classes 12 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 8 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 16 \
        --num_workers 6 \
        --warmup_epochs 3 \
        --tubelet_size 1 \
        --epochs 30 \
        --lr 2e-4 \
        --drop_path 0.35 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --dist_eval \
        --test_best \
        --bf16