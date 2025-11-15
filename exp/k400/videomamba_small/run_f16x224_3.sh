#!/bin/bash
module load cuda/11.8
module load gcc/12.2

export PATH=/HOME/scw6fgn/run/anaconda3/bin:$PATH
source activate  /HOME/scw6fgn/run/anaconda3/envs/Videomamba
export PYTHONUNBUFFERED=1

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
#date
#tar -xf multiview_action_videos.tar -C /HOME/scw6fgn/run/dataset
#date

#JOB_NAME='(3.18.2)_etloss_stillmix)'
#JOB_NAME='(3.18)_etloss_stillmix)'
JOB_NAME='(3.18.2)_etloss_stillmix)'

OUTPUT_DIR="/HOME/scw6fgn/run/VideoMamba-main/videomamba/video_sm/exp/k400/videomamba_small/output/$JOB_NAME"
LOG_DIR="/HOME/scw6fgn/run/VideoMamba-main/videomamba/video_sm/exp/k400/videomamba_small/output/logs/${JOB_NAME}"
PREFIX='/HOME/scw6fgn/run/dataset/multiview_action_videos/'
DATA_PATH='/HOME/scw6fgn/run/dataset/multiview_action_videos/'

PARTITION='video5'
GPUS=1
GPUS_PER_NODE=1
CPUS_PER_TASK=6

srun -p $PARTITION \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        python /HOME/scw6fgn/run/VideoMamba-main/videomamba/video_sm/run_class_finetuning_2.py \
        --model videomamba_small \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'Kinetics_sparse' \
        --split ',' \
        --nb_classes 10 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 2 \
        --num_sample 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 16 \
        --num_workers 4 \
        --warmup_epochs 5 \
        --tubelet_size 1 \
        --epochs 50 \
        --lr 2e-4 \
        --drop_path 0.35 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --test_num_segment 3 \
        --test_num_crop 4 \
        --dist_eval \
        --test_best \
        --bf16
