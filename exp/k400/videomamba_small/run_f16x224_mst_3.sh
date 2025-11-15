#!/bin/bash
module load cuda/11.8
module load gcc/12.2

export PATH=/HOME/scw6fgn/run/anaconda3/bin:$PATH
source activate  /HOME/scw6fgn/run/anaconda3/envs/Videomamba
export PYTHONUNBUFFERED=1

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

#JOB_NAME='(3.31)_baseline+cut_random_view1+2)'
#JOB_NAME='(3.14)_0.5*etloss_nohsic_nostillmix)'
#JOB_NAME='(3.31)_baseline+cut_random)'
#JOB_NAME='(4.8)_baseline+1/2_view12_0.8_5)'
# 互放，1/2大小，80%应用，5帧插入随机位置
#JOB_NAME='(4.8)_baseline+1.2_view12_0.8_7)'
# 互放，1/2大小，80%应用，7帧插入随机位置
#JOB_NAME='(4.8)_baseline+1.2_view12_0.8_3)'
# 互放，1/2大小，80%应用，3帧插入随机位置
#JOB_NAME='(4.8)_baseline+1.3_2view12_0.8_5)'
# 互放两图，1/3大小，80%应用，5帧插入随机位置
#JOB_NAME='(4.8)_baseline+1.3_2view12_0.8_3)'
# 互放两图，1/3大小，80%应用，3帧插入随机位置
#JOB_NAME='(4.8)_baseline+1.3_2view12_0.8_7)'
# 互放两图，1/3大小，80%应用，7帧插入随机位置
#JOB_NAME='(4.8)_baseline+1.3_2view12_0.8_12)'
# 互放两图，1/3大小，80%应用，7帧插入随机位置


#JOB_NAME='(4.8)_baseline+1.4_2view12_0.8_3)'
# 互放两图，1/4大小，80%应用，3帧插入随机位置
#JOB_NAME='(4.8)_baseline+1.4_2view12_0.8_7)'
# 互放两图，1/4大小，80%应用，7帧插入随机位置
#JOB_NAME='(4.8)_baseline+1.4_2view12_0.8_12)'
# 互放两图，1/4大小，80%应用，7帧插入随机位置


#JOB_NAME='(4.8)_baseline+1.4_3view12_0.8_3)'
# 互放三图，1/4大小，80%应用，3帧插入随机位置
#JOB_NAME='(4.8)_baseline+1.4_3view12_0.8_7)'
# 互放三图，1/4大小，80%应用，7帧插入随机位置
#JOB_NAME='(4.8)_baseline+1.4_3view12_0.8_15)'
# 互放三图，1/4大小，80%应用，15帧插入随机位置,证据融合

#JOB_NAME='(4.8)_baseline+1.4_3view12_0.8_7_et)'
# 互放三图，1/4大小，80%应用，7帧插入随机位置,证据融合

#JOB_NAME='(4.9)_baseline+1.2_1view12_0.8_15)'
# 互放单图，1/2大小，80%应用，15帧插入随机位置,证据融合


#JOB_NAME='(4.9)_baseline+1.2_2view12_0.8_12)'
# 互放双图，1/2大小，80%应用，12帧插入随机位置,平均融合

#JOB_NAME='(4.17)_baseline+1.4_3view_0.7_12_1test_3et)'
# 互放三图，1/4大小，70%应用，12帧插入随机位置,3et，23训练1测试

#JOB_NAME='(4.25)_baseline_et3_casia_1322_1_32+1.4_3view_0.8_30)'
#JOB_NAME='(4.25)_baseline_et3_casia_2311_1_32+1.4_3view_0.8_30)'
#JOB_NAME='(4.25)_baseline_et3_casia_1233_1_32+1.4_3view_0.8_30)'
#baseline裁剪数据集,32帧,1空间裁剪，1/4大小，80%应用，30帧插入随机位置，互放三图

#JOB_NAME='(4.26)_baseline_et2_casia_1322_1_32+1.4_2view_0.8_10)'
#JOB_NAME='(4.26)_baseline_et2_casia_2311_1_32+1.4_2view_0.8_10)'
#JOB_NAME='(4.26)_baseline_et2_casia_1233_1_32+1.4_2view_0.8_10)'
#baseline裁剪数据集,32帧,1空间裁剪，1/4大小，80%应用，10帧插入随机位置，互放2图

#JOB_NAME='(4.26_2)_baseline_et2_casia_1322_1_32+1.4_2view_0.8_10)'
#JOB_NAME='(4.26_2)_baseline_et2_casia_2311_1_32+1.4_2view_0.8_10)'
#JOB_NAME='(4.26_2)_baseline_et2_casia_1233_1_32+1.4_2view_0.8_10)'
#baseline裁剪数据集,32帧,1空间裁剪，1/4大小，80%应用，10帧插入随机位置，互放2图


#JOB_NAME='(4.28)_baseline_et2_casia_1322_1_32+1.4_2view_0.8_10)'
#JOB_NAME='(4.28)_baseline_et2_casia_2311_1_32+1.4_2view_0.8_10)'
#JOB_NAME='(4.28)_baseline_et2_casia_1233_1_32+1.4_2view_0.8_10)'
#baseline裁剪数据集,32帧,1空间裁剪，1/4大小，80%应用，10帧插入随机位置，互放3图
#JOB_NAME='(4.28)_baseline_et2_casia_2311_1_32+1.4_3view_0.8_20)'
#JOB_NAME='(4.28)_baseline_et2_casia_2311_1_32+1.4_3view_0.8_10)'
#JOB_NAME='(4.28)_baseline_et2_casia_2311_1_32+1.4_3view_0.8_30)'

#JOB_NAME='(4.28)_baseline_et2_casia_2311_1_32+1.2_1view_0.5_30)'
#JOB_NAME='(4.28)_baseline_et2_casia_2311_1_32+1.3_1view_0.5_30)'
#JOB_NAME='(4.28)_baseline_et2_casia_2311_1_32+1.3_1view_0.5_5)'

# k=1（互放单图）,80%，对比，5ce2,3帧互放，大小1/2,1/3,1/4,1/5
#JOB_NAME='(5.22)_2_1233)'
#JOB_NAME='(5.22)_3_1233)'
#JOB_NAME='(5.22)_4_1233)'
#JOB_NAME='(5.22)_5_1233)'

# k=2（互访双图）,80%，对比，5ce2,3帧互放，大小1/2,1/3,1/4,1/5
#JOB_NAME='(5.22)_2_2_0.8_1233)'
#JOB_NAME='(5.22)_2_3_0.8_1233)'
#JOB_NAME='(5.22)_2_4_0.8_1233)'
#JOB_NAME='(5.22)_2_5_0.8_1233)'

# k=3（互访三图）,80%，对比，5ce2,3帧互放，大小1/2,1/3,1/4,1/5
#JOB_NAME='(5.22)_3_2_0.8_1233)'
#JOB_NAME='(5.22)_3_3_0.8_1233)'
#JOB_NAME='(5.22)_3_4_0.8_1233)'
#JOB_NAME='(5.22)_3_5_0.8_1233)'

# casia
# k=3（互访双图）,80%，对比，0.5ce2,5帧互放，大小1/2,1/3,1/4,1/5，尽量不重叠
#JOB_NAME='(5.23)_3_2_0.8_2311)'
#JOB_NAME='(5.23)_3_3_0.8_2311)'
#JOB_NAME='(5.23)_3_4_0.8_2311)'
#JOB_NAME='(5.23)_3_5_0.8_2311)'

# k=3（互访双图）,80%，对比，0.5ce2,15帧互放，大小1/2,1/3,1/4,1/5，尽量不重叠
#JOB_NAME='(5.23)_3_2_2311)'
#JOB_NAME='(5.23)_3_3_2311)'
#JOB_NAME='(5.23)_3_4_2311)'
#JOB_NAME='(5.23)_3_5_2311)'
# 2和5互换

# k=3（互访双图）,80%，对比，0.5ce2,15帧互放，大小1/2,1/3,1/4,1/5，尽量不重叠，50%叠加
#JOB_NAME='(5.24)_3_2_2311)'
#JOB_NAME='(5.24)_3_3_2311)'
#JOB_NAME='(5.24)_3_4_2311)'
#JOB_NAME='(5.24)_3_5_2311)'
# k=3（互访双图）,80%，对比，0.5ce2,10帧互放，大小1/2,1/3,1/4,1/5，尽量不重叠，50%叠加
#JOB_NAME='(5.24)_2_3_2_2311)'
#JOB_NAME='(5.24)_2_3_3_2311)'
#JOB_NAME='(5.24)_2_3_4_2311)'
#JOB_NAME='(5.24)_2_3_5_2311)'
# k=3（互访双图）,80%，对比，5ce2,3帧互放，大小1/2,1/3,1/4,1/5，尽量不重叠，50%叠加,nucla
#JOB_NAME='(5.24)_3_2_1233)'
#JOB_NAME='(5.24)_3_3_1233)'
#JOB_NAME='(5.24)_3_4_1233)'
#JOB_NAME='(5.24)_3_5_1233)'
# k=3（互访双图）,80%，对比，5ce2,6帧互放，大小1/2,1/3,1/4,1/5，尽量不重叠，50%叠加,nucla
#JOB_NAME='(5.24)_2_3_2_1233)'
#JOB_NAME='(5.24)_2_3_3_1233)'
#JOB_NAME='(5.24)_2_3_4_1233)'
#JOB_NAME='(5.24)_2_3_5_1233)'
# k=3（互访双图）,80%，对比，5ce2,6帧互放，大小1/，尽量不重叠，0，20,40,60,80,100叠加,nucla
#JOB_NAME='(5.24)_0_1233)'
#JOB_NAME='(5.24)_20_1233)'
#JOB_NAME='(5.24)_40_1233)'
#JOB_NAME='(5.24)_60_1233)'
#JOB_NAME='(5.24)_80_1233)'
#JOB_NAME='(5.24)_100_1233)'
# k=3（单方三图）,80%，对比，5ce2,6帧互放，大小1/4，尽量不重叠，0，20,40,60,80,100叠加,nucla

#JOB_NAME='(5.24)_201_1233)'
#JOB_NAME='(5.24)_401_1233)'
#JOB_NAME='(5.24)_601_1233)'
#JOB_NAME='(5.24)_801_1233)'
#JOB_NAME='(5.24)_1001_1233)'
# 10帧，1/4大小，k=3,0.5et2，casia，融合
#JOB_NAME='(5.24)_0_2311)'
#JOB_NAME='(5.24)_20_2311)'
#JOB_NAME='(5.24)_40_2311)'
#JOB_NAME='(5.24)_60_2311)'
#JOB_NAME='(5.24)_80_2311)'
#JOB_NAME='(5.24)_100_2311)'


# 空间,k=1,60%融合
#JOB_NAME='(5.24)_01_2_2311)'
#JOB_NAME='(5.24)_023_2_2311)'
#JOB_NAME='(5.24)_01_4_2311)'
#JOB_NAME='(5.24)_01_5_2311)'
#JOB_NAME='(5.25)_base_tt_2)'
JOB_NAME='(5.26)_edl_10'
#OUTPUT_DIR="/HOME/scw6fgn/run/VideoMamba-main/videomamba/video_sm/exp/k400/videomamba_small/output24nucla/$JOB_NAME"
#LOG_DIR="/HOME/scw6fgn/run/VideoMamba-main/videomamba/video_sm/exp/k400/videomamba_small/output24nucla/logs/${JOB_NAME}"
#PREFIX='/HOME/scw6fgn/run/dataset/multiview_action_videos/csv12/'
#DATA_PATH='/HOME/scw6fgn/run/dataset/multiview_action_videos/csv12/'
#PREFIX='/HOME/scw6fgn/run/dataset/SinglePerson1/csv23/'
#DATA_PATH='/HOME/scw6fgn/run/dataset/SinglePerson1/csv23/'
OUTPUT_DIR="/HOME/scw6fgn/run/VideoMamba-main/videomamba/video_sm/exp/k400/videomamba_small/outputcasia25/$JOB_NAME"
LOG_DIR="/HOME/scw6fgn/run/VideoMamba-main/videomamba/video_sm/exp/k400/videomamba_small/outputcasia25/logs/${JOB_NAME}"
PREFIX='/HOME/scw6fgn/run/dataset/SinglePerson1/csv23/'
DATA_PATH='/HOME/scw6fgn/run/dataset/SinglePerson1/csv23/'

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
        python /HOME/scw6fgn/run/VideoMamba-main/videomamba/video_sm/run_class_finetuning_mst_2.py \
        --model videomamba_small \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'Kinetics_sparse' \
        --split ',' \
        --nb_classes 8 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
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
