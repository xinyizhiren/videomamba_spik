#!/bin/bash
module load cuda/11.8
module load gcc/12.2

export PATH=/HOME/scw6fgn/run/anaconda3/bin:$PATH
source activate /HOME/scw6fgn/run/anaconda3/envs/Videomamba
#export PATH=/HOME/scw6fgn/run/anaconda3/envs/Videomamba/bin:$PATH
export PYTHONUNBUFFERED=1

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

#JOB_NAME='(3.31)_baseline+cut)'
#JOB_NAME='(3.14)_0.5*etloss_nohsic_nostillmix)'
#JOB_NAME='(3.19)_baseline)'
#JOB_NAME='(4.1)_baseline+rcut+3)'
#叠加3张（t，t-1，t-2），1/9大小一列，只处理view1，视频外随机，视频内也随机位置
#JOB_NAME='(4.1)_baseline+rcut_g+3)'
#叠加3张（t，t-1，t-2），1/9大小一列，只处理view1，视频外随机，视频内固定位置
# JOB_NAME='(4.2)_baseline+rcut_g+3_r)'
#叠加3张（t，t-1，t-2），1/9大小，只处理view1，视频外随机，视频内固定位置，三个不并排随机
#JOB_NAME='(4.2)_baseline+rcut_g+3_r_16)'
#叠加3张（t，t-1，t-2），1/16大小，只处理view1，视频外随机，视频内固定位置，三个不并排随机
#JOB_NAME='(4.7)_base_et+rcut_g+3_r_16)'
#叠加3张（t，t-1，t-2），1/16大小，证据融合，只处理view1，视频外随机，视频内固定位置，三个不并排随机,100%应用，单应用
#JOB_NAME='(4.7)_base_et+rcut_g+3_r_16_0.8)'
#叠加3张（t，t-1，t-2），1/16大小，证据融合，只处理view1，视频外随机，视频内固定位置，三个不并排随机,80%应用，单应用
#JOB_NAME='(4.7)_base_et+rcut_g+3_r_16_0.6)'
#叠加3张（t，t-1，t-2），1/16大小，证据融合，只处理view1，视频外随机，视频内固定位置，三个不并排随机,60%应用，单应用
#JOB_NAME='(4.8)_baseline)'
#JOB_NAME='(4.8)_baseline)_et'
#_et是方法三，token当证据量
#JOB_NAME='(4.8)_baseline)_et_2'
#head当证据量融合，方法二
#JOB_NAME='(4.9)_baseline_et_2loss'
#方法三+双视角loss
#JOB_NAME='(4.13)_baseline_edlloss'
#方法三+edlloss
#JOB_NAME='(4.13)_baseline_edlloss_2'
##方法二+edlloss
#JOB_NAME='(4.13)_baseline_edlloss_4'
##方法四+edlloss
#JOB_NAME='(4.16)_baseline)'
#23训练1测试
#JOB_NAME='(4.16)_baseline)_2et'
#23训练1测试，et方法2
#JOB_NAME='(4.16)_baseline)_3et'
#23训练1测试，et方法3

#JOB_NAME='(4.16)_baseline_2test)'
#13训练2测试
#JOB_NAME='(4.16)_baseline)_2test_2et'
#13训练2测试,et方法2
#JOB_NAME='(4.16)_baseline)_2test_3et'
#13训练2测试,et方法3

#JOB_NAME='(4.17)_baseline)_1test_2et'
#23训练1测试，et方法2，更换为融合head
#JOB_NAME='(4.17)_baseline)_2test_2et'
#13训练2测试，et方法2，更换为融合head

#JOB_NAME='(4.21)_baseline)_casia'
#互动数据集
#JOB_NAME='(4.21)_baseline)_casia_64'
#互动数据集
#JOB_NAME='(4.21)_baseline)_casia_64_221'
#互动数据集,22训练1测试
#JOB_NAME='(4.21)_baseline)'
#
#JOB_NAME='(4.23)_baseline_single)'
#12训练，3测试，对齐

#JOB_NAME='(4.23)_baseline_casia)'
#12平均融合casia,2e-4
#JOB_NAME='(4.23)_baseline_casia_lr6)'
#12平均融合casia,6e-4
#JOB_NAME='(4.23)_baseline_casia_lr-5)'
#12平均融合casia,2e-5
#JOB_NAME='(4.23)_baseline_casia_1e-4_1231)'
#12平均融合casia,1e-4

#JOB_NAME='(4.23)_baseline_single_1122)'
#平均融合
#JOB_NAME='(4.23)_baseline_single_2233)'
#平均融合
#JOB_NAME='(4.23)_baseline_single_3311)'
#平均融合

#JOB_NAME='(4.23)_baseline_all_et_1233)'
#et融合
#JOB_NAME='(4.23)_baseline_all_et3_1233)'
#方法3融合

#JOB_NAME='(4.23)_baseline_casia_2111)'
#平均融合,6e-4,2视角训练，1视角测试

#JOB_NAME='(4.25)_baseline_casia_1233_lr1e4)'
#baseline裁剪数据集
#JOB_NAME='(4.25)_baseline_casia_1122_lr1e4)'
#baseline裁剪数据集
#JOB_NAME='(4.25)_baseline_casia_1233_lr2e4)'
#baseline裁剪数据集
#JOB_NAME='(4.25)_baseline_casia_1122_lr2e4)'
#baseline裁剪数据集
#JOB_NAME='(4.25)_baseline_casia_2211_lr2e4)'
#baseline裁剪数据集
#JOB_NAME='(4.25)_baseline_casia_1122_lr2e4_2)'
#JOB_NAME='(4.25)_baseline_casia_1122test)'

#JOB_NAME='(4.25)_baseline_et_casia_1233_lr2e4)'
#baseline裁剪数据集

#JOB_NAME='(4.25)_baseline_casia_1233_100lr1e4)'
#baseline裁剪数据集

#JOB_NAME='(4.25)_baseline_et3_casia_1233_1)'
#baseline裁剪数据集et3,16帧,1空间裁剪
#JOB_NAME='(4.25)_baseline_et3_casia_1233_1_32)'
#baseline裁剪数据集et3,32帧,1空间裁剪
#JOB_NAME='(4.25)_baseline_et3_casia_2311_1_32)'
#baseline裁剪数据集et3,32帧,1空间裁剪
#JOB_NAME='(4.25)_baseline_et3_casia_1322_1_32)'
#baseline裁剪数据集et3,32帧,1空间裁剪

#JOB_NAME='(4.25)_baseline_casia_1322_1_32)'
#JOB_NAME='(4.25)_baseline_casia_1233_1_32)'
#JOB_NAME='(4.25)_baseline_casia_2311_1_32)'
#baseline，32帧,1空间裁剪

#JOB_NAME='(4.26)_baseline_et_casia_1322_1_32)'
#JOB_NAME='(4.26)_baseline_et_casia_1233_1_32)'
#JOB_NAME='(4.26)_baseline_et_casia_2311_1_32)'
#baseline，32帧,1空间裁剪,et2

#JOB_NAME='(4.26)_baseline_et3_casia_1233_1_32)'
#baseline裁剪数据集et3,32帧,1空间裁剪

#JOB_NAME='(4.26)_baseline_single_1123_1_32)'
#JOB_NAME='(4.26)_baseline_single_2213_1_32)'
#JOB_NAME='(4.26)_baseline_single_3312_1_32)'
#baseline裁剪数据集et3,32帧,1空间裁剪,单视角

#JOB_NAME='(4.25)_baseline_casia_1322_1_32)'
#JOB_NAME='(4.25)_baseline_casia_1233_1_32)'
#JOB_NAME='(4.28)_baseline_casia_2311_1_32)'
#JOB_NAME='(4.28)_baseline_casia_2311_4_32)'
#JOB_NAME='(4.28)_baseline_casia_1322_4_32)'
#JOB_NAME='(4.28)_baseline_casia_1233_4_32)'
#baseline，32帧,1空间裁剪
#JOB_NAME='(4.28)_baseline_et3_edlloss_casia_2311_1_32)'

#JOB_NAME='(4.28)_baseline_et3_edlloss_casia_2311_4_32)'
#JOB_NAME='(4.28)_baseline_et3_edlloss_casia_1322_4_32)'
#JOB_NAME='(4.28)_baseline_et3_edlloss_casia_1233_4_32)'

#JOB_NAME='(4.28)_baseline_et2_edlloss_casia_2311_4_32)'
#JOB_NAME='(4.28)_baseline_et2_1edlloss_casia_2311_4_32)'


#JOB_NAME='(4.28)_baseline_et2_0.6_1.3_3view1_0.5edlloss_casia_2311_4_32)'
#只处理了view1,概率60%，1/3大小，edl损失
#JOB_NAME='(4.28)_baseline_et2_0.6_1.3_3view_0.5edlloss_casia_2311_4_32)'
#处理了view1+view2,概率60%，1/3大小，edl损失
#JOB_NAME='(4.28)_baseline_et2_0.6_1.4_3view_0.5edlloss_casia_2311_4_32)'
#处理了view1+view2,概率60%，1/4大小，edl损失

#JOB_NAME='(5.1)_test)'
# 2e-4
#JOB_NAME='(5.1)_test_2)'
# 6e-4
#JOB_NAME='(5.1)_test_3)'
# 6e-3
#JOB_NAME='(5.1)_test_4)'
# 6e-3
#JOB_NAME='(5.1)_test_5)'
# 6e-3 tau=4.0, v_threshold=0.5
#JOB_NAME='(5.1)_test_6)'
# 2e-5 tau=4.0, v_threshold=0.5
#JOB_NAME='(5.1)_test_et_7)'
# 2e-5 tau=4.0, v_threshold=0.5
#JOB_NAME='(5.1)_test_et_8)'
# 2e-4 tau=4.0, v_threshold=0.5
#JOB_NAME='(5.1)_test_et_9)'
# 2e-4 tau=4.0, v_threshold=0.1
#JOB_NAME='(5.1)_test_et_10)'
# 2e-4 tau=2
#JOB_NAME='(5.2)_test_1)'
# 1e-3 tau=2,2+22层
#JOB_NAME='(5.2)_test_2)'
# 1e-3 tau=2,2+22层
#JOB_NAME='(5.2)_test_3)'
# 1e-3 baseline
#JOB_NAME='(5.2)_test_4)'
# 1e-3 baseline,预训练
#JOB_NAME='(5.2)_test_5)'
# 2e-4 baseline,无预训练
#JOB_NAME='(5.2)_test_6)'
# 2e-4 baseline,预训练
#JOB_NAME='(5.2)_test_7)'
# 2e-4 baseline,预训练,模型2，nucla
#JOB_NAME='(5.2)_test_8)'
# 2e-4 baseline,预训练,模型2，casia
#JOB_NAME='(5.2)_test_9)'
# 2e-4 baseline,预训练,模型2，casia,100轮
#JOB_NAME='(5.3)_test_1)'
# 2e-4 baseline,预训练,模型2，nucla,100轮,16时间步
#JOB_NAME='(5.3)_test_2)'
# 2e-4 baseline,预训练,模型2，nucla,100轮,32时间步
#JOB_NAME='(5.3)_test_3)'
# 2e-4 baseline,预训练,模型2，nucla,100轮,24时间步
#JOB_NAME='(5.3)_test_4)'
# 2e-4 baseline,预训练,模型2，nucla,100轮,8时间步

# nucla数据集
#JOB_NAME='(5.21)_0.5ce2_1233)'
#JOB_NAME='(5.21)_0.1ce2__1233)'
# 0.1目录下是5
#JOB_NAME='(5.21)_1ce2__1233)'
#JOB_NAME='(5.21)_5ce2__1233)'
# 5目录下是0.1
#JOB_NAME='(5.21)_0ce2__1233)'
#JOB_NAME='(5.21)_10ce2__1233)'

#JOB_NAME='(5.22)_0.5ce3_1233)'
#JOB_NAME='(5.22)_0.1ce3__1233)'
#JOB_NAME='(5.22)_1ce3__1233)'
# 0.5和1互换

#JOB_NAME='(5.22)_5ce3__1233)'
#JOB_NAME='(5.22)_0ce3__1233)'
#JOB_NAME='(5.22)_10ce3__1233)'

# k=1（互访双图）,80%，对比，5ce2,3帧互放，大小1/2,1/3,1/4,1/5
#JOB_NAME='(5.22)_2_0.8_1233)'
#JOB_NAME='(5.22)_3_0.8_1233)'
#JOB_NAME='(5.22)_4_0.8_1233)'
#JOB_NAME='(5.22)_5_0.8_1233)'

# k=2（互访双图）,80%，对比，5ce2,3帧互放，大小1/2,1/3,1/4,1/5
#JOB_NAME='(5.22)_2_2_0.8_1233)'
#JOB_NAME='(5.22)_2_3_0.8_1233)'
#JOB_NAME='(5.22)_2_4_0.8_1233)'
#JOB_NAME='(5.22)_2_5_0.8_1233)'

#casia,23训练1验证
#JOB_NAME='(5.22)_0.1_2311)'
#JOB_NAME='(5.22)_0.5_2311)'
#JOB_NAME='(5.22)_1_2311)'
#JOB_NAME='(5.22)_5_2311)'
#JOB_NAME='(5.22)_10_2311)'
#JOB_NAME='(5.22)_0_2311)'
#JOB_NAME='(6.8)_2311)'
#JOB_NAME='(6.9)_test2'
#JOB_NAME='(6.9)_hope'
#JOB_NAME='(6.9)_hope2'
JOB_NAME='(6.12)_hope'
OUTPUT_DIR="/HOME/scw6fgn/run/VideoMamba-main/videomamba/video_sm/exp/k400/videomamba_small/ntu/$JOB_NAME"
LOG_DIR="/HOME/scw6fgn/run/VideoMamba-main/videomamba/video_sm/exp/k400/videomamba_small/ntu/logs/${JOB_NAME}"
#PREFIX='/HOME/scw6fgn/run/dataset/multiview_action_videos/csv12/'
#DATA_PATH='/HOME/scw6fgn/run/dataset/multiview_action_videos/csv12/'
#PREFIX='/HOME/scw6fgn/run/dataset/ixmas_avi_csv/'
#DATA_PATH='/HOME/scw6fgn/run/dataset/ixmas_avi_csv/'
PREFIX='/HOME/scw6fgn/run/dataset/'
DATA_PATH='/HOME/scw6fgn/run/dataset/'

#PREFIX='/HOME/scw6fgn/run/dataset/Interaction/'
#DATA_PATH='/HOME/scw6fgn/run/dataset/Interaction/'

#PREFIX='/HOME/scw6fgn/run/dataset/SinglePerson1/csv23/'
#DATA_PATH='/HOME/scw6fgn/run/dataset/SinglePerson1/csv23/'

PARTITION='video5'
GPUS=4
GPUS_PER_NODE=4
CPUS_PER_TASK=6

srun -p $PARTITION \
--job-name=${JOB_NAME} \
--gres=gpu:${GPUS_PER_NODE} \
--ntasks=${GPUS} \
--ntasks-per-node=${GPUS_PER_NODE} \
--cpus-per-task=${CPUS_PER_TASK} \
--kill-on-bad-exit=1 \
python /HOME/scw6fgn/run/VideoMamba-main/videomamba/video_sm/run_class_finetuning_et.py \
--model videomamba_small \
--data_path ${DATA_PATH} \
--prefix ${PREFIX} \
--data_set 'Kinetics_sparse' \
--split ',' \
--nb_classes 60 \
--log_dir ${OUTPUT_DIR} \
--output_dir ${OUTPUT_DIR} \
--batch_size 10 \
--num_sample 2 \
--input_size 224 \
--short_side_size 224 \
--save_ckpt_freq 50 \
--num_frames 8 \
--num_workers 5 \
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

