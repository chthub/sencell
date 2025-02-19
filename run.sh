#!/bin/bash
#SBATCH --job-name=sencell
#SBATCH --output="log/%j_log.txt"
#SBATCH --account=PCON0022
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=50G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=10:00:00
#SBATCH --gpus-per-node=1

if [ ! -d "./log" ]; then
    mkdir log
fi

if [ ! -d "./outputs" ]; then
    mkdir outputs
fi

set -x

# conda activate deepmaps_env

# dataset_name=disease
# sencell_num=250

# dataset_name=disease1
# sencell_num=250

# dataset_name=healthy
# sencell_num=100

# dataset_name=hd
# sencell_num=350

# env=osc
env=server


if [ $env == server ]; then
    cd outputs
    dir_name=$1 
    dataset_name=$3
    sencell_num=$4
    mkdir $dir_name
    cd ..

    time python -u main.py \
        --exp_name ${dataset_name}_${dir_name} \
        --output_dir ./outputs/${dir_name} \
        --gat_epoch 30 \
        --device_index $2 \
        --sencell_num ${sencell_num} \
        --retrain

elif [ $env == 'osc' ]; then
    cd /users/PCON0022/haocheng/sencell/outputs
    mkdir ${SLURM_JOB_ID}
    cd ..
    
    time ~/.conda/envs/deepmaps_env/bin/python -u main.py \
                                            --exp_name ${dataset_name}_${SLURM_JOB_ID} \
                                            --output_dir ./outputs/${SLURM_JOB_ID} \
                                            --gat_epoch 30 \
                                            --sencell_num ${sencell_num} \
                                            --retrain
fi


# 固定nonsengene选择的随机种子，sencellnum=75
# Submitted batch job 13978331
# Submitted batch job 13978332
# Submitted batch job 13978333

# 所有gene都是nonsengene, sencellnum=75
# Submitted batch job 13978673
# Submitted batch job 13978674
# Submitted batch job 13978675

# 注释掉random seed代码
# Submitted batch job 13978680
# Submitted batch job 13978681
# Submitted batch job 13978682

# 不更新cell embedding, sencellnum=75
# Submitted batch job 13988087
# Submitted batch job 13988088
# Submitted batch job 13988089

# 不更新cell embedding, sencellnum=100
# Submitted batch job 13988090
# Submitted batch job 13988091
# Submitted batch job 13988092

# sencell稳定下来了，sengene还没稳定,sencellnum=100
# Submitted batch job 13989669
# Submitted batch job 13989670
# Submitted batch job 13989671

# healthy, sencell=100
# Submitted batch job 13992830
# Submitted batch job 13992837
# Submitted batch job 13992838

# disease, sencell=250, out of memory
# Submitted batch job 13992831
# Submitted batch job 13992833
# Submitted batch job 13992834

# disease, sencell=250
# Submitted batch job 13995010
# Submitted batch job 13995011
# Submitted batch job 13995012

# disease1, sencell=250
# Submitted batch job 14000959
# Submitted batch job 14000964
# Submitted batch job 14000965

# healthy, sencell=100, 15轮
# Submitted batch job 14000967
# Submitted batch job 14000968
# Submitted batch job 14000969

# healthy, sencell=100， gat epoch=1
# Submitted batch job 14001068
# Submitted batch job 14001069
# Submitted batch job 14001070

# disease1, sencell=250, gat epoch=1
# Submitted batch job 14001082
# Submitted batch job 14001083
# Submitted batch job 14001084

# hd, sencell num=350
# Submitted batch job 14001124
# Submitted batch job 14001125
# Submitted batch job 14001126