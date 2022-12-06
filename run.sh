#!/bin/bash
#SBATCH --job-name=sencell
#SBATCH --output="log/%j_log.txt"
#SBATCH --account=PCON0022
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=10:00:00
#SBATCH --gpus-per-node=1

set -x

# conda activate deepmaps_env

dataset_name=s5
sencell_num=100

cd /users/PCON0022/haocheng/sencell/outputs
mkdir ${SLURM_JOB_ID}
cd ..

time ~/.conda/envs/deepmaps_env/bin/python -u main.py \
                                        --exp_name ${dataset_name}_${SLURM_JOB_ID} \
                                        --output_dir ./outputs/${SLURM_JOB_ID} \
                                        --sencell_num ${sencell_num} \
                                        --retrain

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