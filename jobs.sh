# for sencell_num in 25 50 75 100 125 150 175
# do

# sbatch --export=sencell_num=${sencell_num} run.sh
# sbatch --export=sencell_num=${sencell_num} run.sh
# sbatch --export=sencell_num=${sencell_num} run.sh

# done





nohup python -u main.py --exp_name fixbatch --batch_id 0 --device_index 1 --output_dir ./outputs/23-11-28-21-45-fixbatch > ./log/fixRNA_0.log 2>&1 &
nohup python -u main.py --exp_name fixbatch --batch_id 1 --device_index 2 --output_dir ./outputs/23-11-28-21-46-fixbatch > ./log/fixRNA_1.log 2>&1 &
nohup python -u main.py --exp_name fixbatch --batch_id 2 --device_index 3 --retrain > ./log/fixRNA_2.log 2>&1 &
nohup python -u main.py --exp_name fixbatch --batch_id 3 --device_index 4 --retrain > ./log/fixRNA_3.log 2>&1 &
nohup python -u main.py --exp_name fixbatch --batch_id 4 --device_index 5 --retrain > ./log/fixRNA_4.log 2>&1 &




# Submitted batch job 13975477
# Submitted batch job 13975478
# Submitted batch job 13975479
# Submitted batch job 13975480
# Submitted batch job 13975481
# Submitted batch job 13975482
# Submitted batch job 13975483
# Submitted batch job 13975484
# Submitted batch job 13975485
# Submitted batch job 13975486
# Submitted batch job 13975487
# Submitted batch job 13975488
# Submitted batch job 13975489
# Submitted batch job 13975490
# Submitted batch job 13975491
# Submitted batch job 13975492
# Submitted batch job 13975493
# Submitted batch job 13975494
# Submitted batch job 13975495
# Submitted batch job 13975496
# Submitted batch job 13975497

# Submitted batch job 13976109
# Submitted batch job 13976110
# Submitted batch job 13976111
# Submitted batch job 13976112
# Submitted batch job 13976113
# Submitted batch job 13976114
# Submitted batch job 13976115
# Submitted batch job 13976116
# Submitted batch job 13976117
# Submitted batch job 13976118
# Submitted batch job 13976119
# Submitted batch job 13976120
# Submitted batch job 13976121
# Submitted batch job 13976122
# Submitted batch job 13976123
# Submitted batch job 13976124
# Submitted batch job 13976125
# Submitted batch job 13976126
# Submitted batch job 13976127
# Submitted batch job 13976128
# Submitted batch job 13976129