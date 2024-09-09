#!/bin/bash

# NCU flags
ncu_flags="--set full --import-source yes --log-file /home/sylab/PunchShadow/WBPR/profiles/ncu/logs/ncu_log_bcsr.txt"

# Profiling file path
profiling_file_path="/home/sylab/PunchShadow/WBPR/profiles/ncu"

# bipartite file path
bipartite_file_path="/home/sylab/PunchShadow/WBPR/bipartite_data"

# Snap file path
snap_file_path="/home/sylab/PunchShadow/WBPR/snap_cap1_data"

echo "BCSR--------------------"

# # Command 1 - 
# echo "1: Running movielens-t-i_super.txt..."
# echo "\t start TC: $(date)"
# ncu $ncu_flags -o $profiling_file_path/movielen-u-i_bcsr_TC.ncu-rep ./maxflow -v 2 -f $bipartite_file_path/movielens-t-i_super.txt -a 0 -t 1 -a 0
# echo "\t end TC: $(date)"
# echo "\t start TLPNS: $(date)"
# ncu $ncu_flags -o $profiling_file_path/movielen-u-i_bcsr_TLPNS.ncu-rep ./maxflow -v 2 -f $bipartite_file_path/movielens-t-i_super.txt -a 0 -t 1 -a 1
# echo "\t end TLPNS: $(date)"
# echo "-------------------------------------------"

# # Command 2 -
# echo "2: Running youtube_super.txt..."
# echo "\t start TC: $(date)"
# ncu $ncu_flags -o $profiling_file_path/youtube_bcsr_TC.ncu-rep ./maxflow -v 2 -f $bipartite_file_path/youtube_super.txt -a 0 -t 1 -a 0
# echo "\t end TC: $(date)"
# echo "\t start TLPNS: $(date)"
# ncu $ncu_flags -o $profiling_file_path/youtube_bcsr_TLPNS.ncu-rep ./maxflow -v 2 -f $bipartite_file_path/youtube_super.txt -a 0 -t 1 -a 1
# echo "\t end TLPNS: $(date)"
# echo "-------------------------------------------"

# Command 3 -
echo "3: Running roadNet_PA_cap1.txt..."
echo "\t start TC: $(date)"
ncu $ncu_flags -o $profiling_file_path/roadNet_PA_bcsr_TC.ncu-rep ./maxflow -v 2 -f $snap_file_path/roadNet_PA_cap1.txt -a 0 -t 1 -a 0
echo "\t end TC: $(date)"
echo "\t start TLPNS: $(date)"
ncu $ncu_flags -o $profiling_file_path/roadNet_PA_bcsr_TLPNS.ncu-rep ./maxflow -v 2 -f $snap_file_path/roadNet_PA_cap1.txt -a 0 -t 1 -a 1
echo "\t end TLPNS: $(date)"
echo "-------------------------------------------"

# Command 4 -
echo "4: Running cit-Patents_cap1.txt..."
echo "\t start TC: $(date)"
ncu $ncu_flags -o $profiling_file_path/cit-Patents_bcsr_TC.ncu-rep ./maxflow -v 2 -f $snap_file_path/cit-Patents_cap1.txt -a 0 -t 1 -a 0
echo "\t end TC: $(date)"
echo "\t start TLPNS: $(date)"
ncu $ncu_flags -o $profiling_file_path/cit-Patents_bcsr_TLPNS.ncu-rep ./maxflow -v 2 -f $snap_file_path/cit-Patents_cap1.txt -a 0 -t 1 -a 1
echo "\t end TLPNS: $(date)"


