#!/bin/sh
# run jobs strictly one after another (each job uses both cores)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/claude/es
python3 sweep.py cfg_s5.json res_s5.jsonl 2 > log_s5.txt 2>&1
python3 e_prefill.py res_prefill.jsonl 1024 128 h1k > log_pf1.txt 2>&1 &
python3 e_prefill.py res_prefill.jsonl 32768 512 h32k > log_pf2.txt 2>&1
wait
