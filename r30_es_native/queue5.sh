#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/claude/es
while pgrep -f "queue3[.]sh" > /dev/null; do sleep 15; done
python3 sweep.py cfg_s8.json res_s8.jsonl 2 > log_s8.txt 2>&1
