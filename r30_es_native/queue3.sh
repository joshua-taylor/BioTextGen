#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/claude/es
while pgrep -f "queue[.]sh|queue2[.]sh" > /dev/null; do sleep 15; done
python3 sweep.py cfg_s7.json res_s7.jsonl 2 > log_s7.txt 2>&1
