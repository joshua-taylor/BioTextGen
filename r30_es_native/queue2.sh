#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/claude/es
while pgrep -f "queue[.]sh" > /dev/null; do sleep 15; done
python3 sweep.py cfg_s6.json res_s6.jsonl 2 > log_s6.txt 2>&1
EVO='[["c",1],["c",2],["c",3],["c",4],["c",5],["c",6],["c",7],["c",8],["ex",11],["ex",3],["wd"],["pw"]]'
python3 e_long.py res_long.jsonl evo-L2-h32k "{\"feats\": $EVO, \"horizon\": 32768, \"cap\": 1024}" '[[0,128],[30000,1024],[100000,2048],[300000,4096],[1000000,8192]]' > log_long1.txt 2>&1 &
python3 e_long.py res_long.jsonl evo-L2-h1k "{\"feats\": $EVO, \"horizon\": 1024, \"cap\": 128}" '[[0,128],[30000,1024],[100000,2048],[300000,4096],[1000000,8192]]' > log_long2.txt 2>&1
wait
