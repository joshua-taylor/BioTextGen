#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/claude/es
while pgrep -f "queue3[.]sh|queue5[.]sh" > /dev/null; do sleep 15; done
EVO='[["c",1],["c",2],["c",3],["c",4],["c",5],["c",6],["c",7],["c",8],["ex",11],["ex",3],["wd"],["pw"]]'
python3 e_long_fixed.py res_longfix.jsonl evo-L2-h32k "{\"feats\": $EVO, \"horizon\": 32768, \"cap\": 512}" '[[0,128],[30000,1024],[100000,2048],[300000,4096],[1000000,8192]]' > log_longfix.txt 2>&1 &
EVO_STEPS=300 python3 evo2.py res_evo3.jsonl 12 6 6 '{"horizon": 32768}' > log_evo3.txt 2>&1
wait
