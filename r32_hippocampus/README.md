# Round 32 — merged test: does a hippocampus-style memory beat text retrieval read twice?

One script, `hippo_test.py`. It fine-tunes SmolLM2-135M, separately for each arm, on a stream of freshly generated fictional
"fact worlds", then tests on held-out worlds, on one big memory holding 8 worlds, and on real names from MQuAKE.

## Run on Kaggle (T4, Internet ON)
```bash
!git clone -b round32-hippocampus https://github.com/joshua-taylor/BioTextGen && cd BioTextGen/r32_hippocampus
# 1) smoke test, ~10-15 min: checks the GPU path end to end
!cd BioTextGen/r32_hippocampus && python hippo_test.py --arms rag2,memfull --steps 200 --n_eval 2 --out smoke.json
# 2) full run (all 9 arms, ~3-5 h). Resumable: finished arms are skipped if you re-run with the same --out
!cd BioTextGen/r32_hippocampus && python hippo_test.py --arms all --steps 3000 --out r32_results.json
```
The script clones MQuAKE itself. If a session runs out, re-run the same command: it continues from the last finished arm.
To split across sessions: `--arms ctx_gold,mem_gold,rag1,rag2` then `--arms mem1way,mem2way,mem2way_twice,memlink_once,memfull`.

## Arms
| Arm | Facts reach the model as | Read |
|---|---|---|
| ctx_gold | the needed facts (+2 random) as text in the prompt — ceiling | given |
| mem_gold | the needed facts (+2 random) as 4 memory vectors each — can vectors carry a fact? | given |
| rag1 / rag2 | dense retrieval of fact sentences, as text | once / twice |
| mem1way | memory vectors, filed under subject only | once |
| mem2way | filed under subject and object | once |
| mem2way_twice | two-way, second read keyed from what the first read returned | twice |
| memlink_once | two-way + integrative linking (fact f + fact g whose subject is f's object) | once |
| memfull | two-way + linking + read twice | twice |

Retrieval heads are trained with the same supervision for text and memory arms (which stored fact a question needs).
Memory writes use no gradient through the LM (local encoding of each fact); the LM is fine-tuned to read.

## Kill rules (printed at the end)
- **K-main**: memfull beats rag2 by ≥5 points on reverse AND two-hop questions, in held-out worlds and the 8-world memory, with less context. Fail → the hippocampus adds nothing over retrieval; stop.
- **K-twoway**: two-way filing adds ≥20 points on reverse (mem2way vs mem1way).
- **K-link**: linking adds ≥10 points on two-hop (memfull vs mem2way_twice); else drop it.
- **K-capacity**: mem_gold within 10 points of ctx_gold on forward questions; else the vectors are too lossy (raise `--m`).

Please share the whole printed output (tables + kill rules) and `r32_results.json`.
