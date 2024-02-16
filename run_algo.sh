#!/bin/bash

export CUDA_VISIBLE_DEVICES=0 
export WANDB_API_KEY=1de9d7f978f2d0084e0fa2694af45584a1eb31e8

tasks=(toy marine-easy-medium-expert marine-hard-expert marine-hard-medium)

for t in "${tasks[@]}"
do
   for i in {0..2}
   do
      echo "===========================Running task $t for seed=$i==========================="
      ok=`python src/main.py --mto --config=hybrid --env-config=sc2_offline --task-config=$t --seed=$i`
      sleep 10s
   done
done