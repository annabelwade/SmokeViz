#!/bin/bash

for seed in {13..20}; do
    sbatch --export=EXP_NUM=3,SEED=$seed --output=logs/exp3_seed$seed.log --job-name=exp3_seed$seed run_model.script
done
