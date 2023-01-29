#!/bin/bash

model=${1:-casc_agent_1_new}

echo "Generating statistics for all experimental configurations of model $model" 
echo "python ./scripts/compute_all_stats.py --model_name $model"
python ./scripts/plots/compute_all_stats.py --model_name $model
