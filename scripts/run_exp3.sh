#!/bin/bash

num_eps=${1:-3}

echo "Repeating experiment 1 for Agent 2"
./scripts/run_exp.sh $num_eps casc_agent_2_new

echo "Repeating experiment 1 for Agent 3"
./scripts/run_exp.sh $num_eps casc_agent_3_new

echo "Repeating experiment 1 for Agent 4"
./scripts/run_exp.sh $num_eps casc_agent_4_new

echo "Generating csv files for every model"
python ./scripts/plots/compute_all_stats.py --model_name casc_agent_1_new
python ./scripts/plots/compute_all_stats.py --model_name casc_agent_2_new
python ./scripts/plots/compute_all_stats.py --model_name casc_agent_3_new
python ./scripts/plots/compute_all_stats.py --model_name casc_agent_4_new