#!/bin/bash

model=${1:-casc_agent_1_new}

echo "Generating Figure 5 Results.." 
echo "python ./scripts/plots/plot_exp1_Fig5.py --model_name $model"
python ./scripts/plots/plot_exp1_Fig5.py --model_name $model

echo "Generating Figure 6 Results.."
echo "python ./scripts/plots/plot_exp1_Fig6.py --model_name $model"
echo "python ./scripts/plots/plot_exp1_Fig6.py --model_name $model -gaussian"
python ./scripts/plots/plot_exp1_Fig6.py --model_name $model 
python ./scripts/plots/plot_exp1_Fig6.py --model_name $model -gaussian

echo "Generating Figure 7 Results.."
python ./scripts/plots/collect_distance_data.py --model_name $model # Generates average per 1m statistics tables for each experimental configuration
python ./scripts/plots/plot_exp1_Fig7.py --model_name $model

cp -r -n models/$model/experiments ../results/raw_data