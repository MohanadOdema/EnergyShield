#!/bin/bash

model=${1:-casc_agent_1_new}

echo "Generating Figure 8 Results.." 
echo "python ./scripts/plots/plot_exp2.py --model_name $model --mode windows"
python ./scripts/plots/plot_exp2.py --model_name $model --mode windows

echo "Generating Figure 9 Results.." 
echo "python ./scripts/plots/plot_exp2.py --model_name $model --mode energy"
python ./scripts/plots/plot_exp2.py --model_name $model --mode energy

cp -r -n models/$model/experiments ../results/raw_data