#!/bin/bash

model=${1:-casc_agent_1_new}
num_eps=${2:-3}

echo "Carla Simulation Run 1/12 - Local/N=0/S=0"
python run.py --model_name $model --offload_policy local --num_episodes $num_eps

echo "Carla Simulation Run 2/12 - Local/N=1/S=0"
python run.py --model_name $model --offload_policy local --num_episodes $num_eps  -gaussian

echo "Carla Simulation Run 3/12 - Local/N=0/S=1"
python run.py --model_name $model --offload_policy local --num_episodes $num_eps  -safety_filter

echo "Carla Simulation Run 4/12 - Local/N=1/S=1"
python run.py --model_name $model --offload_policy local --num_episodes $num_eps   -gaussian -safety_filter

echo "Carla Simulation Run 5/12 - Eager/N=0/S=0"
python run.py --model_name $model --offload_policy Shield2 --num_episodes $num_eps 

echo "Carla Simulation Run 6/12 - Eager/N=1/S=0"
python run.py --model_name $model --offload_policy Shield2 --num_episodes $num_eps  -gaussian

echo "Carla Simulation Run 7/12 - Eager/N=0/S=1"
python run.py --model_name $model --offload_policy Shield2 --num_episodes $num_eps  -safety_filter

echo "Carla Simulation Run 8/12 - Eager/N=1/S=1"
python run.py --model_name $model --offload_policy Shield2 --num_episodes $num_eps   -gaussian -safety_filter

echo "Carla Simulation Run 9/12 - Uniform/N=0/S=0"
python run.py --model_name $model --offload_policy Shield2 --off_belay --num_episodes $num_eps 

echo "Carla Simulation Run 10/12 - Uniform/N=1/S=0"
python run.py --model_name $model --offload_policy Shield2 --off_belay --num_episodes $num_eps  -gaussian

echo "Carla Simulation Run 11/12 - Uniform/N=0/S=1"
python run.py --model_name $model --offload_policy Shield2 --off_belay --num_episodes $num_eps  -safety_filter

echo "Carla Simulation Run 12/12 - Uniform/N=1/S=1"
python run.py --model_name $model --offload_policy Shield2 --off_belay --num_episodes $num_eps   -gaussian -safety_filter