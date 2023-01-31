#!/bin/bash

num_eps=${1:-3}
model=${2:-casc_agent_1_new}

# vary phi 5, 10 , 20 at q = 1
# vary q at 9 , 19 , 49 for phi =10

# Queuing Delays variation @ phi = 10

echo "Additional Wireless Variation Run 1/5 - phi=10/q=10"
python run.py --model_name $model --offload_policy Shield2 --num_episodes $num_eps --off_belay --carla_map Town04_OPT -safety_filter --phi_scale 10 --queue_state 9

echo "Additional Wireless Variation Run 2/5 - phi=10/q=20"
python run.py --model_name $model --offload_policy Shield2 --num_episodes $num_eps --off_belay --carla_map Town04_OPT -safety_filter --phi_scale 10 --queue_state 19

echo "Additional Wireless Variation Run 3/5 - phi=10/q=50"
python run.py --model_name $model --offload_policy Shield2 --num_episodes $num_eps --off_belay --carla_map Town04_OPT -safety_filter --phi_scale 10 --queue_state 49

# Throughput variation @ q = 1 ms (20 Mbps case was already generated in exp1)

echo "Additional Wireless Variation Run 4/5 - phi=5/q=1"
python run.py --model_name $model --offload_policy Shield2 --num_episodes $num_eps --off_belay --carla_map Town04_OPT -safety_filter --phi_scale 5

echo "Additional Wireless Variation Run 5/5 - phi=10/q=1"
python run.py --model_name $model --offload_policy Shield2 --num_episodes $num_eps --off_belay --carla_map Town04_OPT -safety_filter --phi_scale 10

echo "phi=20/q=1 case was already generated as part of experiment 1"

./scripts/exp2_generate_results.sh