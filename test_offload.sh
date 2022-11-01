set -x

# Queuing Delays @ phi = 20

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --queue_state 9

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --queue_state 9

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --queue_state 19

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --queue_state 19

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --queue_state 49

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --queue_state 49

# Queueing Delays @ phi = 10

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10 --queue_state 9

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --phi_scale 10 --queue_state 9

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10 --queue_state 19

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --phi_scale 10 --queue_state 19

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10 --queue_state 49

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --phi_scale 10 --queue_state 49

# Queuing Delays @ 10 ms

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --queue_state 9

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --queue_state 9

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --queue_state 19

python run.py --model_name casc_agent_1 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --queue_state 19