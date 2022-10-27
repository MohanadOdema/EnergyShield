set -x

# On server run example:
# sudo env "PATH=$PATH" python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 100 -safety_filter -display_off --port 2000

# Casc_agent_A from aicps server on short route - Local
python run.py --model_name casc_agent_A -obstacle -test --deadline 100 --offload_policy local --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT

python run.py --model_name casc_agent_A -obstacle -test --deadline 100 --offload_policy local --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent_A -obstacle -test --deadline 100 --offload_policy local --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent_A -obstacle -test --deadline 100 --offload_policy local --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT  -gaussian -safety_filter

# Casc_agent_A from aicps server on short route - Shield Eager
python run.py --model_name casc_agent_A -obstacle -test --deadline 100 --offload_policy Shield2 --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT

python run.py --model_name casc_agent_A -obstacle -test --deadline 100 --offload_policy Shield2 --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent_A -obstacle -test --deadline 100 --offload_policy Shield2 --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent_A -obstacle -test --deadline 100 --offload_policy Shield2 --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT  -gaussian -safety_filter

# Casc_agent_A from aicps server on short route - Shield off_belay
python run.py --model_name casc_agent_A -obstacle -test --deadline 100 --offload_policy Shield2 --off_belay --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT

python run.py --model_name casc_agent_A -obstacle -test --deadline 100 --offload_policy Shield2 --off_belay --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent_A -obstacle -test --deadline 100 --offload_policy Shield2 --off_belay --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent_A -obstacle -test --deadline 100 --offload_policy Shield2 --off_belay --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT  -gaussian -safety_filter







#Test casc model 1
# python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30 -safety_filter

# python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30 -gaussian -safety_filter

# python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30

# python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30 -gaussian

#Test casc model 2
# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30 -safety_filter

# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30 -gaussian -safety_filter

# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30

# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30 -gaussian

# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30

# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30 -gaussian

# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30 -safety_filter

# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30 -gaussian -safety_filter

#Test casc model 3

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 --offload_policy local --deadline 100 --phi_scale 10 --srate 900 --obs_start_idx 20 --num_episodes 50 -penalize_dist_obstacle --len_route short --spawn_random --reward reward_speed_centering_angle_add

# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 70 -safety_filter

# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 70 -gaussian -safety_filter

# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 70

# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 70 -gaussian

# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30

# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30 -gaussian

# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30 -safety_filter

# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30 -gaussian -safety_filter

# # New models from rcpslserver (w/new shield r-2) w/layers

# python run.py --model_name casc_agent_B -obstacle --len_obs 4 --carla_map Town04 --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10

# python run.py --model_name casc_agent_B -obstacle --len_obs 4 --carla_map Town04 --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10 -gaussian

# python run.py --model_name casc_agent_B -obstacle --len_obs 4 --carla_map Town04 --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10 -safety_filter 

# python run.py --model_name casc_agent_B -obstacle --len_obs 4 --carla_map Town04 --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10 -gaussian -safety_filter

# python run.py --model_name casc_agent_C -obstacle --len_obs 4 --carla_map Town04 --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10

# python run.py --model_name casc_agent_C -obstacle --len_obs 4 --carla_map Town04 --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10 -gaussian

# python run.py --model_name casc_agent_C -obstacle --len_obs 4 --carla_map Town04 --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10 -safety_filter 

# python run.py --model_name casc_agent_C -obstacle --len_obs 4 --carla_map Town04 --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10 -gaussian -safety_filter

# python run.py --model_name casc_agent_A -obstacle --len_obs 4 --carla_map Town04 --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 40

# python run.py --model_name casc_agent_A -obstacle --len_obs 4 --carla_map Town04 --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 40 -gaussian

# python run.py --model_name casc_agent_A -obstacle --len_obs 4 --carla_map Town04 --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 20 -safety_filter 

# python run.py --model_name casc_agent_A -obstacle --len_obs 4 --carla_map Town04 --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 20 -gaussian -safety_filter

# # New models from rcpslserver (w/new shield r-2) w/o layers
 
# python run.py --model_name casc_agent_B -obstacle --len_obs 4 --carla_map Town04_OPT --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10

# python run.py --model_name casc_agent_B -obstacle --len_obs 4 --carla_map Town04_OPT --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10 -gaussian

# python run.py --model_name casc_agent_B -obstacle --len_obs 4 --carla_map Town04_OPT --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10 -safety_filter 

# python run.py --model_name casc_agent_B -obstacle --len_obs 4 --carla_map Town04_OPT --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10 -gaussian -safety_filter

# python run.py --model_name casc_agent_C -obstacle --len_obs 4 --carla_map Town04_OPT --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10

# python run.py --model_name casc_agent_C -obstacle --len_obs 4 --carla_map Town04_OPT --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10 -gaussian

# python run.py --model_name casc_agent_C -obstacle --len_obs 4 --carla_map Town04_OPT --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10 -safety_filter 

# python run.py --model_name casc_agent_C -obstacle --len_obs 4 --carla_map Town04_OPT --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 10 -gaussian -safety_filter

# python run.py --model_name casc_agent_A -obstacle --len_obs 4 --carla_map Town04_OPT --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 40

# python run.py --model_name casc_agent_A -obstacle --len_obs 4 --carla_map Town04_OPT --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 40 -gaussian

# python run.py --model_name casc_agent_A -obstacle --len_obs 4 --carla_map Town04_OPT --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 20 -safety_filter 

# python run.py --model_name casc_agent_A -obstacle --len_obs 4 --carla_map Town04_OPT --offload_policy local --deadline 100 --obs_start_idx 40 --len_route medium -penalize_dist_obstacle -test --estimation_fn worst --num_episodes 20 -gaussian -safety_filter  


