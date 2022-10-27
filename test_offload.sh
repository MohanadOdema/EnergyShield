set -x

########## Shield vs Local ########

# Local 
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 100 --obs_start_idx 40 --num_episodes 6 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 100 --obs_start_idx 40 --num_episodes 3 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 100 --obs_start_idx 40 --num_episodes 5 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 100 --obs_start_idx 40 --num_episodes 5 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# Shield 
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# Uniform Shield 
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian -safety_filter


######### Varying throughput ##########

sh test_offload2.sh 

######### Local Belays #############

# @ 40 ms
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --local_belay --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --local_belay --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --local_belay --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --local_belay --carla_map Town04_OPT -gaussian -safety_filter

# @ 100 ms
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --local_belay --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --local_belay --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --local_belay --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --local_belay --carla_map Town04_OPT -gaussian -safety_filter

# # Offload (40 ms)
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# # Offload (100 ms)
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# # Uniform Offload (40 ms)
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT 

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian -safety_filter

# # Uniform Offload (100 ms)
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT 

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian -safety_filter

# # Offload w/ failsafe (40 ms)
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# # Offload w/ failsafe (100 ms)
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# # Adaptive (40 ms)
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# # Adaptive (100 ms)
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# # Uniform Adaptive (40 ms)
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT 

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian -safety_filter

# # Uniform Adaptive (100 ms)
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT 

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian -safety_filter

# # Adaptive w/ failsafe (40 ms)
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# # Adaptive w/ failsafe (100 ms)
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

