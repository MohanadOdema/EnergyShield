set -x

# Two things to try next if time: 
# - local with local belay (gonna do it in next subsection anw)
# - 200 msec deadlines (not sure what's the value -- except for the specific values for a certain experiment)

# Shield 
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# Uniform Shield 
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian -safety_filter

# Local 
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy local --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# Offload (40 ms)
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# Offload (100 ms)
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# Uniform Offload (40 ms)
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian -safety_filter

# Uniform Offload (100 ms)
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian -safety_filter

# Offload w/ failsafe (40 ms)
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# Offload w/ failsafe (100 ms)
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# Adaptive (40 ms)
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# Adaptive (100 ms)
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# Uniform Adaptive (40 ms)
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian -safety_filter

# Uniform Adaptive (100 ms)
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian -safety_filter

# Adaptive w/ failsafe (40 ms)
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# Adaptive w/ failsafe (100 ms)
python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT 

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter

# -------------------------------------------------------------------------------------------------------------
# Offload
# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian -safety_filter

# # Offload_failsafe
# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload_failsafe --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload_failsafe --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload_failsafe --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian -safety_filter

# # Adaptive
# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian -safety_filter

# # Adaptive_failsafe
# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive_failsafe --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive_failsafe --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive_failsafe --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian -safety_filter

# # Safety FIlter for smaller dealdines
# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload --deadline 40 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload --deadline 100 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload_failsafe --deadline 40 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload_failsafe --deadline 100 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive --deadline 40 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive --deadline 100 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive_failsafe --deadline 40 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive_failsafe --deadline 100 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -gaussian -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload --deadline 40 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload --deadline 100 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload_failsafe --deadline 40 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload_failsafe --deadline 100 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload_failsafe --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive --deadline 40 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive --deadline 100 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive_failsafe --deadline 40 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive_failsafe --deadline 100 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

# python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive_failsafe --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter


