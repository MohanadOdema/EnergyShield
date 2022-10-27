set -x

#phi
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 5

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --phi_scale 5

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --phi_scale 10

# Queuing Delays @ phi = 20

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --queue_state 9

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --queue_state 9

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --queue_state 19

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --queue_state 19

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --queue_state 49

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --queue_state 49

# Queueing Delays @ phi = 10

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10 --queue_state 9

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --phi_scale 10 --queue_state 9

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10 --queue_state 19

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --phi_scale 10 --queue_state 19

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10 --queue_state 49

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --phi_scale 10 --queue_state 49


# Shield@ 5 Mbps
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT --phi_scale 5

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian --phi_scale 5

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 5

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter --phi_scale 5

# Uniform Shield@ 5 Mbps
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT --phi_scale 5

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian --phi_scale 5

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --phi_scale 5

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian -safety_filter --phi_scale 5

# Shield@ 10 Mbps
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT --phi_scale 10

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian --phi_scale 10

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -gaussian -safety_filter --phi_scale 10

# Uniform Shield@ 10 Mbps
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT --phi_scale 10

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian --phi_scale 10

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --phi_scale 10

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -gaussian -safety_filter --phi_scale 10


# Queuing Delays @ 10 ms

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --queue_state 9

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --queue_state 9

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --queue_state 19

python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle -test --len_route short --cont_control True --off_belay --carla_map Town04_OPT -safety_filter --queue_state 19




# # phi values of 5, 10, and 20 Mbps runs. (all w/ safety filter on and off_belay)

# # Deadline 40: 5 policies*3 phis 

# # 5 Mbps
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 5

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 5

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 5

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 5

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 40 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 5

# # 10 Mbps
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 40 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10

# # 20 Mbps
# # python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 40 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 20

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 20

# # python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 40 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 20

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 40 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 20

# # python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 40 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 20


# # Deadline 100: 5 policies*3 phis 

# # 5 Mbps
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 5

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 5

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 5

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 5

# # python run.py --model_name casc_agen100t4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 5

# # 10 Mbps
# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10

# # python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 10

# # 20 Mbps
# # python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload --deadline 100 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 20

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy offload_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 20

# # python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive --deadline 100 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 20

# python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy adaptive_failsafe --deadline 100 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 20

# # python run.py --model_name casc_agent4 -obstacle --len_obs 4 --offload_policy Shield2 --deadline 100 --obs_start_idx 40 --num_episodes 35 --off_belay -penalize_dist_obstacle -test --len_route short --cont_control True --carla_map Town04_OPT -safety_filter --phi_scale 20