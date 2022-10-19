set -x
# Need to fix phi_scale and srate for all experiments.

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

python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload --deadline 40 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload --deadline 100 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload_failsafe --deadline 40 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload_failsafe --deadline 100 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy offload_failsafe --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive --deadline 40 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive --deadline 100 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive_failsafe --deadline 40 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive_failsafe --deadline 100 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter

python run.py --model_name casc_agent3 -obstacle --len_obs 4 -test --offload_policy adaptive_failsafe --deadline 200 --phi_scale 20 --srate 900 --obs_start_idx 20 --len_route short --num_episodes 35 -safety_filter


