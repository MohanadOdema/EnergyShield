set -x

# On server run example:
# sudo env "PATH=$PATH" python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 100 -safety_filter -display_off --port 2000

#Test casc model 1
python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30 -safety_filter

python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30 -gaussian -safety_filter

python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30

python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30 -gaussian

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
