set -x

#Test casc model 1
# python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 3 --obs_start_idx 30 --num_episodes 30

# python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 3 --obs_start_idx 30 --num_episodes 30 -gaussian

# python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 3 --obs_start_idx 30 --num_episodes 30 -safety_filter

# python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 3 --obs_start_idx 30 --num_episodes 30 -gaussian -safety_filter

python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30 -safety_filter

python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30 -gaussian -safety_filter

python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30

python run.py --model_name casc_agent1 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 30 -gaussian

# python run.py --model_name casc_agent4 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 10

# python run.py --model_name casc_agent4 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 10 -gaussian

# python run.py --model_name casc_agent4 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 10 -safety_filter

# python run.py --model_name casc_agent4 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 10 -gaussian -safety_filter

#Test casc model 2
# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 3 --obs_start_idx 30 --num_episodes 30

# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 3 --obs_start_idx 30 --num_episodes 30 -gaussian

# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 3 --obs_start_idx 30 --num_episodes 30 -safety_filter

# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 3 --obs_start_idx 30 --num_episodes 30 -gaussian -safety_filter

# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30

# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30 -gaussian

# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30 -safety_filter

# python run.py --model_name casc_agent2 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30 -gaussian -safety_filter

#Test casc model 3
# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 3 --obs_start_idx 30 --num_episodes 30

# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 3 --obs_start_idx 30 --num_episodes 30 -gaussian

# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 3 --obs_start_idx 30 --num_episodes 30 -safety_filter

# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 3 --obs_start_idx 30 --num_episodes 30 -gaussian -safety_filter

# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30

# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30 -gaussian

# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30 -safety_filter

# python run.py --model_name casc_agent3 -obstacle -test --offload_policy local --len_obs 5 --obs_start_idx 15 --num_episodes 30 -gaussian -safety_filter
