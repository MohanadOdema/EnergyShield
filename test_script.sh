set -x

# Ex docker server instance
# docker run --name carla2 -d -e SDL_VIDEODRIVER=offscreen -p 3000-3002:3000-3002 -it --runtime=nvidia --gpus 'device=1' carlasim/carla:0.9.11 ./CarlaUE4.sh -carla-port=3000 -opengl -nosound

# On server run example:
# sudo env "PATH=$PATH" python run.py --model_name casc_agent_1 -obstacle -test --offload_policy local --len_obs 4 --obs_start_idx 20 --num_episodes 100 -safety_filter -display_off --port 2000

python run.py --model_name casc_agent_1 -obstacle -test --deadline 100 --offload_policy local --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT

python run.py --model_name casc_agent_1 -obstacle -test --deadline 100 --offload_policy local --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent_1 -obstacle -test --deadline 100 --offload_policy local --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent_1 -obstacle -test --deadline 100 --offload_policy local --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT  -gaussian -safety_filter

python run.py --model_name casc_agent_1 -obstacle -test --deadline 100 --offload_policy Shield2 --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT

python run.py --model_name casc_agent_1 -obstacle -test --deadline 100 --offload_policy Shield2 --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent_1 -obstacle -test --deadline 100 --offload_policy Shield2 --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent_1 -obstacle -test --deadline 100 --offload_policy Shield2 --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT  -gaussian -safety_filter

python run.py --model_name casc_agent_1 -obstacle -test --deadline 100 --offload_policy Shield2 --off_belay --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT

python run.py --model_name casc_agent_1 -obstacle -test --deadline 100 --offload_policy Shield2 --off_belay --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT -gaussian

python run.py --model_name casc_agent_1 -obstacle -test --deadline 100 --offload_policy Shield2 --off_belay --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT -safety_filter

python run.py --model_name casc_agent_1 -obstacle -test --deadline 100 --offload_policy Shield2 --off_belay --len_obs 4 --obs_start_idx 40 --num_episodes 35 -penalize_dist_obstacle --len_route short --cont_control True --carla_map Town04_OPT  -gaussian -safety_filter