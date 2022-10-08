#Ex Train 
python run.py --model_name casc_agent1 -obstacle --img_resolution 80p --len_route medium --reward_fn reward_speed_centering_angle_add -penalize_dist_obstacle --no_rendering --len_obs 5 

# Ex Train with randomization
python run.py --model_name casc_agent2 -obstacle --len_obs 5 --offload_policy local --deadline 100 --phi_scale 10 --srate 900 --obs_start_idx 15 --num_episodes 0 --spawn_random -display_off --port 6000

# Ex docker command
docker run --name carla2 -d -e SDL_VIDEODRIVER=offscreen -p 3000-3002:3000-3002 -it --runtime=nvidia --gpus 'device=1' carlasim/carla:0.9.11 ./CarlaUE4.sh -carla-port=3000 -opengl -nosound
