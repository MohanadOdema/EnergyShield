#Train 
python run.py --model_name casc_agent1 -obstacle --img_resolution 80p --len_route medium --reward_fn reward_speed_centering_angle_add -penalize_dist_obstacle --no_rendering --len_obs 5 

# docker command
docker run -d -e SDL_VIDEODRIVER=offscreen -e SDL_HINT_CUDA_DEVICE=0 -p 2000-2002:2000-2002 -it --runtime=nvidia --gpus all carlasim/carla:0.9.11 ./CarlaUE4.sh -opengl -nosound -windowed -ResX=800 -ResY=600