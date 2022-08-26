#Train 
python run.py --model_name casc_agent1 -obstacle --img_resolution 80p --len_route medium --reward_fn reward_speed_centering_angle_add -penalize_dist_obstacle --no_rendering --len_obs 5 