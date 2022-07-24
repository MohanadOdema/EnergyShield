import os
import random
import argparse
from CarlaEnv.offloading_utils import full_local_latency, bottleneck_local_latency
from run import train
import math
import tensorflow as tf
from multiprocessing import Process

def det_max_deadline(lower, step, multiplier):
	i = 1
	while (round(lower) + i) % step != 0:
		i = i + 1 
	current = round(lower) + i
	return current + step * multiplier

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Repeat experimental analysis across multiple deadlines")

	# PPO hyper parameters
	parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
	parser.add_argument("--lr_decay", type=float, default=1.0, help="Per-episode exponential learning rate decay")
	parser.add_argument("--discount_factor", type=float, default=0.99, help="GAE discount factor")
	parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
	parser.add_argument("--ppo_epsilon", type=float, default=0.2, help="PPO epsilon")
	parser.add_argument("--initial_std", type=float, default=1.0, help="Initial value of the std used in the gaussian policy")
	parser.add_argument("--value_scale", type=float, default=1.0, help="Value loss scale factor")
	parser.add_argument("--entropy_scale", type=float, default=0.01, help="Entropy loss scale factor")
	parser.add_argument("--horizon", type=int, default=128, help="Number of steps to simulate per training step")
	parser.add_argument("--num_epochs", type=int, default=3, help="Number of PPO training epochs per traning step")
	parser.add_argument("--batch_size", type=int, default=32, help="Epoch batch size")
	parser.add_argument("--num_episodes", type=int, default=2, help="Number of episodes to train for (0 or less trains forever)")

	# VAE parameters
	parser.add_argument("--vae_model", type=str,
						default="vae/models/seg_bce_cnn_zdim64_beta1_kl_tolerance0.0_data/",
						help="Trained VAE model to load")
	parser.add_argument("--vae_model_type", type=str, default=None, help="VAE model type (\"cnn\" or \"mlp\")")
	parser.add_argument("--vae_z_dim", type=int, default=None, help="Size of VAE bottleneck")

	# Environment settings
	parser.add_argument("--synchronous", type=int, default=True, help="Set this to True when running in a synchronous environment")
	parser.add_argument("--fps", type=int, default=30, help="Set this to the FPS of the environment")
	parser.add_argument("--action_smoothing", type=float, default=0.0, help="Action smoothing factor")
	parser.add_argument("-start_carla", action="store_true", help="Automatically start CALRA with the given environment settings")

	# Training parameters
	parser.add_argument("--model_name", type=str, required=True, help="Name of the model to train. Output written to models/model_name", choices=['agent1', 'agent2', 'agent3', 'BasicAgent', 'BehaviorAgent'])
	parser.add_argument("--reward_fn", type=str,
						default="reward_speed_centering_angle_multiply",
						help="Reward function to use. See reward_functions.py for more info.")
	parser.add_argument("--seed", type=int, default=0,
						help="Seed to use. (Note that determinism unfortunately appears to not be garuanteed " +
							 "with this option in our experience)")
	parser.add_argument("--eval_interval", type=int, default=100, help="Number of episodes between evaluation runs")
	parser.add_argument("-record_eval", action="store_true", default=False,
						help="If True, save videos of evaluation episodes " +
							 "to models/model_name/videos/")

	# Safety Filter Setting
	parser.add_argument("-safety_filter", action="store_true", default=False, help="Filter Control actions")
	parser.add_argument("-penalize_steer_diff", action="store_true", default=False, help="Penalize RL's steering output and filter's output")
	parser.add_argument("-penalize_dist_obstacle", action="store_true", default=False, help="Add a penalty when the RL agent gets closer to an obstacle")

	parser.add_argument("-obstacle", action="store_true", default=False, help="Add obstacles")
	parser.add_argument("-gaussian", action="store_true", default=False, help="Randomize obstacles location using gaussian distribution")
	parser.add_argument("--track", type=int, default=1, help="Track Number")
	parser.add_argument("--pos_mul", type=int, default=2, help='obstacle position multiplier')
	parser.add_argument("-test", action="store_true", default=False, help="test")

	parser.add_argument("-restart", action="store_true",
						help="If True, delete existing model in models/model_name before starting training")

	# AV pipeline Offloading
	parser.add_argument("--arch", type=str, help="Name of the model running on the AV platform", choices=['ResNet18', 'ResNet50', 'DenseNet169', 'ViT', 'ResNet18_mimic', 'ResNet50_mimic', 'DenseNet169_mimic', 'all'], default='ResNet50')
	parser.add_argument("--offload_position", type=str, help="Offloading position", choices=['direct', '0.5_direct', '0.25_direct', 'bottleneck', 'all'], default='direct')
	parser.add_argument("--offload_policy", type=str, help="Offloading policy", choices=['local', 'offload', 'offload_failsafe', 'adaptive', 'adaptive_failsafe', 'all'], default='offload')    
	parser.add_argument("--bottleneck_ch", type=int, help="number of bottleneck channels", choices=[3,6,9,12], default=6)
	parser.add_argument("--bottleneck_quant", type=int, help="quantization of the output", choices=[8,16,32], default=8)
	parser.add_argument("--HW", type=str, help="AV Hardware", choices=['PX2', 'TX2', 'Orin', 'Xavier', 'Nano'], default='PX2')
	parser.add_argument("--deadline", type=int, help="time window", default=100)
	parser.add_argument("--img_resolution", type=str, help="enter offloaded image resolution", choices=['480p', '720p', '1080p', 'Radiate', 'TeslaFSD', 'Waymo', 'all'], default='720p')
	parser.add_argument("--comm_tech", type=str, help="the wireless technology", choices=['LTE', 'WiFi', '5G'], default='LTE')
	parser.add_argument("--conn_overhead", action="store_true", default=False, help="Account for the connection establishment overhead separately alongside data transfer")
	parser.add_argument("--rayleigh_sigma", type=int, help="Scale of the throughput's Rayleigh distribution -- default is the value from collected LTE traces", default=13.62)    
	parser.add_argument("--noise_scale", type=float, default=5, help="noise scale/variance")

    # Carla Config file
	parser.add_argument("--carla_map", type=str, default='Town04', help="load map")
	parser.add_argument("--no_rendering", action='store_true', help="disable rendering")
	parser.add_argument("--weather", default='WetCloudySunset', help="set weather preset, use --list to see available presets")

	# Deadline sweep settings
	parser.add_argument("--start_iter", type=int, default=None, help="override the minimal deadline value")
	parser.add_argument("--stepsize", type=int, default=10, help="deadline increments in ms")
	parser.add_argument("--multiplier", type=int, default=10, help="maximum deadline sweep value")

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

	params = vars(parser.parse_args())

	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	tf.compat.v1.reset_default_graph()

	# sweep over architectures
	if params['arch'] == 'all':
		architectures=['ResNet18', 'ResNet50']
	else:
		architectures=[params['arch']]

	# sweep over offloading policies
	if params['offload_policy'] == 'all':
		policies=['local', 'offload', 'offload_failsafe', 'adaptive', 'adaptive_failsafe']
	else:
		policies=[params['offload_policy']]

	# sweep over offloading positions
	if params['offload_position'] == 'all':
		positions=['direct', '0.5_direct', '0.25_direct', 'bottleneck']
	else:
		positions=[params['offload_position']]

	# sweep over image resolutions
	if params['img_resolution'] == 'all':
		resolutions=['Radiate', '480p', '720p']
	else:
		resolutions=[params['img_resolution']]

	for resolution in resolutions:
		print('-'*80)
		params['img_resolution'] = resolution
		for architecture in architectures:
			print('-'*80)
			params['arch'] = architecture 
			for policy in policies:
				print('-'*80)
				params['offload_policy'] = policy 
				for position in positions: 					
					print('-'*80)
					params['offload_position'] = position
					if (params['offload_position'] in ['0.5_direct', '0.25_direct']) and (params['offload_policy'] == 'local'):
						continue 			# local will not change
					params['arch'] = params['arch'][:8]     # prefix string (e.g., ResNet18)
					if 'mimic' not in params['arch'] and 'bottleneck' in params["offload_position"]:
						params['arch'] = params['arch'] + '_mimic'
					print('resolution', params['img_resolution'], '\tarch:', params['arch'], '\tposition:',params['offload_position'], '\tpolicy:', params['offload_policy'])
					# sweep over deadlines
					lower_bound = full_local_latency[params["img_resolution"]][params["HW"]][params["arch"]]
					max_deadline_value = det_max_deadline(lower_bound, params['stepsize'], params['multiplier'])
					if params['start_iter'] is None:
						deadline = lower_bound
					else:
						deadline = params['start_iter']
						params['start_iter'] = None 		# start fresh next combination
					assert deadline <= max_deadline_value and deadline >= lower_bound
					params["deadline"] = deadline
					p = Process(target=train, args=(params, True, False,)) 				# Using this format to kill a client's TCP connection once finished
					# train(params, True, False) 
					p.start()
					p.join()
					if p.is_alive():
						p.kill()	
					# print(params["deadline"])
					while params["deadline"] <= max_deadline_value:
						params["deadline"] = round(params["deadline"]) + 1     # To get readings at multiples of common numbers as 5,10,etc
						if params["deadline"] % params["stepsize"] == 0:
							tf.compat.v1.reset_default_graph()
							p = Process(target=train, args=(params, True, False,))
							# train(params, True, False) 
							p.start()
							p.join()
							if p.is_alive():
								p.kill()		
							# print(params['deadline'])
