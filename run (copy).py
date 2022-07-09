import os
import random
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt

from vae_common import create_encode_state_fn, load_vae
from ppo import PPO
from reward_functions import reward_functions
# from run_eval import run_eval
from utils import VideoRecorder, compute_gae, plot_trajectories
from vae.models import ConvVAE, MlpVAE

from CarlaEnv.carla_offload_env import CarlaOffloadEnv as CarlaEnv
from CarlaEnv.agents.navigation import basic_agent, behavior_agent


def train(params, start_carla=True, restart=False):
    # Read parameters
    learning_rate               = params["learning_rate"]
    lr_decay                    = params["lr_decay"]
    discount_factor             = params["discount_factor"]
    gae_lambda                  = params["gae_lambda"]
    ppo_epsilon                 = params["ppo_epsilon"]
    initial_std                 = params["initial_std"]
    value_scale                 = params["value_scale"]
    entropy_scale               = params["entropy_scale"]
    horizon                     = params["horizon"]
    num_epochs                  = params["num_epochs"]
    num_episodes                = params["num_episodes"]
    batch_size                  = params["batch_size"]
    vae_model                   = params["vae_model"]
    vae_model_type              = params["vae_model_type"]
    vae_z_dim                   = params["vae_z_dim"]
    synchronous                 = params["synchronous"]
    fps                         = params["fps"]
    action_smoothing            = params["action_smoothing"]
    model_name                  = params["model_name"]
    reward_fn                   = params["reward_fn"]
    seed                        = params["seed"]
    eval_interval               = params["eval_interval"]
    record_eval                 = params["record_eval"]
    safety_filter               = params["safety_filter"]
    obstacle                    = params["obstacle"]
    penalize_steer_diff         = params["penalize_steer_diff"]
    penalize_dist_obstacle      = params["penalize_dist_obstacle"]
    test                        = params["test"]
    gaussian                    = params["gaussian"]
    track                       = params["track"]
    follow_waypoints            = params["follow_waypoints"]


    if 'mimic' not in params['arch'] and params["offload_position"] == 'bottleneck':
        if params['arch'] == 'ResNet18':
            params['arch'] = 'ResNet18_mimic'
        elif params['arch'] == 'ResNet50':
            params['arch'] = 'ResNet50_mimic'
        elif params['arch'] == 'DenseNet169':
            params['arch'] = 'DenseNet169_mimic'
        else:
            raise ValueError("No mimic architecture supported for the selected architecture!")


    # Set seeds
    if isinstance(seed, int):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Load VAE
    vae = load_vae(vae_model, vae_z_dim, vae_model_type) # OD: load pretrained VAE that generates latent vecotrs from the RGB images to train the policy network  
    
    # Override params for logging
    params["vae_z_dim"] = vae.z_dim
    params["vae_model_type"] = "mlp" if isinstance(vae, MlpVAE) else "cnn"

    print("")
    print("Training parameters:")
    for k, v, in params.items(): print('  {} , {}'.format(k,v))
    print("")

    # Create state encoding fn
    measurements_to_include = set(["steer", "throttle", "speed", "xi", "r"])        # OD: remaining state observations; vehicle inertial measurements and the last two terms representing relative angle and distance between vehicle and nearest obstacle
    encode_state_fn = create_encode_state_fn(vae, measurements_to_include)

    # Create env
    print("Creating environment")
    env = CarlaEnv(obs_res=(160, 80),
                   action_smoothing=action_smoothing,
                   encode_state_fn=encode_state_fn,
                   reward_fn=reward_functions[reward_fn],
                   synchronous=synchronous,
                   fps=fps,
                   start_carla=start_carla,
                   apply_filter=safety_filter,
                   obstacle=obstacle,
                   penalize_steer_diff=penalize_steer_diff,
                   penalize_dist_obstacle=penalize_dist_obstacle,
                   gaussian=gaussian,
                   track=track,
                   params=params)
    if isinstance(seed, int):
        env.seed(seed)
    best_eval_reward = -float("inf")

    # Environment constants
    input_shape = np.array([vae.z_dim + len(measurements_to_include)])
    num_actions = env.action_space.shape[0]

    subdirs_path = os.path.join("models", model_name, "experiments", params['img_resolution'], params['arch'], params['offload_policy']+"_"+params['HW']+"_"+str(params['deadline']))

    # Create model
    print("Creating model")
    model = basic_agent()

    # For every episode
    train_data_path = os.path.join(subdirs_path, "train_data.csv")
    valid_data_path = os.path.join(subdirs_path, "valid_data.csv")

    initial_row = ['episode_idx', 'reward','obstacle_hit', 'curb_hit', 'dist_traveled', 'dist_to_obstacle', 'avg_speed', 
                    'avg_latency', 'avg_energy', 'missed_deadlines', 'max_succ_interrupts', 'missed_offloads', 'misguided_energy']
    if not os.path.exists(train_data_path):
        with open(train_data_path, 'a', newline='') as fd:
            csv_writer = csv.writer(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(initial_row)

    if not os.path.exists(valid_data_path):
        with open(valid_data_path, 'a', newline='') as fd:
            csv_writer = csv.writer(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(initial_row)
    while num_episodes <= 0 or model.get_episode_idx() < num_episodes:
        try:
            episode_idx = model.get_episode_idx()
            # Run evaluation periodically

            if seed > 0:
                env_seed = seed
            else:
                env_seed = episode_idx
            state, terminal_state, total_reward = env.reset(is_training=not test, seed=env_seed), False, 0  # 'state' constitues multipe components
            # print(state) # state is the 64-dim encoder output + 5 measurements to include
            if record_eval:
                rendered_frame = env.render(mode="rgb_array")
                video_filename = os.path.join(model.video_dir, "episode{}.avi".format(episode_idx))
                # Init video recording
                print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape,
                                                                      int(env.average_fps)))
                video_recorder = VideoRecorder(video_filename,
                                               frame_size=rendered_frame.shape,
                                               fps=env.average_fps)
                video_recorder.add_frame(rendered_frame)
            else:
                video_recorder = None
            # Reset environment
            ego_x, ego_y, obstacle_x, obstacle_y, xi, r, rl_steer, rl_throttle, filter_steer, filter_throttle, sim_time, filter_applied, action_none= [], [], [], [], [], [], [], [], [], [], [], [], []
            probe_tu, exp_tu, selected_action, correct_action, probe_latency, probe_energy, actual_latency, actual_energy, exp_latency, exp_energy, miss_flag = [], [], [], [], [], [], [], [], [], [], []

            # While episode not done
            print("Episode {} (Step {})".format(episode_idx,model.get_train_step_idx()))
            route, current_waypoint_index = None, None
            while not terminal_state:
                states, taken_actions, values, rewards, dones = [], [], [], [], []
                for _ in range(horizon):               # number of steps to simulate per training step (128)
                    if follow_waypoints:
                        action, value = 'auto', 0
                    else: 
                        action, value = model.predict(state, write_to_summary=True)
                    # Perform action
                    new_state, reward, terminal_state, info, offloading_info = env.step(action)
                    if info["closed"] == True:
                        exit(0)

                    env.extra_info.extend([
                        "Episode {}".format(episode_idx),
                        # "Training...",
                        # "",
                        "Value:  % 20.2f" % value
                    ])


                    if video_recorder is not None:
                        rendered_frame = env.render(mode="rgb_array")
                        video_recorder.add_frame(rendered_frame)
                    else:
                        env.render()
                    total_reward += reward

                    # For training the ppo (irrelevant)
                    states.append(state)         # [T, *input_shape]
                    taken_actions.append(action) # [T,  num_actions]
                    values.append(value)         # [T]
                    rewards.append(reward)       # [T]
                    dones.append(terminal_state) # [T]

                    # environment measurements
                    xi.append(round(info["xi"], 3))
                    r.append(round(info["r"], 3))
                    filter_steer.append(round(info["filter_steer"], 3))
                    filter_throttle.append(round(info["filter_throttle"], 3))
                    rl_throttle.append(round(info["rl_throttle"], 3))
                    rl_steer.append(round(info["rl_steer"], 3))
                    sim_time.append(round(info["sim_time"], 3))
                    filter_applied.append(round(info["filter_applied"], 3))
                    action_none.append(round(info["action_none"], 3))
                    ego_x.append(round(info["ego_x"], 3))
                    ego_y.append(round(info["ego_y"], 3))
                    if info["obstacle_x"] is not None:
                        obstacle_x.append(round(info["obstacle_x"], 3))
                    if info["obstacle_y"] is not None:
                        obstacle_y.append(round(info["obstacle_y"], 3))
                    current_waypoint_index = info["current_waypoint_index"]
                    state = new_state

                    # offloading measurements
                    probe_tu.append(round(offloading_info["probe_tu"], 3))
                    exp_tu.append(round(offloading_info["probe_tu"] + offloading_info["delta_tu"], 3))
                    selected_action.append(offloading_info["selected_action"])
                    correct_action.append(offloading_info["correct_action"])
                    try:
                        probe_latency.append(round(offloading_info["probe_latency"], 3))
                        probe_energy.append(round(offloading_info["probe_energy"], 3))
                    except TypeError:
                        probe_latency.append(offloading_info["probe_latency"])
                        probe_energy.append(offloading_info["probe_energy"])
                    actual_latency.append(round(offloading_info["actual_latency"], 3))
                    actual_energy.append(round(offloading_info["actual_energy"], 3))
                    exp_latency.append(round(offloading_info["exp_latency"], 3))
                    exp_energy.append(round(offloading_info["exp_energy"], 3))
                    miss_flag.append(offloading_info["missed_deadline_flag"])

                    if terminal_state:
                        route = info["route"]
                        break

                # Calculate last value (bootstrap value)
                _, last_values = model.predict(state) # []

                # Compute GAE -- Generalized Advantage Estimator
                advantages = compute_gae(rewards, values, last_values, dones, discount_factor, gae_lambda)
                returns = advantages + values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Flatten arrays
                states        = np.array(states)
                taken_actions = np.array(taken_actions)
                returns       = np.array(returns)
                advantages    = np.array(advantages)

                T = len(rewards)
                assert states.shape == (T, *input_shape)
                assert taken_actions.shape == (T, num_actions)
                assert returns.shape == (T,)
                assert advantages.shape == (T,)

                # Train for some number of epochs
                if not test:
                    model.update_old_policy() # θ_old <- θ
                for _ in range(num_epochs):
                    num_samples = len(states)
                    indices = np.arange(num_samples)
                    np.random.shuffle(indices)
                    for i in range(int(np.ceil(num_samples / batch_size))):
                        # Sample mini-batch randomly
                        begin = i * batch_size
                        end   = begin + batch_size
                        if end > num_samples:
                            end = None
                        mb_idx = indices[begin:end]

                        # Optimize network
                        if not test:
                            #print("train")
                            model.train(states[mb_idx], taken_actions[mb_idx],
                                    returns[mb_idx], advantages[mb_idx])

            # The direct waypoints route coordinates for plotting reference 
            completed_route = route[:current_waypoint_index]
            completed_x = [x.transform.location.x for (x,_) in completed_route]
            completed_y = [x.transform.location.y for (x,_) in completed_route]

            plot_trajectories(ego_x, ego_y, completed_x, completed_y, obstacle_x, obstacle_y, model.plot_dir, episode_idx)

            df = pd.DataFrame({'sim_time': pd.Series(sim_time), 'r':pd.Series(r), 'xi':pd.Series(xi), 'rl_throttle':pd.Series(rl_throttle), 'rl_steer':pd.Series(rl_steer), 
                                'miss_flag': pd.Series(miss_flag), 'probe_tu': pd.Series(probe_tu), 'exp_tu': pd.Series(exp_tu), 'selected_action': pd.Series(selected_action), 'correct_action': pd.Series(correct_action),
                                'probe_latency': pd.Series(probe_latency), 'exp_latency': pd.Series(exp_latency), 'probe_energy': pd.Series(probe_energy), 'exp_energy': pd.Series(exp_energy)})
            df.to_csv(model.plot_dir + '/train_' + str(episode_idx) + '.csv')
            # Write episodic values
            if video_recorder is not None:
                video_recorder.release()
            if (env.distance_traveled > 0.0):
                data_row =  [episode_idx, round(env.total_reward1,3), env.obstacle_hit, env.curb_hit, round(env.distance_traveled,3), round(env.min_distance_to_obstacle, 3), round(3.6 * env.speed_accum / env.step_count, 3),
                            np.mean(exp_latency), np.mean(exp_energy), env.energy_monitor.missed_deadlines, env.energy_monitor.max_succ_interrupts, env.energy_monitor.missed_offloads, env.energy_monitor.misguided_energy]
                if test:
                    with open(valid_data_path, 'a', newline='') as fd:
                        csv_writer = csv.writer(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(data_row)
                else:
                    with open(train_data_path, 'a', newline='') as fd:
                        csv_writer = csv.writer(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(data_row)
            model.write_episodic_summaries()

            if not test:
                model.save()
        except KeyboardInterrupt:
            env.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trains a CARLA agent with PPO")

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
    parser.add_argument("--num_episodes", type=int, default=0, help="Number of episodes to train for (0 or less trains forever)")

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
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to train. Output written to models/model_name")
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
    parser.add_argument("-follow_waypoints", action="store_true",
                        help="following waypoints rather than the RL agent")

    # AV pipeline Offloading
    parser.add_argument("--arch", type=str, help="Name of the model running on the AV platform", choices=['ResNet18', 'ResNet50', 'DenseNet169', 'ViT', 'ResNet18_mimic', 'ResNet50_mimic', 'DenseNet169_mimic'], default='ResNet50')
    parser.add_argument("--offload_position", type=str, help="Offloading position", choices=['direct', '0.5_direct', '0.25_direct', 'bottleneck'], default='direct')
    parser.add_argument("--offload_policy", type=str, help="Offloading policy", choices=['local', 'offload', 'offload_failsafe', 'adaptive', 'adaptive_failsafe'], default='direct')    
    parser.add_argument("--bottleneck_ch", type=int, help="number of bottleneck channels", choices=[3,6,9,12], default=6)
    parser.add_argument("--bottleneck_quant", type=int, help="quantization of the output", choices=[8,16,32], default=8)
    parser.add_argument("--HW", type=str, help="AV Hardware", choices=['PX2', 'TX2', 'Orin', 'Xavier', 'Nano'], default='PX2')
    parser.add_argument("--deadline", type=int, help="time window", default=100)
    parser.add_argument("--img_resolution", type=str, help="enter offloaded image resolution", choices=['480p', '720p', '1080p', 'Radiate', 'TeslaFSD', 'Waymo'], default='720p')
    parser.add_argument("--comm_tech", type=str, help="the wireless technology", choices=['LTE', 'WiFi', '5G'], default='LTE')
    parser.add_argument("--conn_overhead", action="store_true", default=False, help="Account for the connection establishment overhead separately alongside data transfer")
    parser.add_argument("--rayleigh_sigma", type=int, help="Scale of the throughput's Rayleigh distribution -- default is the value from collected LTE traces", default=13.62)    
    parser.add_argument("--noise_scale", type=float, default=5, help="noise scale/variance")

    # Metrics to record (energy, missed deadlines, longest accumulated latency (99th percentile?))
    # Maybe even do with sudden and no interrupts?
    # shall I do like 50-100 episodes initially, and then average over the entire thingy?

    params = vars(parser.parse_args())

    # Remove a couple of parameters that we dont want to log
    start_carla = params["start_carla"]; del params["start_carla"]
    restart = params["restart"]; del params["restart"]

    tf.compat.v1.reset_default_graph()

    # Start training
    train(params, start_carla, restart)