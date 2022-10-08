# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# The original working code with cascaded autoencoder, a standard object detector 
# both providing inputs to the controller. All implementations are in tensorflow.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import random
import time
import shutil
import pandas as pd
import numpy as np

import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import gym

from vae_common import create_encode_state_fn, load_vae
from ppo import PPO
from ppo_cascade import PPO_CASCADE
from reward_functions import reward_functions
from utils import VideoRecorder, compute_gae, plot_trajectories, plot_energy_stats, dir_manager
from vae.models import ConvVAE, MlpVAE
from wrappers import SafetyFilter

# Object Detection functionalities
from detector import Detector, mask_detection_boxes
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder

from CarlaEnv.carla_offload_env import CarlaOffloadEnv as CarlaEnv
# from CarlaEnv.agents.navigation import basic_agent, behavior_agent

parent_directory = os.path.abspath(os.path.join(__file__,os.pardir + "/" + os.pardir))
PATH_TO_LABELS = parent_directory + '/TensorFlow/models/research/object_detection/data/mscoco_label_map.pbtxt'

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
    deadline                    = params["deadline"]

    if 'mimic' not in params['arch'] and 'bottleneck' in params["offload_position"]:
        if params['arch'] == 'ResNet18':
            params['arch'] = 'ResNet18_mimic'
        elif params['arch'] == 'ResNet50':
            params['arch'] = 'ResNet50_mimic'
        elif params['arch'] == 'DenseNet169':
            params['arch'] = 'DenseNet169_mimic'
        else:
            raise ValueError("No mimic architecture supported for the selected architecture!")

    if params['observation_res'] == '80':
        obs_res = (160,80)                      # order for CarlaEnv and VAE, reverse order for detector
    elif params['observation_res'] == '480':
        obs_res = (852,480)
    else:
        raise ValueError("{} is Unidentified resolution for observation frame.".format(params['observation_dims']))

    # For naming purposes
    if params['offload_policy'] == 'local' and params['local_belay']:       
        sup_string = 'belay'
    elif params['offload_policy'] == 'local' and params['local_early']:
        sup_string = 'early'
    elif params['offload_policy'] == 'local':
        sup_string = 'cont'
    elif params['off_belay']:                # for offloading policies
        sup_string = 'belay'
    else:
        sup_string = 'early'

    subdirs_path = os.path.join("models", model_name, "experiments", "obs_"+str(params['len_obs'])+"_route_"+str(params['len_route']), params['img_resolution'] +"_"+ params['arch'] +"_"+ params['offload_policy'] +"_"+ sup_string, params['HW']+"_"+str(params['deadline'])+"_Safety_"+str(params['safety_filter'])+"_noise_"+str(params['gaussian']))

    # Set seeds
    if isinstance(seed, int):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Load VAE
    vae = load_vae(vae_model, vae_z_dim, vae_model_type) # load pretrained VAE that generates latent vecotrs from the RGB images to train the policy network  
    # Override params for logging
    params["vae_z_dim"] = vae.z_dim
    params["vae_model_type"] = "mlp" if isinstance(vae, MlpVAE) else "cnn"

    # Create state encoding fn
    measurements_to_include = set(["steer", "throttle", "speed", "xi", "r"])        # OD: remaining state observations; vehicle inertial measurements and the last two terms representing relative angle and distance between vehicle and nearest obstacle
    encode_state_fn = create_encode_state_fn(vae, measurements_to_include)

    # Create env
    print("Creating environment")
    env = CarlaEnv(obs_res=obs_res,       # 160*80
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
                   model_name=model_name,
                   params=params)

    object_detector = Detector(input_shape=obs_res[::-1], model='FasterRCNN_ResNet152')
    bbox_shape = np.array([10*4])  # detections output size from the object detector (originally 300*4 but we max 5*4 to keep small network size)
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)   

    # print(category_index.items())

    if params['carla_map'] is not None: 
        params['carla_map'] = None          # To only load the map once
    if params['weather'] is not None:
        params['weather'] = None

    if isinstance(seed, int):
        env.seed(seed)
    best_eval_reward = -float("inf")

    # Environment constants
    input_shape = np.array([vae.z_dim + len(measurements_to_include)]) # [69] for the PPO
    num_actions = env.action_space.shape[0]         # steer and throttle if PPO

    # Create model     
    if model_name.startswith('agent'): 
        print("Creating model")
        model = PPO(input_shape, env.action_space, 
                    learning_rate=learning_rate, lr_decay=lr_decay,
                    epsilon=ppo_epsilon, initial_std=initial_std,
                    value_scale=value_scale, entropy_scale=entropy_scale,
                    model_dir=os.path.join("models", model_name), 
                    subdirs=subdirs_path)
    elif model_name.startswith('casc_agent'):
        print("Creating cascaded model")
        model = PPO_CASCADE(input_shape, bbox_shape, env.action_space,
                    learning_rate=learning_rate, lr_decay=lr_decay,
                    epsilon=ppo_epsilon, initial_std=initial_std,
                    value_scale=value_scale, entropy_scale=entropy_scale,
                    model_dir=os.path.join("models", model_name), 
                    subdirs=subdirs_path)
    else:
        print("Direct automated control following route")
        model = dir_manager(model_dir=os.path.join("models", model_name), subdirs=subdirs_path)

    if safety_filter:
        env.safety_filter = SafetyFilter()
    
    if restart:
        shutil.rmtree(model.model_dir)
        for d in model.dirs:
            os.makedirs(d)
    if model_name.startswith('agent') or model_name.startswith('casc_agent'):
        model.init_session()
        if not restart:
            model.load_latest_checkpoint()
        model.write_dict_to_summary("hyperparameters", params, 0)
    object_detector.init_session()

    # For every episode
    train_data_path = os.path.join(subdirs_path, "train_data.csv")
    valid_data_path = os.path.join(subdirs_path, "valid_data.csv")

    initial_row = ['episode_idx', 'reward','obstacle_hit', 'curb_hit', 'dist_traveled', 'dist_to_obstacle', 'avg_speed', 'avg_center_deviance',
                    'avg_latency', 'avg_energy', 'missed_windows', 'missed_deadlines', 'missed_controls', 'max_succ_interrupts', 'missed_offloads', 'misguided_energy']
    if not os.path.exists(train_data_path):
        with open(train_data_path, 'a', newline='') as fd:
            csv_writer = csv.writer(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(initial_row)

    final_episode_iteration = model.get_episode_idx() + num_episodes

    if not os.path.exists(valid_data_path):
        with open(valid_data_path, 'a', newline='') as fd:
            csv_writer = csv.writer(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(initial_row)
    while num_episodes <= 0 or model.get_episode_idx() < final_episode_iteration:
        try:
            start = time.time()
            episode_idx = model.get_episode_idx()
            if seed > 0:
                env_seed = seed
            else:
                env_seed = episode_idx
            state, terminal_state, total_reward = env.reset(is_training=not test, seed=env_seed), False, 0   
            # state is the 64-dim encoder output + 5 measurements to include
            if record_eval and env.display is not None:
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

            recorder_path = parent_directory + "/EnergyShield/" + model.log_dir + str(episode_idx) + '_log.log'

            env.client.start_recorder(recorder_path, True)

            # Reset environment
            ego_x, ego_y, obstacle_x, obstacle_y, xi, r, rl_steer, rl_throttle, filter_steer, filter_throttle, sim_time, filter_applied, action_none, total_rewards = [], [], [], [], [], [], [], [], [], [], [], [], [], []
            delta_T, phi_est, rtt_est, que_est, phi_true, rtt_true, que_true, miss_flag, transit_window, rx_flag, transit_flag, recover_flag, belaying = [], [], [], [], [], [], [], [], [], [], [], [], []
            selected_action, correct_action, est_latency, est_energy, true_latency, true_energy, exp_latency, exp_energy = [], [], [], [], [], [], [], []

            # first detections based on new frame
            observation_input = env.observation[np.newaxis,:,:,:].astype(np.uint8)
            detections = object_detector.detect(observation_input)      
            masked_detections = mask_detection_boxes(detections, min_score_threshold=0.5)
            masked_detections = masked_detections.flatten()[:10*4]

            # While episode not done
            try:
                print("Episode {} (Step {})".format(episode_idx,model.get_train_step_idx()))
            except AttributeError:
                print("Episode {}".format(episode_idx))
            route, current_waypoint_index = None, None

            plt.figure()
            while not terminal_state:
                states, input_boxes, taken_actions, values, rewards, dones = [], [], [], [], [], []
                for _ in range(horizon):               # number of steps to simulate per training step (128)
                    # flatten for predict and train

                    # dict_keys(['detection_anchor_indices', 'raw_detection_scores', 'raw_detection_boxes', 'detection_scores', 
                    # 'detection_classes', 'detection_multiclass_scores', 'num_detections', 'detection_boxes'])

                    # *achor_indices.shape: (1,300), raw_detection_scores.shape: (1,300,91), detection_scores: (1,300) *boxes.shape: (1,300,4), *detections.shape: (1,)
                    # print(detections['detection_anchor_indices'].shape)
                    # print(detections['raw_detection_scores'].shape)
                    # print(detections['detection_boxes'].shape)
                    # print(detections['num_detections'])

                    # print('indices', detections['detection_anchor_indices'])        # anchor indices are to indicate which anchor was used for the correspoiding RPI (i.e. the center of the bounding box)
                    # print('BBOX ', detections['detection_boxes'])                   # This is not global detection box
                    # print('scroes', np.max(detections['detection_scores']))         # how many classes are active

                    # print(detections['detection_boxes'][0][0])

                    # first control output predictions
                    if model_name.startswith('agent'):
                        action, value = model.predict(state, write_to_summary=True)
                    elif model_name.startswith('casc_agent'):
                        action, value = model.predict(state, masked_detections, write_to_summary=True)
                    else:
                        action, value = np.array([0, 0]), 0

                    image_np_with_detections = env.observation.copy()

                    # if env.offloading_manager.rx_flag is True:
                    viz_utils.visualize_boxes_and_labels_on_image_array(
                            image_np_with_detections,
                            np.squeeze(detections['detection_boxes']),
                            np.squeeze(detections['detection_classes']),
                            np.squeeze(detections['detection_scores']),
                            category_index,
                            use_normalized_coordinates=True,
                            max_boxes_to_draw=200,
                            min_score_thresh=.50,
                            agnostic_mode=False)

                    plt.imshow(image_np_with_detections)

                    # TODO: Implement an if condition that saves an image every now and then when detecting an object
                    # plt.savefig('/home/mohanadodema/carla.png')

                    # Perform action
                    new_state, reward, terminal_state, info, offloading_info = env.step(action)
                    if info["closed"] == True:
                        exit(0)
                    env.extra_info.extend([
                        "Episode {}".format(episode_idx),
                    ])
 
                    if env.display is not None:
                        if video_recorder is not None:
                            rendered_frame = env.render(mode="rgb_array")
                            video_recorder.add_frame(rendered_frame)
                        else:
                            env.render()
                    total_reward += reward

                    # new dashcam observation
                    if env.offloading_manager.rx_flag is True:  # if processing successful
                        observation_input = env.observation[np.newaxis,:,:,:].astype(np.uint8)
                        detections = object_detector.detect(observation_input)      
                        new_masked_detections = mask_detection_boxes(detections, min_score_threshold=0.5)
                        new_masked_detections = new_masked_detections.flatten()[:10*4]

                    # For training the ppo 
                    states.append(state)         # [T, *input_shape]
                    input_boxes.append(masked_detections)   # [T, 10*4]
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
                    total_rewards.append(round(reward, 3))
                    if info["obstacle_x"] is not None:
                        obstacle_x.append(round(info["obstacle_x"], 3))
                    if info["obstacle_y"] is not None:
                        obstacle_y.append(round(info["obstacle_y"], 3))
                    current_waypoint_index = info["current_waypoint_index"]
                    state = new_state
                    masked_detections = new_masked_detections

                    # offloading measurements
                    phi_est.append(round(offloading_info["phi_est"], 3))
                    rtt_est.append(round(offloading_info["rtt_est"], 3))
                    que_est.append(round(offloading_info["que_est"], 3))
                    phi_true.append(round(offloading_info["phi_true"], 3))
                    rtt_true.append(round(offloading_info["rtt_true"], 3))
                    que_true.append(round(offloading_info["que_true"], 3))
                    selected_action.append(offloading_info["selected_action"])
                    correct_action.append(offloading_info["correct_action"])
                    try:
                        est_latency.append(round(offloading_info["est_latency"], 3))
                        est_energy.append(round(offloading_info["est_energy"], 3))
                    except TypeError:
                        est_latency.append(offloading_info["est_latency"])
                        est_energy.append(offloading_info["est_energy"])
                    true_latency.append(round(offloading_info["true_latency"], 3))
                    true_energy.append(round(offloading_info["true_energy"], 3))
                    exp_latency.append(round(offloading_info["exp_latency"], 3))
                    exp_energy.append(round(offloading_info["exp_energy"], 3))
                    delta_T.append(offloading_info["delta_T"])
                    miss_flag.append(offloading_info["missed_window_flag"])
                    rx_flag.append(offloading_info["rx_flag"])
                    transit_flag.append(offloading_info["transit_flag"])
                    recover_flag.append(offloading_info["recover_flag"])
                    transit_window.append(offloading_info["transit_window"])
                    belaying.append(offloading_info["belaying"])

                    if terminal_state:
                        route = info["route"]
                        break
                # Calculate last value (bootstrap value)
                if model_name.startswith('agent'):
                    _, last_values = model.predict(state) # []
                elif model_name.startswith('casc_agent'):
                    _, last_values = model.predict(state, masked_detections)
                else:
                    last_values = 1.0            # dummy values
                
                # Compute GAE -- Generalized Advantage Estimator
                advantages = compute_gae(rewards, values, last_values, dones, discount_factor, gae_lambda)
                returns = advantages + values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Flatten arrays
                states        = np.array(states)
                input_boxes   = np.array(input_boxes)
                taken_actions = np.array(taken_actions)
                returns       = np.array(returns)
                advantages    = np.array(advantages)

                T = len(rewards)
                assert states.shape == (T, *input_shape)
                assert input_boxes.shape == (T, 10*4)
                assert taken_actions.shape == (T, num_actions)
                assert returns.shape == (T,)
                assert advantages.shape == (T,)

                # Train for some number of epochs
                if (not test) and ((model_name.startswith('agent')) or model_name.startswith('casc_agent')):
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
                        if (not test) and (model_name.startswith('agent')):
                            #print("train")
                            model.train(states[mb_idx], taken_actions[mb_idx],
                                    returns[mb_idx], advantages[mb_idx])
                        elif (not test) and (model_name.startswith('casc_agent')):
                            model.train(states[mb_idx], input_boxes[mb_idx], taken_actions[mb_idx],
                                    returns[mb_idx], advantages[mb_idx])

            env.client.stop_recorder()

            print("Episode took {:.2f} seconds".format(time.time()-start))

            # The direct waypoints route coordinates for plotting reference 
            completed_route = route[:current_waypoint_index]
            completed_x = [x.transform.location.x for (x,_) in completed_route]
            completed_y = [x.transform.location.y for (x,_) in completed_route]

            if not params["no_save"]:

                plot_trajectories(ego_x, ego_y, completed_x, completed_y, obstacle_x, obstacle_y, model.plot_dir, episode_idx)
                plot_energy_stats(exp_latency, exp_energy, phi_true, miss_flag, model.plot_dir, episode_idx, params)

                df = pd.DataFrame({'sim_time': pd.Series(sim_time), 'r':pd.Series(r), 'xi':pd.Series(xi), 'ego_x': pd.Series(ego_x), 'ego_y': pd.Series(ego_y), 'rl_throttle':pd.Series(rl_throttle), 'rl_steer':pd.Series(rl_steer), 'rewards': pd.Series(total_rewards),
                                    '': "", 'delta_T': pd.Series(delta_T), 'transit_window': pd.Series(transit_window), 'rx_flag': pd.Series(rx_flag), 'miss_flag': pd.Series(miss_flag), 'transit_flag': pd.Series(transit_flag), 'recover_flag': pd.Series(recover_flag), 'belaying': pd.Series(belaying), 
                                    '': "", 'selected_action': pd.Series(selected_action), 'correct_action': pd.Series(correct_action), 'est_latency': pd.Series(est_latency), 'exp_latency': pd.Series(exp_latency), 'est_energy': pd.Series(est_energy), 'exp_energy': pd.Series(exp_energy),
                                    '': "", 'phi_est': pd.Series(phi_est), 'rtt_est': pd.Series(rtt_est), 'que_est': pd.Series(que_est), 'phi_true': pd.Series(phi_true), 'rtt_true': pd.Series(rtt_true), 'que_true': pd.Series(que_true)})
                df.to_csv(model.plot_dir + '/train_' + str(episode_idx) + '.csv')
                # Write episodic values
                if video_recorder is not None:
                    video_recorder.release()
                if (env.distance_traveled > 0.0):
                    data_row = [episode_idx, round(env.total_reward1,3), env.obstacle_hit, env.curb_hit, round(env.distance_traveled,3), round(env.min_distance_to_obstacle, 3), round(3.6 * env.speed_accum / env.step_count, 3), round(env.center_lane_deviation/env.step_count, 3), 
                                round(np.mean(exp_latency),3), round(np.mean(exp_energy),3), env.offloading_manager.missed_windows, env.offloading_manager.missed_deadlines, env.missed_controls, env.offloading_manager.max_succ_interrupts, env.offloading_manager.missed_offloads, env.offloading_manager.misguided_energy]
                    if test:
                        with open(valid_data_path, 'a', newline='') as fd:
                            csv_writer = csv.writer(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow(data_row)
                    else:
                        with open(train_data_path, 'a', newline='') as fd:
                            csv_writer = csv.writer(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow(data_row)
            if model_name.startswith('agent') or model_name.startswith('casc_agent'):
                model.write_episodic_summaries()    #TODO: remove the write tensorflow events
            else:
                model.inc_count()

            if (not test) and ((model_name.startswith('agent')) or model_name.startswith('casc_agent')):
                model.save()

        except KeyboardInterrupt:
            env.close()

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
    parser.add_argument("--time_window", type=int, default=20, help="The discretized time window duration in ms")
    parser.add_argument("--spawn_random", action="store_true", help="Spawn ego vehicle at random position and orientation")

    # Training parameters
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to train. Output written to models/model_name") #choices=['agent1', 'agent2', 'agent3', 'agent4', 'casc_agent1', 'casc_agent2', 'casc_agent3', 'casc_agent4', 'BasicAgent', 'BehaviorAgent'])
    parser.add_argument("--reward_fn", type=str, default="reward_speed_centering_angle_multiply", help="Reward function to usfe. See reward_functions.py for more info.")
    parser.add_argument("--seed", type=int, default=0, help="Seed to use. (Note that determinism unfortunately appears to not be garuanteed with this option in our experience)")
    parser.add_argument("--eval_interval", type=int, default=100, help="Number of episodes between evaluation runs")
    parser.add_argument("-record_eval", action="store_true", default=False, help="If True, save videos of evaluation episodes to models/model_name/videos/")

    # Safety Filter Setting
    parser.add_argument("-safety_filter", action="store_true", default=False, help="Filter Control actions")
    parser.add_argument("-penalize_steer_diff", action="store_true", default=False, help="Penalize RL's steering output and filter's output")
    parser.add_argument("-penalize_dist_obstacle", action="store_true", default=False, help="Add a penalty when the RL agent gets closer to an obstacle")
    parser.add_argument("-obstacle", action="store_true", default=False, help="Add obstacles")
    parser.add_argument("-gaussian", action="store_true", default=False, help="Randomize obstacles location using gaussian distribution")
    parser.add_argument("--track", type=int, default=1, help="Track Number")
    parser.add_argument("-test", action="store_true", default=False, help="test")

    parser.add_argument("-restart", action="store_true",
                        help="If True, delete existing model in models/model_name before starting training")

    # AV pipeline Offloading
    # TODO: Write a description for the different offloading policies
    parser.add_argument("--arch", type=str, help="Name of the model running on the AV platform", choices=['ResNet18', 'ResNet50', 'ResNet152', 'DenseNet169', 'ViT', 'ResNet18_mimic', 'ResNet50_mimic', 'DenseNet169_mimic'], default='ResNet152')
    parser.add_argument("--offload_position", type=str, help="Offloading position", choices=['direct', '0.5_direct', '0.25_direct', '0.11_direct', 'bottleneck'], default='direct')
    parser.add_argument("--offload_policy", type=str, help="Offloading policy", choices=['local', 'offload', 'offload_failsafe', 'adaptive', 'adaptive_failsafe', 'Shield'], default='offload')    
    parser.add_argument("--bottleneck_ch", type=int, help="number of bottleneck channels", choices=[3,6,9,12], default=6)
    parser.add_argument("--bottleneck_quant", type=int, help="quantization of the output", choices=[8,16,32], default=8)
    parser.add_argument("--HW", type=str, help="AV Hardware", choices=['PX2', 'TX2', 'Orin', 'Xavier', 'Nano'], default='PX2')
    parser.add_argument("--deadline", type=int, help="dealdine in ms", default=20)                    # Single time window is 20 ms
    parser.add_argument("--img_resolution", type=str, help="enter offloaded image resolution", choices=['80p', '480p', '720p', '1080p', 'Radiate', 'TeslaFSD', 'Waymo'], default='80p')
    parser.add_argument("--comm_tech", type=str, help="the wireless technology", choices=['LTE', 'WiFi', '5G'], default='WiFi')
    parser.add_argument("--rayleigh_sigma", type=int, help="Scale of the throughput's Rayleigh distribution -- default is the value from collected LTE traces", default=20)#13.62)    
    parser.add_argument("--noise_scale", type=float, default=5, help="gaussian noise scale/variance")
    parser.add_argument("--off_belay", action='store_true', default=False, help="belay till delta_T expires to resume processing or execute local at the last attainable window" )        
    parser.add_argument("--local_belay", action='store_true', default=False, help="belay local execution until the last execution window of the dealdine")
    parser.add_argument("--local_early", action='store_true', default=False, help="instantly perform local execution at the first attainable window of the dealdine")
    parser.add_argument("--hold_det_img", default=True, help="hold detector image after transmission/belay initiated" )
    parser.add_argument("--hold_vae_img", default=False, help="hold VAE image after transmission/belay initiated")
    parser.add_argument("--cont_control", default=False, help="Controller keeps updating based on state variable even if the OD is offloaded (fixed detections from last frame)")

    # Netowrk Sampling and Estimation Parameters
    parser.add_argument("--buffer_size", type=int, default=5, help="moving average window size")
    parser.add_argument("--estimation_fn", type=str, default='avg', help="vehicle estimation function of the network conditions")
    parser.add_argument("--phi_scale", type=float, default=20, help="scale parameter for the channel capacity pdf")
    parser.add_argument("--phi_shift", type=float, default=0, help="shift parameter for the channel capacity pdf")
    parser.add_argument("--rtt_dist", type=str, default='gamma', help="use gamma or rayleigh pdf for rtt", choices=['rayleigh', 'gamma'])
    parser.add_argument("--rtt_shape", type=float, default=0, help="shape parameter for RTT pdf")   #1.25  # 3.5
    parser.add_argument("--rtt_scale", type=float, default=0, help="scale parameter for RTT pdf")   #2     # 5.5
    parser.add_argument("--rtt_shift", type=float, default=0, help="shift from zero for RTT pdf")   #2     # 10
    parser.add_argument("--qsize", type=int, default=4000 , help='queue size at the server')
    parser.add_argument("--arate", type=int, default=970 , help='arrival rate for queue size pdf')
    parser.add_argument("--srate", type=int, default=1000 , help='service rate for queue size pdf')

    # Carla Config file
    parser.add_argument("--carla_map", type=str, default='Town04_OPT', help="load map")
    parser.add_argument("--no_rendering", action='store_true', help="disable rendering")
    parser.add_argument("--weather", default='WetCloudySunset', help="set weather preset, use --list to see available presets")
    parser.add_argument("-display_off", action='store_true', help='Turn off display running experiments on the server')
    parser.add_argument("--port", type=int, default=2000, help='set the port for communicating with the host in case of server operation')

    # Additional Carla Offloading env options
    parser.add_argument("--len_route", type=str, default='short', help="The route array length -- longer routes support more obstacles but extends sim time")
    parser.add_argument("--len_obs", type=int, default=1, help="How many objects to be spawned given len_route is satisfied")
    parser.add_argument("--obs_step", type=int, default=0, help="Objects distribution along the route after the first one")
    parser.add_argument("--obs_start_idx", type=int, default=30, help="spawning index of first obstacle")
    parser.add_argument("--no_save", action='store_true', help="code experiment no save to disk")
    parser.add_argument("--observation_res", type=str, default='80', help="The input observation dims for the object detector")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    params = vars(parser.parse_args())

    if params["display_off"]:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    assert params["hold_det_img"] is True and params["hold_vae_img"] is False

    # Remove a couple of parameters that we dont want to log
    start_carla = params["start_carla"]; del params["start_carla"]
    restart = params["restart"]; del params["restart"]

    # tf.compat.v1.reset_default_graph()

    # Start training
    train(params, start_carla, restart)