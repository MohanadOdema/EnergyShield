import os
import subprocess
import time

import carla
import random
import gym
import pygame
from gym.utils import seeding
from pygame.locals import *
import numpy as np

from hud import HUD
from planner import RoadOption, compute_route_waypoints
from wrappers import *
from offloading_utils import *
from agents.navigation import *
# from .navigation.behavior_agent import BehaviorAgent
from OD_utils import *
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class CarlaOffloadEnv(gym.Env):
    """

        Note that you may also need to add the following line to
        Unreal/CarlaUE4/Config/DefaultGame.ini to have a specific map included in the package:       
        +MapsToCook=(FilePath="/Game/Carla/Maps/Town07")
    """

    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    def __init__(self, host="127.0.0.1", port=2000,
                 viewer_res=(1280, 720), obs_res=(1280, 720),       # spectator camera and dash camera resolutions
                 reward_fn=None, encode_state_fn=None, 
                 synchronous=True, fps=30, action_smoothing=0.9,
                 start_carla=True, apply_filter=False, obstacle=False, penalize_steer_diff=False, penalize_dist_obstacle=False, gaussian=False, track=1, params=None, mode='train'):
        """
            Initializes a gym-like environment that can be used to interact with CARLA.

            Connects to a running CARLA enviromment (tested on version 0.9.5) and
            spwans a lincoln mkz2017 passenger car with automatic transmission.
            
            This vehicle can be controlled using the step() function,
            taking an action that consists of [steering_angle, throttle].

        """
        self.obstacle_en            = obstacle
        self.track                  = track
        self.gaussian               = gaussian
        self.obstacle_hit           = False
        self.curb_hit               = False
        self.penalize_dist_obstacle = penalize_dist_obstacle
        self.total_reward1          = 0.0
        self.total_reward2          = 0.0
        self.safety_filter          = None
        self.penalize_steer_diff    = penalize_steer_diff
        self.steer_diff_avg         = 0
        self.apply_filter_counter   = 0
        self.steer_cap              = 0.6428
        self.position_multiplier    = params["pos_mul"]
        self.follow_waypoints       = params["follow_waypoints"]
        self.agent                  = None

        self.energy_monitor = OffloadingManager(params)
        self.throughput_prober = UploadThroughputSampler(params)

        if apply_filter:
            self.safety_filter = SafetyFilter()

        # Initialize pygame for visualization
        pygame.init()
        pygame.font.init()
        width, height = viewer_res
        if obs_res is None:
            out_width, out_height = width, height
        else:
            out_width, out_height = obs_res     # OD: set to (160,80) default from the calling function
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.synchronous = synchronous

        # Setup gym environment
        self.seed()
        self.action_space = gym.spaces.Box(np.array([-self.steer_cap, 0]), np.array([self.steer_cap, 1]), dtype=np.float32) # steer, throttle -- first array is the lowest acceptable values, 2nd is the highest
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(*obs_res, 3), dtype=np.float32)
        self.metadata["video.frames_per_second"] = self.fps = self.average_fps = fps
        self.spawn_point = 1
        self.action_smoothing = action_smoothing
        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
        self.valid_obstacle = False
        self.world = None
        self.steer_diff = 0.0
        try:
            # Connect to carla
            self.client = carla.Client(host, port)
            self.client.set_timeout(2.0)

            # Create world wrapper
            self.world = World(self.client)

            if self.world is not None:
                self.world.destroy()

            if self.synchronous:
                settings = self.world.get_settings()
                settings.fixed_delta_seconds = 0.01     # Fixed time step in the simulation environment
                settings.synchronous_mode = True
                self.world.apply_settings(settings)

            # Get spawn location
            start_index, end_index, road_option_count = 0,0,0
            if track == 1:
                start_index = 46
                end_index   = 205           # These spawn positions are on the lane path [3,7,69,197,205,239]
                road_option_count = 1
            lap_start_wp = self.world.map.get_waypoint(self.world.map.get_spawn_points()[start_index].location)
            spawn_transform = lap_start_wp.transform
            spawn_transform.location += carla.Location(z=1.0)
            lap_end_wp = self.world.map.get_waypoint(self.world.map.get_spawn_points()[end_index].location)
            spawn_transform_end = lap_end_wp.transform
            spawn_transform_end.location += carla.Location(z=1.0)
            self.route_waypoints = compute_route_waypoints(self.world.map, lap_start_wp, lap_end_wp, resolution=1.0,
                                                           plan=[
                                                               RoadOption.STRAIGHT]*road_option_count)  # + [RoadOption.RIGHT] * 2 + [RoadOption.STRAIGHT] * 5)
            self.route_waypoints = self.route_waypoints[:int(len(self.route_waypoints)//4)]

            self.destination = self.route_waypoints[100][0].transform.location

            # print(len(self.route_waypoints))
            self.current_waypoint_index = 0
            self.checkpoint_waypoint_index = 0

            self.vehicle = Vehicle(self.world, spawn_transform,
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e))

            if self.follow_waypoints:
                self.agent = BasicAgent(self.vehicle)

            # Create hud
            self.hud = HUD(width, height)  
            self.hud.set_vehicle(self.vehicle)
            self.world.on_tick(self.hud.on_world_tick)

            # Create cameras
            self.dashcam = Camera(self.world, out_width, out_height,
                                  transform=camera_transforms["dashboard"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image(e),
                                  sensor_tick=0.0 if self.synchronous else 1.0/self.fps)
            self.camera  = Camera(self.world, width, height,
                                  transform=camera_transforms["spectator"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e),
                                  sensor_tick=0.0 if self.synchronous else 1.0/self.fps)

            # Create obstacle
            self.obstacles = []
            self.create_obstacles = True
            print("init obstacles")

        except Exception as e:
            self.close()
            raise e

        # Reset env to set initial state
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def reset(self, is_training=True, seed=0):

        # Do a soft reset (teleport vehicle)
        self.seed(int(seed+1))
        self.xi = 0
        self.r  = -1
        self.track_completed = False
        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        #self.vehicle.control.brake = float(0.0)
        self.vehicle.tick()

        self.energy = 0.0
        self.off_latency = 0.0      # If the current policy is local it translates to the local execution latency
        self.action = 0
        self.probe_tu = 0
        self.delta_tu = 0

        # Always start from the beginning
        self.curb_hit = False
        self.obstacle_hit = False
        self.steer_diff_avg = 0
        self.apply_filter_counter = 0
        waypoint, _ = self.route_waypoints[0]
        self.current_waypoint_index = 0
        transform = waypoint.transform
        transform.location += carla.Location(z=0)
        self.vehicle.set_transform(transform)
        self.vehicle.set_simulate_physics(False) # Reset the car's physics
        self.vehicle.set_simulate_physics(True)
        self.obstacle_percentage = 1.0
        gaussian = self.gaussian
        gasussian_var = 1.15
        if is_training:
            self.obstacle_percentage = 1.0

        self.energy_monitor.reset()     # reset stats for new episode

        if self.agent is not None:
            print(self.destination)
            self.agent.set_destination(self.destination)

        # Creating obstacles
        if self.obstacle_en:
            if random.random() < self.obstacle_percentage:
                print("create obstacles")
                print("len_obstacles", len(self.obstacles))
                #self.valid_obstacle = True
                if self.track == 1:
                    spawn_idx = random.randint(60,90)
                spawn_transform = self.route_waypoints[spawn_idx][0].transform
                spawn_transform.rotation = carla.Rotation(yaw= spawn_transform.rotation.yaw-180)
                if (not is_training) and gaussian:
                    spawn_transform.location.x = spawn_transform.location.x + np.random.normal(0,gasussian_var,1)[0]
                    spawn_transform.location.y = spawn_transform.location.y + np.random.normal(0,gasussian_var,1)[0]
                if self.track == 1:
                    len_obs = 1
                current_idx  = 0
                for i in range(0,len_obs):
                    print("i", i)
                    if self.create_obstacles:
                        self.obstacles.append(Obstacle(self.world, spawn_transform))
                        ret_val = self.obstacles[i].is_valid()      
                        while ret_val is None:              
                            print("try again")          # First obstacle spawning has an issue
                            spawn_idx = random.randint(30,50)
                            if (spawn_idx + current_idx)  >= len(self.route_waypoints):
                                continue
                            spawn_transform = self.route_waypoints[current_idx + spawn_idx][0].transform
                            spawn_transform.rotation = carla.Rotation(yaw= spawn_transform.rotation.yaw-180)
                            if (not is_training) and gaussian:
                                spawn_transform.location.x = spawn_transform.location.x + np.random.normal(0,gasussian_var,1)[0]
                                spawn_transform.location.y = spawn_transform.location.y + np.random.normal(0,gasussian_var,1)[0]
                            self.obstacles[i] = Obstacle(self.world, spawn_transform)
                            ret_val = self.obstacles[i].is_valid()
                    else:
                        if i >= len(self.obstacles):
                           break
                        self.obstacles[i].set_transform(spawn_transform)
                    print(spawn_idx)
                    current_idx += spawn_idx
                    spawn_idx = random.randint(30,50)
                    if spawn_idx + current_idx >= len(self.route_waypoints):
                        break
                    # The next spawning position in the loop
                    spawn_transform = self.route_waypoints[spawn_idx + current_idx][0].transform
                    spawn_transform.rotation = carla.Rotation(yaw= spawn_transform.rotation.yaw-180)
                    # add some randomization
                    if (not is_training) and gaussian:
                        spawn_transform.location.x = spawn_transform.location.x + np.random.normal(0,gasussian_var,1)[0]
                        spawn_transform.location.y = spawn_transform.location.y + np.random.normal(0,gasussian_var,1)[0]
                self.create_obstacles = False 
                print("len obstacles", len(self.obstacles))
                self.obstacle_counter = 0

        if self.synchronous:
            ticks = 0
            while ticks < self.fps * 2:
                self.world.tick()
                # exit(0)
                try:
                    self.world.wait_for_tick(seconds=1.0/self.fps + 0.1)
                    print('You cannot reach here if synchronous!')
                except:
                    pass
                ticks += 1 
        else:
            time.sleep(2.0)

        self.terminal_state = False # Set to True when we want to end episode
        self.track_completed   = False
        self.closed = False         # Set to True when ESC is pressed
        self.extra_info = []        # List of extra info shown on the HUD
        self.observation = self.observation_buffer = None   # Last received observation
        self.viewer_image = self.viewer_image_buffer = None # Last received image to show in the viewer
        self.start_t = time.time()
        self.step_count = 0
        self.is_training = is_training

        self.start_waypoint_index = self.current_waypoint_index
        
        # Metrics
        self.total_reward1  = 0.0
        self.total_reward2  = 0.0
        self.min_distance_to_obstacle = 10000
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.laps_completed = 0.0

        # DEBUG: Draw path
        # self._draw_path(life_time=1000.0, skip=10)

        # Return initial observation
        return self.step(None)[0]

    def close(self):
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def render(self, mode="human"):
        # Get maneuver name
        if self.current_road_maneuver == RoadOption.LANEFOLLOW: maneuver = "Follow Lane"
        elif self.current_road_maneuver == RoadOption.LEFT:     maneuver = "Left"
        elif self.current_road_maneuver == RoadOption.RIGHT:    maneuver = "Right"
        elif self.current_road_maneuver == RoadOption.STRAIGHT: maneuver = "Straight"
        elif self.current_road_maneuver == RoadOption.VOID:     maneuver = "VOID"
        else:                                                   maneuver = "INVALID(%i)" % self.current_road_maneuver

        # Add metrics to HUD
        self.extra_info.extend([
            "Reward: % 19.2f" % self.last_reward,
            "",
            # "Maneuver:        % 11s"       % maneuver,
            "Laps completed:    % 7.2f %%" % (self.laps_completed * 100.0),
            "Distance traveled: % 7d m"    % self.distance_traveled,
            "Center deviance:   % 7.2f m"  % self.distance_from_center,
            # "Avg center dev:    % 7.2f m"  % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h"  % (3.6 * self.speed_accum / self.step_count),
            "",
            "Deadline:         % 1.0f ms"   % (self.energy_monitor.deadline),
            "Probe tu:        %7.2f Mbps"  % (self.probe_tu),
            "Delta tu:        %7.2f Mbps"  % (self.delta_tu),
            "",
            "Local Ergy:        %7.2f mJ"  % (self.energy_monitor.full_local_energy),
            "Exp Ergy:          %7.2f mJ"  % (self.energy_monitor.exp_total_energy),
            "Local Latency:     %7.2f ms"  % (self.energy_monitor.full_local_latency),
            "Exp Latency:       %7.2f ms"  % (self.energy_monitor.exp_total_latency),
            "",
            "Selected Action:     % 1.0f"  % (self.energy_monitor.selected_action),
            "Correct Action:      % 1.0f"  % (self.energy_monitor.correct_action),
            "",
            "Missed Deadlines:     %1.0f"  % (self.energy_monitor.missed_deadlines),
            "Max succ interrupts:  %1.0f"  % (self.energy_monitor.max_succ_interrupts),
            "Missed offloads:      %1.0f"  % (self.energy_monitor.missed_offloads)
        ])

        # Blit image from spectator camera
        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        # Superimpose current observation into top-right corner
        obs_h, obs_w = self.observation.shape[:2]
        view_h, view_w = self.viewer_image.shape[:2]
        pos = (view_w - obs_w - 10, 10)
        self.display.blit(pygame.surfarray.make_surface(self.observation.swapaxes(0, 1)), pos)

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = [] # Reset extra info list

        # Render to screen
        pygame.display.flip()

        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            # Turn display surface into rgb_array
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        elif mode == "state_pixels":
            return self.observation
    
    def obstacle_state_estimator(self, player_transform, obstacle_location, player_speed):
        r_v = np.array([player_transform.location.x, player_transform.location.y]) - np.array(
            [obstacle_location.x, obstacle_location.y])
        self.r = np.linalg.norm(r_v)
        psi = player_transform.rotation.yaw * math.pi/180
        self.xi = math.atan2(player_transform.location.y- obstacle_location.y, player_transform.location.x- obstacle_location.x) - psi
        self.xi = math.atan2(math.sin(self.xi), math.cos(self.xi))


    def step(self, action):
        self.steer_diff = 0.0
        
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")

        # Asynchronous update logic
        if not self.synchronous:
            if self.fps <= 0:
                # Go as fast as possible
                self.clock.tick()
            else:
                # Sleep to keep a steady fps
                self.clock.tick_busy_loop(self.fps)
            
            # Update average fps (for saving recordings)
            if action is not None:
                self.average_fps = self.average_fps * 0.5 + self.clock.get_fps() * 0.5

        if self.obstacle_en:
            self.obstacle_state_estimator(self.vehicle.get_transform(), self.obstacles[self.obstacle_counter].get_transform().location, self.vehicle.get_speed())
        #print("xi", self.xi)
        #print("r", self.r)
        # Take action
        rl_steer = -1000
        rl_throttle = -1000
        filter_steer = -1000
        filter_throttle = -1000
        filter_applied = False
        xi = self.xi
        r = self.r

        self.probe_tu, self.delta_tu = self.throughput_prober.sample(1)                         
        self.energy_monitor.certify_deadline()
        if not self.energy_monitor.verify_combinations():
            print("Local pipeline latency > deadline!")         # operation needs to be verified
            self.close()
        self.energy_monitor.determine_offloading_decision(self.probe_tu, self.delta_tu)      
       
        if self.follow_waypoints:
            control = self.agent.run_step()
            steer = control.steer
            throttle = control.throttle
            # print(steer, type(steer), throttle, type(throttle))
            # exit(0)
            # print(control)

            self.vehicle.control.steer = self.vehicle.control.steer * self.action_smoothing + steer * (1.0-self.action_smoothing)
            self.vehicle.control.throttle = self.vehicle.control.throttle * self.action_smoothing + throttle * (1.0-self.action_smoothing)

            # control.manual_gear_shift = False 
            # self.vehicle.apply_control(control)

            # self.vehicle.set_autopilot(True)
            rl_steer = self.vehicle.control.steer
            rl_throttle = self.vehicle.control.throttle
            filter_steer = self.vehicle.control.steer
            filter_throttle = self.vehicle.control.throttle

        elif action is not None:
            steer, throttle = [float(a) for a in action]
            self.vehicle.control.steer    = self.vehicle.control.steer * self.action_smoothing + steer * (1.0-self.action_smoothing)
            self.vehicle.control.throttle = self.vehicle.control.throttle * self.action_smoothing + throttle * (1.0-self.action_smoothing)
            rl_steer = self.vehicle.control.steer
            rl_throttle = self.vehicle.control.throttle
            if self.safety_filter is not None:
                self.safety_filter.set_filter_inputs(self.xi, self.r)
                self.vehicle.control, self.steer_diff, filter_applied = self.safety_filter.filter_control(self.vehicle.control)
                if filter_applied:
                    self.apply_filter_counter+=1
                    self.steer_diff_avg = (self.steer_diff_avg+self.steer_diff)/self.apply_filter_counter

            filter_steer = self.vehicle.control.steer
            filter_throttle = self.vehicle.control.throttle    

        #filter_data = {rl_steer, rl_throttle, filter_steer, filter_throttle, xi, r}
        # Tick game
        self.hud.tick(self.world, self.clock)
        self.world.tick()

        # Synchronous update logic
        if self.synchronous:
            self.clock.tick()
            while True:
                try:
                    self.world.wait_for_tick(seconds=1.0/self.fps + 0.1)
                    break
                except:
                    # Timeouts happen occasionally for some reason, however, they seem to be fine to ignore
                    self.world.tick()
                break

        # Get most recent observation and viewer image
        self.observation = self._get_observation()
        self.viewer_image = self._get_viewer_image()
        encoded_state = self.encode_state_fn(self)      

        # Get vehicle transform
        transform = self.vehicle.get_transform()
        # fill in vehicle plot data
        ego_x = transform.location.x
        ego_y = transform.location.y
        # Keep track of closest waypoint on the route
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)] 
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(transform.location - wp.transform.location)[:2])    # ignore z-axis
            if dot > 0.0: # Did we pass the waypoint?
                waypoint_index += 1 # Go to next waypoint
            else:
                break
        self.current_waypoint_index = waypoint_index
        obstacle_x = None
        obstacle_y = None
        if self.obstacle_en:
            r_v = np.array([self.vehicle.get_transform().location.x, self.vehicle.get_transform().location.y]) - np.array(
                [self.obstacles[self.obstacle_counter].get_transform().location.x, self.obstacles[self.obstacle_counter].get_transform().location.y]) # could just compare self.r from 394?
            distance_to_obstacle = np.linalg.norm(r_v)
            if distance_to_obstacle < self.min_distance_to_obstacle:
                self.min_distance_to_obstacle = distance_to_obstacle
            # fill in obstacle plot data
            obstacle_x = self.obstacles[self.obstacle_counter].get_transform().location.x
            obstacle_y = self.obstacles[self.obstacle_counter].get_transform().location.y
            # check the closest obstacle to the vehicle
            dist_obstacles = [100000] * len(self.obstacles)
            for i in range(len(self.obstacles)):
                obstacle_loc = self.obstacles[i].get_transform().location
                r_v = np.array([self.vehicle.get_transform().location.x, self.vehicle.get_transform().location.y]) - np.array(
                    [obstacle_loc.x, obstacle_loc.y])
                dist_obstacles[i] = np.linalg.norm(r_v)
            self.obstacle_counter = np.argmin(dist_obstacles)

        # Calculate deviation from center of the lane
        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
        self.next_waypoint, self.next_road_maneuver       = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        # DEBUG: Draw current waypoint
        self.world.debug.draw_point(self.current_waypoint.transform.location, color=carla.Color(0, 255, 0), life_time=1.0)

        # Calculate distance traveled
        self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        # Accumulate speed
        self.speed_accum += self.vehicle.get_speed()

        # Get lap count
        self.laps_completed = (self.current_waypoint_index - self.start_waypoint_index) / len(self.route_waypoints)
        # print(self.current_waypoint_index)
        if self.laps_completed >= 1:
            # End after 3 laps
            self.laps_completed = 1
            self.track_completed = True
            self.terminal_state = True
                
        # Update checkpoint for training
        if self.is_training:
            checkpoint_frequency = 50 # Checkpoint frequency in meters
            self.checkpoint_waypoint_index = (self.current_waypoint_index // checkpoint_frequency) * checkpoint_frequency
        
        # Call external reward fn
        reward1, reward2 = self.reward_fn(self)
        if self.penalize_steer_diff:
            self.last_reward = reward2
        else:
            self.last_reward = reward1
        self.total_reward1 += reward1
        self.total_reward2 += reward2
        self.step_count += 1

        # Check for ESC press
        pygame.event.pump()
        if pygame.key.get_pressed()[K_ESCAPE]:
            self.close()
            self.terminal_state = True

        env_dict = rounded_dict({"closed": self.closed, "ego_x": ego_x, "ego_y": ego_y, "obstacle_x": obstacle_x, "obstacle_y":obstacle_y, 
                        "route":self.route_waypoints, "current_waypoint_index":self.current_waypoint_index, "xi":self.xi, "r": self.r, 
                        "rl_steer":rl_steer, "rl_throttle":rl_throttle, "filter_steer":filter_steer, "filter_throttle":filter_throttle, 
                        "sim_time":time.clock(), "filter_applied": filter_applied, "action_none":(action is None)})
        offloading_dict = rounded_dict({"probe_tu": self.probe_tu[0], "delta_tu":self.delta_tu[0], 
                            "selected_action": self.energy_monitor.selected_action, "correct_action": self.energy_monitor.correct_action,
                            "probe_latency": self.energy_monitor.probe_off_latency, "probe_energy": self.energy_monitor.probe_off_energy,
                            "actual_latency": self.energy_monitor.actual_off_latency, "actual_energy": self.energy_monitor.actual_off_energy,
                            "exp_latency": self.energy_monitor.exp_total_latency, "exp_energy": self.energy_monitor.exp_total_energy,
                            "missed_deadline_flag": self.energy_monitor.missed_deadline_flag, "missed_deadlines": self.energy_monitor.missed_deadlines,
                            "succ_interrupts": self.energy_monitor.succ_interrupts, "max_succ_interrupts": self.energy_monitor.max_succ_interrupts,
                            "missed_offloads": self.energy_monitor.missed_offloads, "misguided_energy": self.energy_monitor.misguided_energy})

        # print('-'*80)
        # print(offloading_dict)
        # print('-'*80)

        return encoded_state, self.last_reward, self.terminal_state, env_dict, offloading_dict

    def _draw_path(self, life_time=60.0, skip=0):
        """
            Draw a connected path from start of route to end.
            Green node = start
            Red node   = point along path
            Blue node  = destination
        """
        for i in range(0, len(self.route_waypoints)-1, skip+1):
            w0 = self.route_waypoints[i][0]
            w1 = self.route_waypoints[i+1][0]
            self.world.debug.draw_line(
                w0.transform.location + carla.Location(z=0.25),
                w1.transform.location + carla.Location(z=0.25),
                thickness=0.1, color=carla.Color(255, 0, 0),
                life_time=life_time, persistent_lines=False)
            self.world.debug.draw_point(
                w0.transform.location + carla.Location(z=0.25), 0.1,
                carla.Color(0, 255, 0) if i == 0 else carla.Color(255, 0, 0),
                life_time, False)
        self.world.debug.draw_point(
            self.route_waypoints[-1][0].transform.location + carla.Location(z=0.25), 0.1,
            carla.Color(0, 0, 255),
            life_time, False)

    def _get_observation(self):
        while self.observation_buffer is None:
            pass
        obs = self.observation_buffer.copy()
        self.observation_buffer = None
        return obs

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _on_collision(self, event):
        obstacle_name = get_actor_display_name(event.other_actor)
        self.hud.notification("Collision with {}".format(obstacle_name))
        print (obstacle_name)
        #if self.total_reward is not None:
        #    self.total_reward-= 10
        if "Pedestrian" in obstacle_name:
            print("Hit a pedestrian!")
            self.obstacle_hit = True
        elif "Fence" in obstacle_name:
            print("Hit a curb")
            self.curb_hit = True
        self.terminal_state = True

    def _on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

def reward_fn(env):
    early_termination = False
    if early_termination:
        # If speed is less than 1.0 km/h after 5s, stop
        if time.time() - env.start_t > 5.0 and env.vehicle.get_speed() < 1.0 / 3.6:
            env.terminal_state = True

        # If distance from center > 7, stop
        if env.distance_from_center > 7.0:
            print("distance > 7")
            env.terminal_state = True
        
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    if np.dot(fwd[:2], wp_fwd[:2]) > 0:
        return env.vehicle.get_speed()
    return 0

if __name__ == "__main__":
    # Example of using CarlaEnv with keyboard controls
    env = CarlaOffloadEnv(obs_res=(160, 80), reward_fn=reward_fn)
    action = np.zeros(env.action_space.shape[0])
    while True:
        env.reset(is_training=True, seed=0)
        while True:
            # Process key inputs
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[K_LEFT] or keys[K_a]:
                action[0] = -0.5
            elif keys[K_RIGHT] or keys[K_d]:
                action[0] = 0.5
            else:
                action[0] = 0.0
            action[0] = np.clip(action[0], -1, 1)
            action[1] = 1.0 if keys[K_UP] or keys[K_w] else 0.0

            # Take action
            obs, _, done, info = env.step(action)
            if info["closed"]: # Check if closed
                exit(0)
            env.render() # Render
            if done: break
    env.close()
