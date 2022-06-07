import numpy as np
from CarlaEnv.wrappers import angle_diff, vector

low_speed_timer = 0
max_distance    = 10.0  # Max distance from center before terminating
target_speed    = 40.0 # kmh

def create_reward_fn(reward_fn, max_speed=-1):
    """
        Wraps input reward function in a function that adds the
        custom termination logic used in these experiments

        reward_fn (function(CarlaEnv)):
            A function that calculates the agent's reward given
            the current state of the environment. 
        max_speed:
            Optional termination criteria that will terminate the
            agent when it surpasses this speed.
            (If training with reward_kendal, set this to 20)
    """
    def func(env):
        terminal_reason = "Running..."

        # Stop if speed is less than 1.0 km/h after the first 5s of an episode
        global low_speed_timer
        low_speed_timer += 1.0 / env.fps
        speed = env.vehicle.get_speed()
        if low_speed_timer > 5.0 and speed < 1.0 / 3.6:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"

        # Stop if distance from center > max distance
        if env.distance_from_center > max_distance:
            env.terminal_state = True
            env.curb_hit = True
            terminal_reason = "Off-track"

        # Stop if speed is too high
        if max_speed > 0 and speed_kmh > max_speed:
            env.terminal_state = True
            terminal_reason = "Too fast"

        # Calculate reward
        reward1 = 0
        reward2 = 0
        if not env.terminal_state:
            reward1, reward2 = reward_fn(env)
        else:
            low_speed_timer = 0.0
            if env.track_completed:
                reward1 += 100
                reward2 += 100
            else:
                reward1 -= 100
                reward2 -= 100

        if env.terminal_state:
            env.extra_info.extend([
                terminal_reason,
                ""
            ])
        return reward1, reward2
    return func

#---------------------------------------------------
# Create reward functions dict
#---------------------------------------------------

reward_functions = {}

# Kenall's (Learn to Drive in a Day) reward function
def reward_kendall(env):
    speed_kmh = 3.6 * env.vehicle.get_speed()
    return speed_kmh

reward_functions["reward_kendall"] = create_reward_fn(reward_kendall)

# Our reward function (additive)
def reward_speed_centering_angle_add(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               + centering factor (1 when centered, 0 when not)
               + angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    min_speed = 35.0 # km/h
    max_speed = 45.0 # km/h

    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)

    speed_kmh = 3.6 * env.vehicle.get_speed()
    if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                  # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
    else:                                         # Otherwise
        speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

    steer_diff_factor = 1.0
    if env.penalize_steer_diff:
        steer_diff_factor = max(1.0 - abs(env.steer_diff/np.deg2rad(180)), 0.0)
    dist_obstacle_factor = 1.0
    if env.obstacle_en and env.penalize_dist_obstacle:
        dist_obstacle_factor = max(min(env.r / 20, 1.0), 0.0)
    # Final reward
    reward1 = (3 * speed_reward)+ (10* centering_factor) + (3 * angle_factor) + (2 * dist_obstacle_factor)
    reward2 = speed_reward + centering_factor + angle_factor + steer_diff_factor + dist_obstacle_factor

    return reward1, reward2

reward_functions["reward_speed_centering_angle_add"] = create_reward_fn(reward_speed_centering_angle_add)

# Our reward function (multiplicative)
def reward_speed_centering_angle_multiply(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    min_speed = 35.0 # km/h
    max_speed = 45.0 # km/h

    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)

    speed_kmh = 3.6 * env.vehicle.get_speed()
    if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                  # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
    else:                                         # Otherwise
        speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)
    #print("steer_diff", env.steer_diff)
    #steer_diff_factor2 = max(1.0 - abs(env.steer_diff/np.deg2rad(180)), 0.0)
    #print("steer_factor", steer_diff_factor2)
    steer_diff_factor = max(1.0 - abs(env.steer_diff/np.deg2rad(90)), 0.0)
    #print("steer_factor", steer_diff_factor)
    # Final reward
    # check if penalizing distance to obstacle is enabled

    dist_obstacle_factor = 1.0
    if env.obstacle_en and env.penalize_dist_obstacle:
        dist_obstacle_factor = max(min(env.r/20, 1.0),0.0)
    # print("dist_obstacle_factor ", dist_obstacle_factor)
    reward1 = speed_reward * centering_factor * angle_factor * dist_obstacle_factor
    reward2 = speed_reward * centering_factor * angle_factor * steer_diff_factor * dist_obstacle_factor
    
    return reward1, reward2


reward_functions["reward_speed_centering_angle_multiply"] = create_reward_fn(reward_speed_centering_angle_multiply)
