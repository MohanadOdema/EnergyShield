import carla

import random
import time
import collections
import math
import numpy as np
import weakref
import pygame
import sys
import onnx
from onnx_tf.backend import prepare
import numpy as np
from numpy import (array, dot, arccos, clip)
import tensorflow
import tensorflow.compat.v1 as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.compat.v1.disable_eager_execution()

def print_transform(transform):
    print("Location(x={:.2f}, y={:.2f}, z={:.2f}) Rotation(pitch={:.2f}, yaw={:.2f}, roll={:.2f})".format(
            transform.location.x,
            transform.location.y,
            transform.location.z,
            transform.rotation.pitch,
            transform.rotation.yaw,
            transform.rotation.roll
        )
    )

def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[:truncate-1] + u"\u2026") if len(name) > truncate else name

def angle_diff(v0, v1):
    """ Calculates the signed angle difference (-pi, pi] between 2D vector v0 and v1 """
    angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
    if angle > np.pi: angle -= 2 * np.pi
    elif angle <= -np.pi: angle += 2 * np.pi
    return angle

def distance_to_line(A, B, p):
    num   = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    if np.isclose(denom, 0):
        return np.linalg.norm(p - A)
    return num / denom

def vector(v):
    """ Turn carla Location/Vector3D/Rotation to np.array """
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])

camera_transforms = {
    "spectator": carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
    "dashboard": carla.Transform(carla.Location(x=1.6, z=1.7))
}

#===============================================================================
# CarlaActorBase
#===============================================================================

class CarlaActorBase(object):
    def __init__(self, world, actor):
        self.world = world
        self.actor = actor
        self.world.actor_list.append(self)
        self.destroyed = False

    def destroy(self):
        if self.destroyed:
            raise Exception("Actor already destroyed.")
        else:
            print("Destroying ", self, "...")
            self.actor.destroy()
            self.world.actor_list.remove(self)
            self.destroyed = True

    def get_carla_actor(self):
        return self.actor

    def tick(self):
        pass

    def __getattr__(self, name):
        """Relay missing methods to underlying carla actor"""
        return getattr(self.actor, name)

#===============================================================================
# CollisionSensor
#===============================================================================

class CollisionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_collision_fn):
        self.on_collision_fn = on_collision_fn

        # Collision history
        self.history = []

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.collision")

        # Create and setup sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: CollisionSensor.on_collision(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_collision(weak_self, event):
        print("on_collision sensor")
        self = weak_self()
        if not self:
            return

        # Call on_collision_fn
        if callable(self.on_collision_fn):

            self.on_collision_fn(event)


#===============================================================================
# LaneInvasionSensor
#===============================================================================

class LaneInvasionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_invasion_fn):
        self.on_invasion_fn = on_invasion_fn

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")

        # Create sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: LaneInvasionSensor.on_invasion(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Call on_invasion_fn
        if callable(self.on_invasion_fn):
            self.on_invasion_fn(event)

#===============================================================================
# Camera
#===============================================================================

class Camera(CarlaActorBase):
    def __init__(self, world, width, height, transform=carla.Transform(),
                 sensor_tick=0.0, attach_to=None, on_recv_image=None,
                 camera_type="sensor.camera.rgb", color_converter=carla.ColorConverter.Raw):
        self.on_recv_image = on_recv_image
        self.color_converter = color_converter

        # Setup camera blueprint
        camera_bp = world.get_blueprint_library().find(camera_type)
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("sensor_tick", str(sensor_tick))

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(camera_bp, transform, attach_to=attach_to.get_carla_actor())
        actor.listen(lambda image: Camera.process_camera_input(weak_self, image))
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)
    
    @staticmethod
    def process_camera_input(weak_self, image):
        self = weak_self()
        if not self:
            return
        if callable(self.on_recv_image):
            image.convert(self.color_converter)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.on_recv_image(array)

    def destroy(self):
        super().destroy()

#===============================================================================
# Vehicle
#===============================================================================
class Obstacle(CarlaActorBase):
    def __init__(self, world, transform=carla.Transform(), obstacle_type="*static*"): # return to "*walker*" or "*static*"
        self.obstacle_transform = None
        self._set_obstacle_transform(transform)
        obstacle_bp = world.get_blueprint_library().filter(obstacle_type)[18]#[int(np.random.choice(np.arange(20)))]   #18
        if obstacle_bp.has_attribute('is_invincible'):
            obstacle_bp.set_attribute('is_invincible', 'false')
        self.obstacle = world.try_spawn_actor(obstacle_bp, self.obstacle_transform)
        super().__init__(world, self.obstacle)

    def is_valid(self):
        return self.obstacle
    def _set_obstacle_transform(self, transform=carla.Transform()):
        player_transform = transform
        player_location = player_transform.location
        player_yaw = player_transform.rotation.yaw * math.pi / 180
        x_offset = random.uniform(15,16)
        y_offset = 0 #random.uniform(-5, 5)
        #print('x_offset', x_offset, 'y_offset', y_offset)
        obstacle_location = carla.Location(
            x=player_location.x + (x_offset * math.cos(player_yaw)) + (y_offset * math.sin(player_yaw)),
            y=player_location.y + (x_offset * math.sin(player_yaw)) + (y_offset * math.cos(player_yaw)),
            z=player_location.z)
        self.obstacle_transform = carla.Transform(obstacle_location,
                                             carla.Rotation(roll=0, pitch=0, yaw=-player_transform.rotation.yaw))
    def set_transform(self, transform):
        #self._set_obstacle_transform(transform)
        #print("obstacle transform")
        #print_transform(transform)
        transform.location += carla.Location(z=1.0)
        self.obstacle.set_transform(transform)
    def tick(self):
        pass
    def destroy(self):
        super().destroy()


class Vehicle(CarlaActorBase):
    def __init__(self, world, transform=carla.Transform(),
                 on_collision_fn=None, on_invasion_fn=None,
                 vehicle_type="vehicle.audi.a2"):
        # Setup vehicle blueprint
        vehicle_bp = world.get_blueprint_library().find(vehicle_type)
        color = vehicle_bp.get_attribute("color").recommended_values[2]
        vehicle_bp.set_attribute("color", color)

        world.destroy()
        # Create vehicle actor
        actor = world.spawn_actor(vehicle_bp, transform)
        print("Spawned actor \"{}\"".format(actor.type_id))
        # print(actor.get_physics_control())
        # exit()

        super().__init__(world, actor)

        # Maintain vehicle control
        self.control = carla.VehicleControl()

        if callable(on_collision_fn):
            self.collision_sensor = CollisionSensor(world, self, on_collision_fn=on_collision_fn)
        if callable(on_invasion_fn):
            self.lane_sensor = LaneInvasionSensor(world, self, on_invasion_fn=on_invasion_fn)

    def tick(self):
        self.actor.apply_control(self.control)

    def get_speed(self):
        velocity = self.get_velocity()
        return np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    def get_closest_waypoint(self):
        return self.world.map.get_waypoint(self.get_transform().location, project_to_road=True)

#===============================================================================
# World
#===============================================================================

class World():
    def __init__(self, client):
        self.world = client.get_world()
        self.map = self.get_map()
        self.actor_list = []

    def tick(self):
        for actor in list(self.actor_list):
            actor.tick()
        self.world.tick()

    def destroy(self):
        print("Destroying all spawned actors")
        for actor in list(self.actor_list):
            actor.destroy()

    def get_carla_world(self):
        return self.world

    def __getattr__(self, name):
        """Relay missing methods to underlying carla object"""
        return getattr(self.world, name)


#===============================================================================
# Safety Filter
#===============================================================================

def output_delta_T(control=None):
    # Based on the measurement of the xi and r, (maybe speed and other factors) - we can output the delta_T 
    # Should include some preprocessing to floor it to a multiplier

    # delta_T LOGIC:
    # It should be defined naturally as a positive int 
    # if 0 or 1, then I have only one window or less to complete execution,  set the recover flag and execute locally for maximum robustness
    # if any number > 1, I compare the current tx window (starts from 1) against it, and if current_Tx == delta-1 and it has not concluded processing yet, it means that next window when current_Tx == delta should be local execution regardless 
    # raise NotImplementedError

    return random.randint(1, 20)

class SafetyFilter2():         # with eager_execution for tensorflow debugging purposes
    def __init__(self):
        self.input_shape = (1,1,2)
        self.input_image = tf.placeholder(shape=self.input_shape, dtype=np.float32, name='xi_and_r')
        model = tf.keras.models.load_model('1.h5')
        self.tf_model1 = model(self.input_image)

    def init_session(self,sess=None, init_logging=True):
        if sess is None:
            self.sess = tf.Session()
            self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        else:
            self.sess = sess

    def filter_control(self):
        input = np.array([[[3, 4]]])
        input = input.astype(np.float32)

        # output = self.tf_model1.run(input)['PathDifferences3_Add'][0,0,0]
        output = self.sess.run(self.tf_model1, feed_dict={self.input_image: input})

        return output

class SafetyFilter():
    def __init__(self):
        # load shield models
        tf.disable_eager_execution()
        model = tf.keras.models.load_model('1.h5')
        self.tf_model1 = model
        model = tf.keras.models.load_model('2.h5')
        self.tf_model2 = model
        model = tf.keras.models.load_model('3.h5')
        self.tf_model3 = model

        self.filtering  = True

        self.beta   = None
        self.r      = None
        self.psi    = None
        self.xi     = None
        self.sigma = 0.28 # 0.48
        self.l_r  = 1.6 # 2
        self.rBar = 4
        self.steer_to_angle = 1.22173

    def toggle_filtering(self):
        self.filtering = not self.filtering
    def barrier(self):
        return self.rBar/(self.sigma*math.cos(self.xi/2) + 1 - self.sigma)
    def filter_control(self, control):
        if (not self.filtering) or (self.xi is None):
            return control, 0, False

        delta = control.steer * self.steer_to_angle# 70 degrees in radians
        self.beta = math.atan(0.5 * math.tan(delta))
        input = np.array([[[self.xi, self.beta]]])
        input = input.astype(np.float32)
        if self.r > self.barrier()+ 0.5:  # 0.6
            return control, 0, False
        elif self.r > self.barrier() + 0.4:     # 0.5
            output = self.tf_model3.predict(input)[0][0][0]
        elif self.r > self.barrier() + 0.2:     # 0.25
            output = self.tf_model2.predict(input)[0][0][0]
        else:
            output = self.tf_model1.predict(input)[0][0][0]
        delta_new = math.atan(2 * math.tan(output))
        new_control = carla.VehicleControl()
        new_control.brake = control.brake
        new_control.gear = control.gear
        new_control.hand_brake = control.hand_brake
        new_control.manual_gear_shift = control.manual_gear_shift
        new_control.reverse = control.reverse
        new_control.steer = float(delta_new/self.steer_to_angle)
        new_control.throttle = control.throttle
        steer_diff = abs(delta_new - delta)
        return new_control, steer_diff, True

    def set_filter_inputs(self, xi, r):
        self.r = r
        self.xi = xi
    

