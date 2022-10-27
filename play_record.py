#!/usr/bin/env python

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Example Command: python play_recording.py --recorder-filename /home/mohanadodema/shielding_offloads/models/BasicAgent/experiments/obs_3_route_medium/Radiate/ResNet18_mimic/adaptive/bottleneck/PX2_140/logs/0_log.log --camera 1
# (Need to have exact actor ids as camera arguments)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

import glob
import os
import sys
import pandas
import csv

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse

def main():

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-s', '--start', metavar='S', default=0.0, type=float, help='starting time (default: 0.0)')
    argparser.add_argument('-d', '--duration', metavar='D', default=0.0, type=float, help='duration (default: 0.0)')
    argparser.add_argument('-f', '--recorder-filename', metavar='F', default="test1.log", help='recorder filename (test1.log)')
    argparser.add_argument('-c', '--camera', metavar='C', default=0, type=int, help='camera follows an actor (ex: 82)')
    argparser.add_argument('-x', '--time-factor', metavar='X', default=1.0, type=float, help='time factor (default 1.0)')
    argparser.add_argument('-i', '--ignore-hero', action='store_true', help='ignore hero vehicles')
    argparser.add_argument('--filename', type=str, help='file path for the recording')
    argparser.add_argument('--draw_path', action='store_true', help='draw trajectories')
    args = argparser.parse_args()

    # args.filename = '/home/mohanadodema/EnergyShield/models/casc_agent3/experiments/obs_4_route_short/80p_ResNet152_local_cont/PX2_20_Safety_True_noise_False/logs/1645_log.log'
    args.filename = '/home/mohanadodema/EnergyShield/models/casc_agent4/experiments/obs_4_route_short/Town04_OPT_ResNet152_local_cont/PX2_100_Safety_False_noise_False/logs/1754_log.log'
    args.filename = '/home/mohanadodema/EnergyShield/models/casc_agent4/experiments/obs_4_route_short/Town04_OPT_ResNet152_local_cont/PX2_100_Safety_True_noise_True/logs/1753_log.log'   # duration 7.3
    # args.filename = '/home/mohanadodema/EnergyShield/models/casc_agent4/experiments/obs_4_route_short/Town04_OPT_ResNet152_local_cont/PX2_100_Safety_True_noise_False/logs/1747_log.log'   # duration 8.3

    try:
        client = carla.Client(args.host, args.port)

        if args.draw_path:
            _draw_path(life_time=1000.0, skip=10)   

        client.set_timeout(60.0)

        # set the time factor for the replayer
        client.set_replayer_time_factor(args.time_factor)

        # set to ignore the hero vehicles or not
        client.set_replayer_ignore_hero(args.ignore_hero)

        # replay the session
        print(client.replay_file(args.filename, args.start, 8, args.camera))

        # print(client.show_recorder_actors(args.filename))


    finally:
        pass

def _draw_path(life_time=60.0, skip=0):
    """
        Draw a connected path from start of route to end.
        Green node = start
        Red node   = point along path
        Blue node  = destination
    """
    return
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
        carla.Color(0, 0, 255), life_time, False)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
