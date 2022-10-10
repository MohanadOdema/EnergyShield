#!/usr/bin/env python

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Example Command: python play_recording.py --recorder-filename /home/mohanadodema/shielding_offloads/models/BasicAgent/experiments/obs_3_route_medium/Radiate/ResNet18_mimic/adaptive/bottleneck/PX2_140/logs/0_log.log --camera 1
# (Need to have exact actor ids as camera arguments)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

import glob
import os
import sys

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
    args = argparser.parse_args()

    args.filename = '/home/mohanadodema/EnergyShield/models/casc_agent3/experiments/obs_4_route_short/80p_ResNet152_local_cont/PX2_20_Safety_True_noise_False/logs/1645_log.log'

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        # set the time factor for the replayer
        client.set_replayer_time_factor(args.time_factor)

        # set to ignore the hero vehicles or not
        client.set_replayer_ignore_hero(args.ignore_hero)

        # replay the session
        print(client.replay_file(args.filename, args.start, args.duration, args.camera))
    finally:
        pass

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
