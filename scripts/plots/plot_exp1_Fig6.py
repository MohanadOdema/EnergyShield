import types
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# Analysis for cont local mode

def obs_y_coordinate(x_coordinate):
    return (-95.715*x_coordinate) + 38737.9974

def obs_x_coordinate(y_coordinate_list):
    l = []
    for y in y_coordinate_list:
        l.append((y-38737.9974)/-95.715)
    return l

plt.rcParams["font.family"] = "Times New Roman"

parser = argparse.ArgumentParser(description="compute stats for an excel file")

parser.add_argument("--model_name", type=str, default='casc_agent_1', help="Name of the model")
parser.add_argument("-safety_filter", action="store_true", default=False, help="Filter Control actions")
parser.add_argument("-gaussian", action="store_true", default=False, help="Randomize obstacles location using gaussian distribution")
parser.add_argument("--offload_policy", type=str, help="Offloading policy", choices=['local', 'offload', 'offload_failsafe', 'adaptive', 'adaptive_failsafe', 'Shield1', 'Shield2'], default='offload')    
parser.add_argument("--phi_scale", type=int, default=20, help="scale parameter for the channel capacity pdf")
parser.add_argument("--deadline", type=int, help="dealdine in ms", default=100)                    # Single time window is 20 ms
parser.add_argument("--len_obs", type=int, default=4, help="How many objects to be spawned given len_route is satisfied")
parser.add_argument("--local_belay", action='store_true', default=False, help="belay local execution until the last execution window of the dealdine")
parser.add_argument("--local_early", action='store_true', default=False, help="instantly perform local execution at the first attainable window of the dealdine")
parser.add_argument("--off_belay", action='store_true', default=False, help="belay till delta_T expires to resume processing or execute local at the last attainable window" )  
parser.add_argument("--file_type", type=str, default='valid', help="The csv file to load")      
parser.add_argument("--len_route", type=str, default='short', help="The route array length -- longer routes support more obstacles but extends sim time")
parser.add_argument("--map", type=str, default='80p', help="80p, Town04, or Town04_OPT")

params = vars(parser.parse_args())

prefix = './models/'+params['model_name']+'/experiments/obs_4_route_short/Town04_OPT_ResNet152_local_cont/'

if params['model_name'] == 'casc_agent_1':
    trajectory_unsafe = prefix + "PX2_100_Safety_False_noise_False/plots/train_1754.csv"    
    trajectory_safe_right = prefix + "PX2_100_Safety_True_noise_False/plots/train_1747.csv"      
    trajectory_safe_left = prefix + "PX2_100_Safety_True_noise_True/plots/train_1753.csv"

else:     # random file for demonstration
    trajectory_unsafe = glob.glob(prefix + "PX2_100_Safety_False_noise_False/plots/*.csv")[0]    
    trajectory_safe_right = glob.glob(prefix + "PX2_100_Safety_True_noise_False/plots/*.csv")[0]     
    trajectory_safe_left = glob.glob(prefix + "PX2_100_Safety_True_noise_True/plots/*.csv")[0]

df_unsafe = pd.read_csv(trajectory_unsafe)
df_safe_right = pd.read_csv(trajectory_safe_right)
df_safe_left = pd.read_csv(trajectory_safe_left)

route_x = pd.read_csv("x_route_" + params['len_route'] + ".csv")
route_y = pd.read_csv("y_route_" + params['len_route'] + ".csv")

ego_x_unsafe = df_unsafe['ego_x']           # False_False_1754 
ego_y_unsafe = df_unsafe['ego_y']
ego_x_safe_right = df_safe_right['ego_x']   # True_False_1747
ego_y_safe_right = df_safe_right['ego_y']
ego_x_safe_left = df_safe_left['ego_x']     # True_True_1753
ego_y_safe_left = df_safe_left['ego_y']

plt.figure(figsize=(4,4))
# plt.locator_params(nbins=4) 

obstacles = []

y_first_figure = [-167, -188, -194, -205]
x_first_figure = obs_x_coordinate(y_first_figure)
x_second_figure = [407.4, 407.8, 407.68, 408.1] # 1753
y_second_figure = [-162, -180, -202.5, -209]
# x_second_figure = [406.7, 404.5, 409.8, 408] # 1781
# y_second_figure = [-161, -182, -202.5, -219]

plt.ylabel("y-coordinates", fontsize=14)
plt.xlabel("x-coordinates", fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.plot(route_x, route_y, color='#2F5061', linewidth=2.5, linestyle='-.')

if params['gaussian']: 
    plt.plot(ego_x_safe_left, ego_y_safe_left, color='#4297A0', linewidth=2, linestyle='-')
    plt.plot(x_second_figure, y_second_figure, color='#AA1945', linewidth=0, marker='_', markersize=70)
    print('here')
else:    
    plt.xlim(405, 413)
    plt.ylim(-240, -120)
    plt.plot(ego_x_unsafe, ego_y_unsafe, linewidth=2.5, color='#870A30', linestyle='-')
    plt.plot(x_first_figure, y_first_figure, color='#AA1945', linewidth=0, marker='_', markersize=70)
    plt.plot(ego_x_safe_right, ego_y_safe_right, color='#4297A0', linewidth=2.5, linestyle='-')

print(route_x)
print(route_y)

plt.tight_layout()
plt.savefig('./results/Fig6_traj_noise_' +str(params['gaussian']) + '.pdf', bbox_inches='tight')