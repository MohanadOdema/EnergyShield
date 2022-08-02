import pandas as pd
import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

parser = argparse.ArgumentParser(description="Script to plot energy and missed deadlines across deadlines for different policies and other fixed hyperparameters")

parser.add_argument("--model_name", type=str, default='BasicAgent', help="Name of the model to train. Output written to models/model_name", choices=['agent1', 'agent2', 'agent3', 'BasicAgent', 'BehaviorAgent'])
parser.add_argument("--len_route", type=str, default='short', help="The route array length -- longer routes support more obstacles but extends sim time")
parser.add_argument("--len_obs", type=int, default=1, help="How many objects to be spawned given len_route is satisfied")
parser.add_argument("--img_resolution", type=str, help="enter offloaded image resolution", choices=['480p', '720p', '1080p', 'Radiate', 'TeslaFSD', 'Waymo'], default='Radiate')
parser.add_argument("--arch", type=str, help="Name of the model running on the AV platform", choices=['ResNet18', 'ResNet50', 'DenseNet169', 'ViT', 'ResNet18_mimic', 'ResNet50_mimic', 'DenseNet169_mimic'], default='ResNet50')
parser.add_argument("--offload_position", type=str, help="Offloading position", choices=['direct', '0.5_direct', '0.25_direct', 'bottleneck'], default='bottleneck')
parser.add_argument("--offload_policy", type=str, help="Offloading policy", choices=['local', 'offload', 'offload_failsafe', 'adaptive', 'adaptive_failsafe', 'all'], default='all')    
parser.add_argument("--HW", type=str, help="AV Hardware", choices=['PX2', 'TX2', 'Orin', 'Xavier', 'Nano'], default='PX2')
parser.add_argument("--primary_axis", type=str, default='avg_energy')
parser.add_argument("--secondary_axis", type=str, default='missed_deadlines')

params = vars(parser.parse_args())

stats = {}
deadlines_list = []
stats['avg_energy']= {}
stats['avg_latency'] = {}
stats['missed_deadlines'] ={}
stats['missed_deadlines'] = {}
stats['max_succ_interrupts'] = {}
stats['missed_offloads'] = {}
stats['misguided_energy'] = {}

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
# sweep over number of obstacles
if params['len_obs'] == 'all':
    len_obstacles = ['1','2','3','4','5']
else:
    len_obstacles = [str(params['len_obs'])]

for len_obs in len_obstacles:
    for resolution in resolutions:
        for architecture in architectures:
            for policy in policies:
                # initialize nested dictionaries for each policy
                stats['avg_energy'][policy] = {}
                stats['avg_latency'][policy] = {}
                stats['missed_deadlines'][policy] = {}
                stats['max_succ_interrupts'][policy] = {}
                stats['missed_offloads'][policy] = {}
                stats['misguided_energy'][policy] = {}
                for position in positions:
                    deadlines_list = []
                    params['arch'] = params['arch'][:8]     # prefix string (e.g., ResNet18)
                    if 'mimic' not in params['arch'] and 'bottleneck' in params["offload_position"]:
                        params['arch'] = params['arch'] + '_mimic'
                    base_path = '/home/mohanadodema/shielding_offloads/models/' + params['model_name'] + '/experiments/obs_' + \
                                len_obs + '_route_' + params['len_route'] + '/' + resolution + '/' + params['arch'] + '/' + policy + \
                                '/' + position + '/'
                    for subdir in os.listdir(base_path):
                        deadline = float(subdir[4:])
                        # deadlines_list.append(deadline)
                        # print(policy, deadline)
                        stats_file_path = base_path + subdir + '/valid_data.csv'
                        df = pd.read_csv(stats_file_path)
                        # record the first entry in the csv file regardless of the episode
                        print(policy, deadline)
                        stats['avg_energy'][policy][deadline] = round(df['avg_energy'].tolist()[0], 2)   
                        stats['avg_latency'][policy][deadline] = round(df['avg_latency'].tolist()[0], 2) 
                        stats['missed_deadlines'][policy][deadline] = round(df['missed_deadlines'].tolist()[0], 2) 
                        stats['max_succ_interrupts'][policy][deadline] = round(df['max_succ_interrupts'].tolist()[0], 2)    
                        stats['missed_offloads'][policy][deadline] = round(df['missed_offloads'].tolist()[0], 2)    
                        stats['misguided_energy'][policy][deadline] = round(df['misguided_energy'].tolist()[0], 2)     

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rcParams["font.family"] = "Times New Roman"

fig, ax = plt.subplots()

# # ax.set_facecolor('whitesmoke')
ax.grid(linestyle='dotted', zorder = 0)

# TODO: Coloring and working on axis string
ax.set_ylabel(params['primary_axis'] + '_per_frame (mJ)', fontsize=10, color='#02428F')
ax.set_xlabel('Deadlines (ms)', fontsize=10)
ax.set_title(params['arch']+'_'+params['img_resolution'], fontsize=12)
ax.tick_params(axis='y', labelcolor='#02428F')

ax2 = ax.twinx()
ax2.set_ylabel(params['secondary_axis'], color='black', fontsize = 10)
ax2.tick_params(axis='y', labelcolor='black')

for dictionary in stats[params['primary_axis']].keys():
    primary_values = []
    secondary_values = []
    deadlines = []
    for key,value1 in stats[params['primary_axis']][dictionary].items():
        value2 = stats[params['secondary_axis']][dictionary][key]
        deadlines.append(key)
        primary_values.append(value1)
        secondary_values.append(value2)
    deadlines, primary_values, secondary_values = (list(t) for t in zip(*sorted(zip(deadlines, primary_values, secondary_values)))) # sort as tuples
    x1 = ax.plot(deadlines, primary_values, '-', label=dictionary, linewidth=2, zorder=2)#, color='')
    x2 = ax2.plot(deadlines, secondary_values, '-.', label=dictionary, linewidth=2, zorder=3, alpha=0.7)

    print(deadlines)
    print(primary_values)
    print(secondary_values)

ax.legend(bbox_to_anchor=(0.2, 0.99), ncol=5)

plt.show()

#TODO: automatice save in .svg
