import csv
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

def track_completion_rate(obs_list, curb_list):
	total = len(obs_list)
	incomplete = 0
	for i,j in zip(obs_list, curb_list):
		if i == True or j == True:
			incomplete += 1
	TCR = (1 - (incomplete/total))*100
	return round(TCR,2)

# This is only for the local execution mode

if not os.path.exists('../results/raw_data'):
   os.makedirs('../results/raw_data')

offloading_modes = ['local_cont', 'Shield2_early', 'Shield2_belay'] 
Safety_filter = ['False','True']
Gaussian_Noise = ['False','True']

parser = argparse.ArgumentParser(description="Parse experiment metadata")

parser.add_argument("--model_name", type=str, default='casc_agent_1', help="Name of the model")
parser.add_argument("-safety_filter", action="store_true", default=False, help="Filter Control actions")
parser.add_argument("-gaussian", action="store_true", default=False, help="Randomize obstacles location using gaussian distribution")
parser.add_argument("--offload_policy", type=str, help="Offloading policy", choices=['local', 'offload', 'offload_failsafe', 'adaptive', 'adaptive_failsafe', 'Shield1', 'Shield2'], default='Shield2')    
parser.add_argument("--phi_scale", type=int, default=20, help="scale parameter for the channel capacity pdf")
parser.add_argument("--deadline", type=int, help="dealdine in ms", default=100)                    # Single time window is 20 ms
parser.add_argument("--len_obs", type=int, default=4, help="How many objects to be spawned given len_route is satisfied")
parser.add_argument("--local_belay", action='store_true', default=False, help="belay local execution until the last execution window of the dealdine")
parser.add_argument("--local_early", action='store_true', default=False, help="instantly perform local execution at the first attainable window of the dealdine")
parser.add_argument("--off_belay", action='store_true', default=False, help="belay till delta_T expires to resume processing or execute local at the last attainable window" )  
parser.add_argument("--file_type", type=str, default='valid', help="The csv file to load")      
parser.add_argument("--len_route", type=str, default='short', help="The route array length -- longer routes support more obstacles but extends sim time")
parser.add_argument("--map", type=str, default='Town04_OPT', help="80p, Town04, or Town04_OPT")
parser.add_argument("--queue_state", type=int, default=None, help='Approximation to set number of tasks in a queue')
parser.add_argument("--start_idx", type=int, default=None, help='if a range is wanted from the excel file')
parser.add_argument("--end_idx", type=int, default=None, help='if a range is wanted from the excel file')
params = vars(parser.parse_args())

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

if params['phi_scale'] != 20:
    phi_string = '_'+str(params['phi_scale'])+'Mbps'
else:
    phi_string = ''

if params['queue_state'] != None:
    q_string = '_queue_'+str(params['queue_state'])+'_'
else:
    q_string = ''


### Retrieve CSV files onto dict
df_dict = {}
results = {}
for mode in offloading_modes:
    for filter in Safety_filter:
        for noise in Gaussian_Noise:
            df_dict[mode+'_'+filter+'_'+noise] = \
                 pd.read_csv(os.path.join("./models", params['model_name'], "experiments", "obs_"+str(params['len_obs'])+"_route_"+str(params['len_route']), 
                              str(params['map'])+"_ResNet152_"+mode+q_string+phi_string, "PX2_"+str(params['deadline'])+"_Safety_"+filter+"_noise_"+noise, str(params['file_type'])+"_data.csv"))

### Energy Plots 

'''
Original Reference Values
energyShield = [90.8, 90, 85.8, 85.9] 
energyShield_uniform = [67.7, 60.6, 51.5, 53.1]
'''

Shield_eager_energy = [np.mean(df_dict['Shield2_early_False_False']['avg_energy']), 
                        np.mean(df_dict['Shield2_early_False_True']['avg_energy']), 
                        np.mean(df_dict['Shield2_early_True_False']['avg_energy']), 
                        np.mean(df_dict['Shield2_early_True_True']['avg_energy'])]

Shield_belay_energy = [np.mean(df_dict['Shield2_belay_False_False']['avg_energy']), 
                        np.mean(df_dict['Shield2_belay_False_True']['avg_energy']), 
                        np.mean(df_dict['Shield2_belay_True_False']['avg_energy']), 
                        np.mean(df_dict['Shield2_belay_True_True']['avg_energy'])]

plt.rcParams["figure.figsize"]=(5,3.2)
plt.rcParams["font.family"] = "Times New Roman"

fig, ax = plt.subplots()
# ax.set_facecolor('#EBEBEB')
# [ax.spines[side].set_visible(False) for side in ax.spines]
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.locator_params(nbins=3) 
ax.set_ylabel('Energy (mJ)', fontsize=14)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.tick_params(labelsize=13)
ax.axhline(y=113.5, color='#821D30')  # Local is always constant

x = np.array([-0.3,0.8,1.9,3])
width = 0.23

ax.bar(x-0.13, Shield_eager_energy, width, color='#140005', edgecolor='#FFFFFF', alpha=0.85, linewidth=0, hatch='', zorder=3) 		#176ab6 #145DA0
ax.bar(x+0.13, Shield_belay_energy, width, color='#FFFFFF', edgecolor='#140005', alpha=0.85, linewidth=0.5, hatch='//', zorder=3) 		#2E8BC0 #EEEDE7

#B1D4E0
ax.set_xticks(x, ['(S=0, N=0)', '(S=0, N=1)', '(S=1, N=0)', '(S=1, N=1)'], rotation=14)
ax.set_ylim([45, 125])
ax.legend(["Local", "Eager", "Uniform"], bbox_to_anchor=(0.5, 1.4), 
           loc='upper center', ncol=3, borderaxespad=5, fontsize=13, borderpad=0.5, frameon=False)
fig.tight_layout()
plt.savefig('../results/Fig5_Energy.pdf', bbox_inches='tight')

### Safety Plots

'''
Original Reference Values:
    default_TCR = [65.7, 100]
    noisy_TCR = [22.9, 100]
    default_reward = [992.2, 1123.9]
    noisy_reward = [703.9, 1133.9]
    norm_default_reward = [992.2/1387.9, 1123.9/1387.9]
    norm_noisy_reward = [703.9/1387.9, 1133.9/1387.9]
'''
configs = ['Local', 'EnergyShield']

default_TCR = [track_completion_rate(df_dict['local_cont_False_False']['obstacle_hit'], df_dict['local_cont_False_False']['curb_hit']),
               track_completion_rate(df_dict['local_cont_True_False']['obstacle_hit'], df_dict['local_cont_True_False']['curb_hit'])]
noisy_TCR = [track_completion_rate(df_dict['local_cont_False_True']['obstacle_hit'], df_dict['local_cont_False_True']['curb_hit']),
               track_completion_rate(df_dict['local_cont_True_True']['obstacle_hit'], df_dict['local_cont_True_True']['curb_hit'])]
norm_default_reward = [np.mean(df_dict['local_cont_False_False']['reward'])/1387.9, np.mean(df_dict['local_cont_True_False']['reward'])/1387.9]
norm_noisy_reward = [np.mean(df_dict['local_cont_False_True']['reward'])/1387.9, np.mean(df_dict['local_cont_True_True']['reward'])/1387.9]

plt.rcParams["figure.figsize"]=(4.2,3.2)
plt.rcParams["font.family"] = "Times New Roman"

fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax.locator_params(nbins=4) 
ax2.locator_params(nbins=4) 

ax.grid(axis='y', alpha=0.3, zorder=0)
# ax.minorticks_on()
ax.tick_params(labelsize=13)
ax2.tick_params(labelsize=13)

x = np.array([1,2])
width = 0.23


ax.bar(x-.13, default_TCR, width, color='#7D3F20', alpha=0.8, linewidth=0, zorder=2) #A49393
ax.bar(x+.13, noisy_TCR, width, color='#FFFFFF', edgecolor='#171515', alpha=0.65,  linewidth=0.5, hatch='/', zorder=2) #E8B4B8

ax2.plot(x-.13, norm_default_reward, color='#7D3F20', linewidth=1.5, zorder=3) #A49393
ax2.plot(x+.13, norm_noisy_reward, color='#171515', linewidth=1.5, zorder=3) #E8B4B8

#B1D4E0
ax.set_xticks(x, ['S=0', 'S=1'], rotation=15)
ax.set_xlim([0.5,2.5])
ax.set_ylabel("TCR (%) (barplot) ", fontsize=14)
ax2.set_ylabel("Reward (lineplot)", fontsize=14)
# ax.set_ylim([20, 100])
plt.legend(["N=0", "N=1"], bbox_to_anchor=(-0.33, 1.4), 
           loc='upper left', ncol=1, borderaxespad=5, fontsize=13, borderpad=0.5, frameon=False)
fig.tight_layout()
plt.savefig('../results/Fig5_Safety.pdf', bbox_inches='tight')
