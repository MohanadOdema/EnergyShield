import os
import numpy as np
import pandas as pd 
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"]=(6,4)

parser = argparse.ArgumentParser(description="compute stats for an excel file")

parser.add_argument("--model_name", type=str, default='casc_agent4', help="Name of the model")
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

safety_filter = ['False', 'True']
gaussian_noise = ['False', 'True']

energy_plt_dict = {}
keys_plt_dict = {}
instances_plt_dict = {}

for safety in safety_filter:
    for noise in gaussian_noise:
        load_data_path = "./plot_data/distance/Hist_Shield2_"+sup_string+"_Safety_"+str(safety)+"_noise_"+str(noise)+".csv"
        df = pd.read_csv(load_data_path)
        keys_plt_dict[safety+'/'+noise] = df['keys']
        energy_plt_dict[safety+'/'+noise] = df['avg_energy']
        print(energy_plt_dict[safety+'/'+noise])
        instances_plt_dict[safety+'/'+noise] = df['instances']


# colors = ['#D6B3A1', '#D69472', '#D6703B', '#D35310']     # yellowish original
# colors = ['#122620', '#B68D40', '#D6AD60', '#F4EBD0']       # emerald entrance
# colors = ['#000000', '#7D3F20', '#D19A30', '#F3CC3C']     # Fall fashion
# colors = ['#96BED3', '#8ABCD6', '#4FA8D5', '#1093D5']     # bluish original
if not params['off_belay']:
    colors = ['#0C2D48', '#145DA0', '#2E8BC0', '#B1D4E0']       # bluish 
else:
    colors = ['#000000', '#7D3F20', '#D19A30', '#F3CC3C']

fig, ax = plt.subplots()
ax.locator_params(nbins=4) 
ax.grid(zorder = 0, alpha=0.3)
# ax.set_xticks(pos+2*(3/4)*width)
# ax.set_xticklabels(index, minor=False, fontsize = 11.5)
# # ax.set_yticklabels(fontsize = 12)

ax.tick_params(labelsize=13)

# ax.set(xlim=[0, 1.35])
# ax.set(ylim=ylims)

ax.set_xlabel('Distance from Obstacle (m)', fontsize = 14)
ax.set_ylabel('Normalized Energy', fontsize = 14)  

ax.plot(keys_plt_dict['False/False'], energy_plt_dict['False/False'], color=colors[0], label='S/N: F/F', linewidth=2)
ax.plot(keys_plt_dict['False/True'], energy_plt_dict['False/True'], color=colors[1], label='S/N: F/T', linewidth=2)
ax.plot(keys_plt_dict['True/False'], energy_plt_dict['True/False'], color=colors[2], label='S/N: T/F', linewidth=2)
ax.plot(keys_plt_dict['True/True'], energy_plt_dict['True/True'], color=colors[3], label='S/N: T/T', linewidth=2)

if not params['off_belay']:
    ax.axvline(x=2, color='black', linewidth=1, linestyle='-.')
ax.axvline(x=3, color='black', linewidth=1, linestyle='-.')

xticks = ax.get_xticks()
ax.set_xticks(xticks[::len(xticks) // 10]) # set new tick positions
ax.tick_params(axis='x', rotation=0) # set tick rotation
ax.margins(x=0) # set tight margins

ax.legend(loc='upper right', bbox_to_anchor=(1, 1),
         ncol=1, frameon = False, fontsize = 13)

fig.tight_layout()

plt.savefig('./plot_data/distance/off_belay_' +str(params['off_belay']) + '.svg', bbox_inches='tight')
plt.show()
# plt.bar(norm_energy_dict.keys(), 
