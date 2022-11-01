import os
import numpy as np
import pandas as pd 
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

plt.rcParams["font.family"] = "Times New Roman"

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

episodes_path = os.path.join("./models", params['model_name'], "experiments", "obs_"+str(params['len_obs'])+"_route_"+str(params['len_route']), str(params['map'])+"_ResNet152_"+str(params['offload_policy'])+"_"+sup_string+q_string+phi_string, 
                            "PX2_"+str(params['deadline'])+"_Safety_"+str(params['safety_filter'])+"_noise_"+str(params['gaussian']), "plots")

episode_summary_path = os.path.join("./models", params['model_name'], "experiments", "obs_"+str(params['len_obs'])+"_route_"+str(params['len_route']), str(params['map'])+"_ResNet152_"+str(params['offload_policy'])+"_"+sup_string+q_string+phi_string, 
                            "PX2_"+str(params['deadline'])+"_Safety_"+str(params['safety_filter'])+"_noise_"+str(params['gaussian']), str(params['file_type'])+"_data.csv")
save_data_path = "./plot_data/Hist_Shield2_"+sup_string+"_Safety_"+str(params['safety_filter'])+"_noise_"+str(params['gaussian'])+".csv"

episode_summary = pd.read_csv(episode_summary_path)
valid_episodes = list(episode_summary['episode_idx'])

# if not os.path.exists(save_data_path):
#     with open(valid_data_path, 'a', newline='') as fd:
#         csv_writer = csv.writer(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         csv_writer.writerow(['keys', 'instances', 'energy'])

n = 40  # 40 bins for 40m 
dist_count_dict = {}            # keys are the distance range in 1 m increments, indexed by the lower bound
norm_energy_dict = {}           # keys are the distance range in 1 m increments, indexed by the lower bound

for i in range(0,20):            # initialize 
    norm_energy_dict[str(i)] = []
norm_energy_dict['>=20'] = []
# norm_energy_dict['[10-20)'] = []

csv_files_raw = glob.glob(os.path.join(episodes_path, "*.csv"))

csv_files = []
for episode in valid_episodes:    # filter valid episodes
    for csv in csv_files_raw:
        if str(episode) in csv:
            csv_files.append(csv)
assert len(csv_files) == 35

# Histogram bins for energy
for i, f in enumerate(csv_files):
    # print(f)
    df = pd.read_csv(f)
    energy = df['exp_energy']
    distance = df['r']

    for r, ergy in zip(distance, energy):
        r = int(r)          # floor it
        if r >= 20:
            norm_energy_dict['>=20'].append(ergy)
        # elif r >= 10 and r < 20:
        #     norm_energy_dict['[10-20)'].append(ergy)  
        elif r < 20 and r >= 0:
            norm_energy_dict[str(r)].append(ergy)
        else:
            print(r, ergy)
            raise ValueError("r not supported!")

# for key, value in norm_energy_dict.items():
#     print(key, len(value))

keys_list = []
instances_list = []
avg_energy_list = []

for key, value in norm_energy_dict.items():
    keys_list.append(key)
    instances_list.append(len(value))
    avg_energy_list.append(round(np.mean(value) / 113.48,2))

df_save = pd.DataFrame({'keys': pd.Series(keys_list), 'instances':pd.Series(instances_list), 'avg_energy':pd.Series(avg_energy_list)})
df_save.to_csv("./plot_data/distance/Hist_Shield2_"+sup_string+"_Safety_"+str(params['safety_filter'])+"_noise_"+str(params['gaussian'])+".csv")


plt.step(keys_list, avg_energy_list, where='post')
plt.show()



 






