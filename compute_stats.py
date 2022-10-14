import os
import numpy as np
import pandas as pd 

import argparse

def compute_conditional_avg(main_list, condition_list1, condition_list2, condition=True):
	clean_list = [] 
	for i,j,k in zip(main_list,condition_list1, condition_list2):
		if j != condition and k != condition:
			clean_list.append(i)
	return round(np.mean(clean_list),2)

def track_completion_rate(obs_list, curb_list):
	total = len(obs_list)
	incomplete = 0
	for i,j in zip(obs_list, curb_list):
		if i == True or j == True:
			incomplete += 1
	TCR = (1 - (incomplete/total))*100
	return round(TCR,2)


parser = argparse.ArgumentParser(description="compute stats for an excel file")

parser.add_argument("--model_name", type=str, default='casc_agent3', help="Name of the model")
parser.add_argument("-safety_filter", action="store_true", default=False, help="Filter Control actions")
parser.add_argument("-gaussian", action="store_true", default=False, help="Randomize obstacles location using gaussian distribution")
parser.add_argument("--offload_policy", type=str, help="Offloading policy", choices=['local', 'offload', 'offload_failsafe', 'adaptive', 'adaptive_failsafe', 'Shield'], default='offload')    
parser.add_argument("--deadline", type=int, help="dealdine in ms", default=20)                    # Single time window is 20 ms
parser.add_argument("--len_obs", type=int, default=4, help="How many objects to be spawned given len_route is satisfied")
parser.add_argument("--local_belay", action='store_true', default=False, help="belay local execution until the last execution window of the dealdine")
parser.add_argument("--local_early", action='store_true', default=False, help="instantly perform local execution at the first attainable window of the dealdine")
parser.add_argument("--off_belay", action='store_true', default=False, help="belay till delta_T expires to resume processing or execute local at the last attainable window" )  
parser.add_argument("--file_type", type=str, default='valid', help="The csv file to load")      
parser.add_argument("--len_route", type=str, default='short', help="The route array length -- longer routes support more obstacles but extends sim time")

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


csv_file_path = os.path.join("./models", params['model_name'], "experiments", "obs_"+str(params['len_obs'])+"_route_"+str(params['len_route']), "80p_ResNet152_"+str(params['offload_policy'])+"_"+sup_string, 
							"PX2_"+str(params['deadline'])+"_Safety_"+str(params['safety_filter'])+"_noise_"+str(params['gaussian']), str(params['file_type'])+"_data.csv")


df = pd.read_csv(csv_file_path)

results_dict = {}

results_dict['TCR'] = track_completion_rate(df['obstacle_hit'], df['curb_hit'])
results_dict['avg_reward'] = round(np.mean(df['reward']),2)
results_dict['cond_avg_reward'] = compute_conditional_avg(df['reward'], df['obstacle_hit'], df['curb_hit'])
results_dict['avg_CD'] = round(np.mean(df['avg_center_deviance']),2)
results_dict['cond_avg_CD'] = compute_conditional_avg(df['avg_center_deviance'], df['obstacle_hit'], df['curb_hit'])
results_dict['avg_energy'] = round(np.mean(df['avg_energy']), 2)
results_dict['missed_windows'] = round(np.mean(df['missed_windows']),2)
results_dict['cond_missed_windows'] = round(compute_conditional_avg(df['missed_windows'], df['obstacle_hit'], df['curb_hit']),2)
results_dict['missed_deadlines'] = round(np.mean(df['missed_deadlines']), 2)
results_dict['cond_missed_deadlines'] = round(compute_conditional_avg(df['missed_deadlines'], df['obstacle_hit'], df['curb_hit']), 2)
results_dict['missed_controls'] = round(np.mean(df['missed_controls']),2)
results_dict['cond_missed_controls'] = round(compute_conditional_avg(df['missed_controls'], df['obstacle_hit'], df['curb_hit']),2)
results_dict['missed_offloads'] = round(np.mean(df['missed_offloads']),2)
results_dict['cond_missed_offloads'] = round(compute_conditional_avg(df['missed_offloads'], df['obstacle_hit'], df['curb_hit']),2)
results_dict['max_succ_interrupts'] = max(df['max_succ_interrupts'])
print("\n")
print("Stats for: ", csv_file_path)
print("\n")

for key,value in results_dict.items():
	print(key, value)






