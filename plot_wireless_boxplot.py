import os
import numpy as np
import pandas as pd 
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

def normalize_energy(ergy_list):
    return [round(x/113.48, 2) for x in ergy_list]

def normalize_windows(windows_list, ticks_list):
    normalized_windows = []
    for w, t in zip(windows_list, ticks_list):
        normalized_windows.append(round(w/t,2)*100)
    return normalized_windows

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"]=(6,2.5)

parser = argparse.ArgumentParser(description="compute stats for an excel file")

parser.add_argument("--off_belay", action='store_true', default=False, help="belay till delta_T expires to resume processing or execute local at the last attainable window" )  
parser.add_argument("--file_type", type=str, default='valid', help="The csv file to load")  
parser.add_argument("--mode", type=str, default='energy', choices=['energy', 'windows'])    

params = vars(parser.parse_args())

if params['off_belay']:                # for offloading policies
    sup_string = 'belay'
else:
    sup_string = 'early'

# ALl safety filter True gaussian False
df_20Mbps_q1 = pd.read_csv(os.path.join("./models", "casc_agent4", "experiments", "obs_4_route_short", "Town04_OPT_ResNet152_Shield2_"+sup_string, 
                            "PX2_100_Safety_True_noise_False", "valid_data.csv"))
df_10Mbps_q1 = pd.read_csv(os.path.join("./models", "casc_agent4", "experiments", "obs_4_route_short", "Town04_OPT_ResNet152_Shield2_"+sup_string+"_10Mbps", 
                            "PX2_100_Safety_True_noise_False", "valid_data.csv"))
df_5Mbps_q1 = pd.read_csv(os.path.join("./models", "casc_agent4", "experiments", "obs_4_route_short", "Town04_OPT_ResNet152_Shield2_"+sup_string+"_5Mbps", 
                            "PX2_100_Safety_True_noise_False", "valid_data.csv"))
df_10Mbps_q9 = pd.read_csv(os.path.join("./models", "casc_agent4", "experiments", "obs_4_route_short", "Town04_OPT_ResNet152_Shield2_"+sup_string+"_queue_9__10Mbps", 
                            "PX2_100_Safety_True_noise_False", "valid_data.csv"))
df_10Mbps_q19 = pd.read_csv(os.path.join("./models", "casc_agent4", "experiments", "obs_4_route_short", "Town04_OPT_ResNet152_Shield2_"+sup_string+"_queue_19__10Mbps", 
                            "PX2_100_Safety_True_noise_False", "valid_data.csv"))
df_10Mbps_q49 = pd.read_csv(os.path.join("./models", "casc_agent4", "experiments", "obs_4_route_short", "Town04_OPT_ResNet152_Shield2_"+sup_string+"_queue_49__10Mbps", 
                            "PX2_100_Safety_True_noise_False", "valid_data.csv"))
save_data_path = "./plot_data/Hist_Shield2_"+sup_string+"_Safety_True_noise_gaussian.csv"

energy_dict = {} 
missed_windows = {} 
ticks = {}

energy_dict['20Mbps'] = normalize_energy(list(df_20Mbps_q1['avg_energy']))
energy_dict['10Mbps'] = normalize_energy(list(df_10Mbps_q1['avg_energy']))
energy_dict['5Mbps'] = normalize_energy(list(df_5Mbps_q1['avg_energy']))
energy_dict['q9'] = normalize_energy(list(df_10Mbps_q9['avg_energy']))
energy_dict['q19'] = normalize_energy(list(df_10Mbps_q19['avg_energy']))
energy_dict['q49'] = normalize_energy(list(df_10Mbps_q49['avg_energy']))

missed_windows['20Mbps'] = normalize_windows(list(df_20Mbps_q1['missed_windows']), list(df_20Mbps_q1['ticks']))
missed_windows['10Mbps'] = normalize_windows(list(df_10Mbps_q1['missed_windows']), list(df_10Mbps_q1['ticks']))
missed_windows['5Mbps'] = normalize_windows(list(df_5Mbps_q1['missed_windows']), list(df_5Mbps_q1['ticks']))
missed_windows['q9'] = normalize_windows(list(df_10Mbps_q9['missed_windows']), list(df_10Mbps_q9['ticks']))
missed_windows['q19'] = normalize_windows(list(df_10Mbps_q19['missed_windows']), list(df_10Mbps_q19['ticks']))
missed_windows['q49'] = normalize_windows(list(df_10Mbps_q49['missed_windows']), list(df_10Mbps_q49['ticks']))

labels_list = [r'$\phi$=20Mbps', r'$\phi$=10Mbps', r'$\phi$=5Mbps', 'q=10 ms', 'q=20 ms', 'q=50 ms']

mega_energy_list = [energy_dict['20Mbps'], energy_dict['10Mbps'], energy_dict['5Mbps'], energy_dict['q9'], energy_dict['q19'], energy_dict['q49']]
mega_missed_windows = [missed_windows['20Mbps'], missed_windows['10Mbps'], missed_windows['5Mbps'], missed_windows['q9'], missed_windows['q19'], missed_windows['q49']]

diamond = dict(markerfacecolor='black', marker='D',markersize=1)   # change outlier shape/color
fig1, ax1 = plt.subplots()
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.locator_params(nbins=5) 

# ax1.set_title('Basic Plot')
if params['mode'] == 'windows':
    ax1.set_ylabel('% Extra Transit Windows', fontsize=14)
else:
    ax1.set_ylabel('Normalized Energy', fontsize=14)
ax1.tick_params(labelsize=13)
if params['mode'] == 'windows':
    bplot = ax1.boxplot(mega_missed_windows, notch=True, flierprops=diamond, showfliers=True, patch_artist=True, labels=labels_list)#, whis=1)#, showfliers=False)
else:
    bplot = ax1.boxplot(mega_energy_list, notch=True, flierprops=diamond, showfliers=True, patch_artist=True, labels=labels_list)#, whis=1)#, showfliers=False)

if params['mode'] == 'windows':
    # colors = ['#0C2D48', '#145DA0', '#2E8BC0', '#B1D4E0']       # bluish 
    colors = ['#145DA0', '#145DA0', '#145DA0', '#B1D4E0', '#B1D4E0', '#B1D4E0']
else:  
    # colors = ['#000000', '#7D3F20', '#D19A30', '#F3CC3C']
    # colors = ['#7D3F20', '#7D3F20', '#7D3F20', '#D19A30', '#D19A30', '#D19A30']
    colors = ['#122620','#122620','#122620', '#8CA2B0', '#8CA2B0','#8CA2B0']
    for median in bplot['medians']:
        median.set_color('red')

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

for median in bplot['medians']:
    print(median.get_xydata()[1])

fig1.tight_layout()

ax1.set_xticks(ax1.get_xticks(), labels_list, rotation=7)



ax1.grid(axis='y', zorder = 0, alpha=0.3)
plt.savefig('./plot_data/wireless/'+str(params['mode'])+'off_belay_' +str(params['off_belay']) + '.svg', bbox_inches='tight')

plt.show()

# plt.bar(norm_energy_dict.keys(), 



 






