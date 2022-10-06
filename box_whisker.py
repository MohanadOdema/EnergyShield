# Box and whisker plot showing the energy/fidelity(reward) trade-offs, with the x-axis being the controller id (4 plots indicating different scenarios gaussian, safety filter, combinations)

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import argparse

plt.rcParams["figure.figsize"]=(4.2,2.2)

parser = argparse.ArgumentParser(description='Box and Whisker Plot for energy/Fidelity')
parser.add_argument('--metric', default='energy', type=str, choices=['energy', 'reward'])
parser.add_argument('--controller', default='casc_agent1', type=str)

args = parser.parse_args()
# edit from here

root = '../../eval/bar_plots_csv/'


# file = root + str(args.arch) + '_fail_' + str(args.fail) + '_lambda_' + str(args.trade_off) + '_LTE.csv'

file1 = root + str(args.arch) + '_fail_1_lambda_' + str(args.trade_off) + '_LTE.csv'
file10 = root + str(args.arch) + '_fail_10_lambda_' + str(args.trade_off) + '_LTE.csv'
file20 = root + str(args.arch) + '_fail_20_lambda_' + str(args.trade_off) + '_LTE.csv'

df1 = pd.read_csv(file1)
df10 = pd.read_csv(file10)
df20 = pd.read_csv(file20)

if args.arch == '50':
	# colors = ['#D6B3A1', '#D69472', '#D6703B', '#D35310']		# yellowish original
	colors = ['#122620', '#B68D40', '#D6AD60', '#F4EBD0']		# emerald entrance
	# colors = ['#000000', '#7D3F20', '#D19A30', '#F3CC3C']		# Fall fashion
	ylims = [0.6, 1.05]
elif args.arch == '18':	
	# colors = ['#96BED3', '#8ABCD6', '#4FA8D5', '#1093D5']		# bluish original
	colors = ['#0C2D48', '#145DA0', '#2E8BC0', '#B1D4E0']		# bluish 
	ylims = [0.75, 1.1]


# Number of offload_fe, full_edge, offload_eex1, eex1
local_counts = [0, 1, 0, 0] 		
drl1_count = df1['DRL_count']
sage1_count = df1['Sage_count']
opt1_count = df1['Opt_count']

drl10_count = df10['DRL_count']
sage10_count = df10['Sage_count']
opt10_count = df10['Opt_count']

drl20_count = df20['DRL_count']
sage20_count = df20['Sage_count']
opt20_count = df20['Opt_count']

#Attempted Failed Offloads
drl1_fail = df1['DRL_fail_off']
sage1_fail = df1['Sage_fail_off']

drl10_fail = df10['DRL_fail_off']
sage10_fail = df10['Sage_fail_off']

drl20_fail = df20['DRL_fail_off']
sage20_fail = df20['Sage_fail_off']


#I need the energies; for that I need to construct a dictionary energy_dict, and include the energy for local, opt, sage, and DRL
energy_dict['df1_drl'] = np.mean(df1['energy_DRL'])
energy_dict['df1_sage'] = np.mean(df1['energy_SAGE'])
energy_dict['df1_opt'] = np.mean(df1['energy_OPT'])
energy_dict['df1_local'] = np.mean(df1['energy_EDGE'])

energy_dict['df10_drl'] = np.mean(df10['energy_DRL'])
energy_dict['df10_sage'] = np.mean(df10['energy_SAGE'])
energy_dict['df10_opt'] = np.mean(df10['energy_OPT'])
energy_dict['df10_local'] = np.mean(df10['energy_EDGE'])

energy_dict['df20_drl'] = np.mean(df20['energy_DRL'])
energy_dict['df20_sage'] = np.mean(df20['energy_SAGE'])
energy_dict['df20_opt'] = np.mean(df20['energy_OPT'])
energy_dict['df20_local'] = np.mean(df20['energy_EDGE'])

sage_ergy = [energy_dict['df1_sage'], energy_dict['df10_sage'], energy_dict['df20_sage']]
local_ergy = [energy_dict['df1_local'], energy_dict['df10_local'], energy_dict['df20_local']]
opt_ergy = [energy_dict['df1_opt'], energy_dict['df10_opt'], energy_dict['df20_opt']]
drl_ergy = [energy_dict['df1_drl'], energy_dict['df10_drl'], energy_dict['df20_drl']]

# print(sage_ergy)
# exit(-1)

normalized_sage_ergy = []
normalized_local_ergy = []
normalized_opt_ergy = []
normalized_drl_ergy = []

for local, sage, opt, drl in zip(local_ergy, sage_ergy, opt_ergy, drl_ergy): 	# all normalized with respect to local
	normalized_sage_ergy.append(sage/local)
	normalized_local_ergy.append(local/local)
	normalized_opt_ergy.append(opt/local)
	normalized_drl_ergy.append(drl/local)


index = ['1', '10', '20']


# fig = plt.figure()
# ax = fig.add_subplot(111)

plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots()

pos = np.array([0.1, .6, 1.1])
width = 0.05 # the width of the bars 

ax.locator_params(nbins=4)

ax.grid(axis='y', zorder = 0)

ax.set_xticks(pos+2*(3/4)*width)
ax.set_xticklabels(index, minor=False, fontsize = 11.5)
# ax.set_yticklabels(fontsize = 12)
ax.tick_params(axis='y', labelsize=11.5)

ax.set(xlim=[0, 1.35])
ax.set(ylim=ylims)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
         ncol=4, frameon = False, fontsize = 12)

ax.set_xlabel('Failure Rate (%)', fontsize = 12)
ax.set_ylabel('Normalized Energy', fontsize = 12)  

#ind = np.arange(len(index))  # the x locations for the groups
edge_hatch = ['oooooo', '......', 'oooo', 'oooooo']
edge_hatch = ['','','','']

ax.bar(pos, normalized_local_ergy, width, color=colors[0], label='Local', hatch = edge_hatch[0], edgecolor = 'black', zorder = 3)  				# color was white and edgecolor was the color 
ax.bar(pos+width, normalized_sage_ergy, width, color=colors[1], label='Sage', hatch = edge_hatch[1], edgecolor = 'black', zorder = 3)
ax.bar(pos+2*width, normalized_drl_ergy, width, color=colors[2], label='DRL (Ours)', hatch = edge_hatch[2], edgecolor = 'black', zorder = 3)
ax.bar(pos+3*width, normalized_opt_ergy, width, color=colors[3], label='Optimal', hatch = edge_hatch[3], edgecolor = 'black', zorder = 3)

if args.legend is True:
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
	         ncol=4, frameon = False, fontsize = 12)
	plt.show()
else:
	plt.savefig('C:/Users/Mohanad Odema/Desktop/Resnet' + str(args.arch) + '_' + str(args.trade_off) + '.svg', bbox_inches='tight')

# plt.show()