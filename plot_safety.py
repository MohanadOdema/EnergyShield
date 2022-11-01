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

plt.rcParams["figure.figsize"]=(4.2,3.2)
plt.rcParams["font.family"] = "Times New Roman"


configs = ['Local', 'EnergyShield']

default_TCR = [65.7, 100]
noisy_TCR = [22.9, 100]
default_reward = [992.2, 1123.9]
noisy_reward = [703.9, 1133.9]
norm_default_reward = [992.2/1387.9, 1123.9/1387.9]
norm_noisy_reward = [703.9/1387.9, 1133.9/1387.9]

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
plt.savefig('./plot_data/Safety.svg', bbox_inches='tight')
plt.show()
#plt.savefig('./accuracy_motiv.pdf', dpi=300)


# plt.bar(x-.25, a6, width, color='#868B8E', edgecolor='#868e89', linewidth=0, hatch='\|\|\|\|\|\|\|') #'#67595E
# plt.bar(x, a0, width, color='#B9B7BD', edgecolor='#928f98', linewidth=0, hatch='o.o.o.o.o.o.o.') #A49393
# plt.bar(x+.25, ours, width, color='#E7D2CC', edgecolor='#EEEDE7', linewidth=0, hatch='......') #E8B4B8