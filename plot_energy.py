import csv
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"]=(5,3.2)
plt.rcParams["font.family"] = "Times New Roman"

energyShield = [90.8, 90, 85.8, 85.9]
energyShield_uniform = [67.7, 60.6, 51.5, 53.1]

fig, ax = plt.subplots()
# ax.set_facecolor('#EBEBEB')
# [ax.spines[side].set_visible(False) for side in ax.spines]
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.locator_params(nbins=3) 
ax.set_ylabel('Energy (mJ)', fontsize=14)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.tick_params(labelsize=13)
ax.axhline(y=113.5, color='#821D30')


# colors = ['#0C2D48', '#145DA0', '#2E8BC0', '#B1D4E0']       # bluish 
colors = ['#145DA0', '#145DA0', '#145DA0', '#B1D4E0', '#B1D4E0', '#B1D4E0']

# colors = ['#000000', '#7D3F20', '#D19A30', '#F3CC3C']
# colors = ['#7D3F20', '#7D3F20', '#7D3F20', '#D19A30', '#D19A30', '#D19A30']
colors = ['#122620','#122620','#122620', '#8CA2B0', '#8CA2B0','#8CA2B0']

# colors = ['#140005', '#464033', '#7E7C73', '#BBC4C2'] black bird
# colors = ['#6F5B3E', '#F9F6F0', '#C4AE78', '#171515'] gingerbread icecream

x = np.array([-0.3,0.8,1.9,3])
print(x)
width = 0.23

ax.bar(x-0.13, energyShield, width, color='#140005', edgecolor='#FFFFFF', alpha=0.85, linewidth=0, hatch='', zorder=3) 		#176ab6 #145DA0
# ax.bar(x+0.13, energyShield_uniform, width, color='#7E7C73', edgecolor='#FFFFFF', alpha=0.85, linewidth=0, hatch='//', zorder=3) 		#2E8BC0 #EEEDE7
ax.bar(x+0.13, energyShield_uniform, width, color='#FFFFFF', edgecolor='#140005', alpha=0.85, linewidth=0.5, hatch='//', zorder=3) 		#2E8BC0 #EEEDE7

#B1D4E0
ax.set_xticks(x, ['(S=0, N=0)', '(S=0, N=1)', '(S=1, N=0)', '(S=1, N=1)'], rotation=14)
ax.set_ylim([45, 125])
ax.legend(["Local", "Eager", "Uniform"], bbox_to_anchor=(0.5, 1.4), 
           loc='upper center', ncol=3, borderaxespad=5, fontsize=13, borderpad=0.5, frameon=False)
fig.tight_layout()
plt.savefig('./plot_data/Energy.svg', bbox_inches='tight')

plt.show()