"""
Small, standalone script to generate figures.

Given a directory from a results dataset, generate figures from there.
This script should be taken as a changing one, and is mean to be saved if we use
a figure into a paper or similar, to ensure reproducibility.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Hardcoded directories
EXPERIMENTS_dir = '/home/gerard/Documents/EXPERIMENTS/LADCE/lambda01/2018-01-19_11-35-08/'
bt_weights = np.load(EXPERIMENTS_dir + 'weights_bootstrapping.npy')
feat_name = ["VENTRICLES", "LHIPPOC", "RHIPPOC", "LINFLATVEN", "RINFLATVEN", "LMIDTEMP", "RMIDTEMP", "LINFTEMP", "RINFTEMP", "LFUSIFORM", "RFUSIFORM", "LENTORHIN", "RENTORHIN"]
# Hardcoded weights
i = 0
agerange = (91 - 54) / 4

bt_weights = np.squeeze(bt_weights)
print(np.array(bt_weights).shape)

sns.set(rc={'figure.figsize': (10, 18)})

for weightsmap in bt_weights:
    # Divide the data into the clusters
    l_age = 54 + agerange * i
    h_age = 54 + agerange * (i + 1)
    plt.figure()
    weightsmap = np.array(weightsmap)
    print(np.array(weightsmap).shape)
    xticks = range(np.shape(weightsmap)[0])
    xticks = [str(x) if i % 50 == 0 else '' for i, x in enumerate(xticks)]
    g = sns.heatmap(data=weightsmap.T, vmin=0, vmax=1,
                    yticklabels=feat_name, cmap="viridis", linewidths=0.0,
                    xticklabels=xticks, cbar=True)
    plt.title("Features for ages between " + str(l_age) + " and " +
              str(h_age) + ".")
    g.tick_params(axis='y', labelsize=7)
    g.tick_params(axis='both', left='off', bottom='off')

    plt.savefig(EXPERIMENTS_dir + 'figures/heatmap_' + str(i) + '.png', bbox_inches='tight')
    i = i + 1
    plt.close()
