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
EXPERIMENTS_dir = '/home/gerard/Documents/EXPERIMENTS/SVMRank/spareAD_cuignendataset_draftpaper/2018-01-10_11-58-04/'
test_file_results = 'cv_results.csv'
train_file_results = 'test_results.csv'
score_results_file = 'data_scores.csv'

# Try different types of plots
cv_results = pd.read_csv(EXPERIMENTS_dir + test_file_results)
test_results = pd.read_csv(EXPERIMENTS_dir + train_file_results)
score_results = pd.read_csv(EXPERIMENTS_dir + score_results_file)

# Plot the different points
sns.set(style="white")
colors = ["faded green", "ochre", "pumpkin orange", "reddish"]
pal = sns.xkcd_palette(colors)

# Regression plot
g = sns.lmplot(x="AGE", y="scores", hue="DX", palette=pal, row_order=['CN', 'MCInc', 'MCIc', 'AD'], hue_order=['CN', 'MCInc', 'MCIc', 'AD'],
               truncate=True, size=8, data=score_results)
plt.savefig(EXPERIMENTS_dir + 'figures/' + 'regplot.png')
plt.show()
plt.close()
# Grid Plot
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
# Select adecuate colors from xkcd pallete ()
colors = ["faded green", "ochre", "pumpkin orange", "reddish"]
pal = sns.xkcd_palette(colors)
g = sns.FacetGrid(score_results, row="DX", row_order=['CN', 'MCInc', 'MCIc', 'AD'], hue_order=['CN', 'MCInc', 'MCIc', 'AD'], hue="DX", aspect=10, size=1, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "scores", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
g.map(sns.kdeplot, "scores", clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)
# Define and use a simple function to label the plot in axes coordinates


def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "scores")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)

# Remove axes details that don't play will with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)
plt.savefig(EXPERIMENTS_dir + 'figures/' + 'gridplot.png')
