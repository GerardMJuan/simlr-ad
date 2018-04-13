"""
Support functions to create plots and figures representing the data.

Ideally, in the future we would create a library to create the figures.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def draw_sim_matrix(S, fig_dir):
    sns.clustermap(S, col_cluster=True, row_cluster=True, robust=True,
                   method='average', metric='seuclidean', figsize=(20, 20))
    plt.savefig(fig_dir + "similarity.png")
    plt.close()


def draw_twodim(S, fig_dir):
    plt.scatter(S[:,0], S[:,1])
    plt.savefig(fig_dir + "twodim.png")
    plt.close()


def create_weight_maps(data, feat_name, file, name):
    """
    Create, a map of the weights, for the cluster

    For each age range, compute the variation of the weights.
    Add labels of the features and of the age ranges
    """
    # From bt_weights, if all the weights in a feature are 0, remove
    # that feature.
    # Select features that are always 0
    data = np.squeeze(data)
    k = 0
    sns.set(rc={'figure.figsize': (10, 18)})
    # Divide the data into the clusters
    plt.figure()
    xticks = range(np.shape(data)[0])
    g = sns.heatmap(data=data.T,
                    yticklabels=feat_name, cmap="viridis", linewidths=0.0,
                    xticklabels=xticks, cbar=True)
    plt.title("Weight maps")
    plt.yticks(rotation=30)
    g.tick_params(axis='y', labelsize=18)
    g.tick_params(axis='both', left='off', bottom='off')
    plt.savefig(file + 'figures/heatmap_' + name + '.png', bbox_inches='tight')
    k = k + 1
    plt.close()


def draw_space(space, clustercolor, file, dx):
    """ Create a two dimensional space with the coloring of the clusters"""

    # Cluster coloring
    for c in set(clustercolor):
        sp = [s for (j, s) in enumerate(space) if clustercolor[j] == c]
        sp = np.array(sp)
        plt.scatter(sp[:, 0], sp[:, 1], edgecolor='none', alpha=0.5, label=c)
    plt.legend()
    plt.savefig(file + 'cluster_space.png', bbox_inches='tight')
    plt.close()
    # DX coloring

    colors = {'AD': 'red', 'LMCI': 'yellow', 'EMCI': 'yellow', 'SMC': 'green', 'CN': 'green'}
    plt.scatter(space[:, 0], space[:, 1],
                c=dx.apply(lambda x: colors[x]), edgecolor='none', alpha=0.9,
                cmap=plt.cm.get_cmap('spectral', 10), s=10)
    plt.legend()
    plt.savefig(file + 'dx_space.png', bbox_inches='tight')
