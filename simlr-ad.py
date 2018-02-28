# -*- coding: utf-8 -*-
"""
Main function for SIMLR_AD clustering.

This function implements the following pipeline:
* Loads the data.
* Calculates the clustering using SIMLR library in MATLAB
* Run statistical tests for the respective data and clusters.

SIMLR method based on: Wang, B., Zhu, J., Pierson, E., Ramazzotti, D., &
Batzoglou, S. (2017). Visualization and analysis of single-cell rna-seq data by
kernel-based similarity learning. Nature Methods, 14(4), 414–416.
http://doi.org/10.1038/nMeth.4207
"""

import configparser
import time
import argparse
import os
import pandas as pd
import numpy as np
from shutil import copy2
from utils.utils import compute_simlr, feat_ranking
from sklearn.model_selection import train_test_split
from utils.stat_utils import compute_randomizedlasso, compute_univariate_test
from utils.fig_utils import draw_space, draw_sim_matrix
from utils.data_utils import load_data

# Adding input parameters.
parser = argparse.ArgumentParser(description='Clustering classification.')
parser.add_argument("--config_file",
                    type=str, nargs=1, required=True, help='config file')
parser.add_argument("--clusters", type=str, nargs=1,
                    required=True, help='Number of clusters')
parser.add_argument("--output_directory_name", type=str, nargs=1,
                    required=True, help='directory where the output will be')


def main():
    """
    Compute all the tasks of the simlr-ad procedure.

    First baseline implementation of the model, with automatic
    selection of clusters.
    """
    t0 = time.time()
    args = parser.parse_args()
    # Load the configuration of a given experiment.
    config = configparser.ConfigParser()
    config.read(args.config_file[0])

    # Load number of clusters
    nclusters = int(args.clusters[0])

    # Load the output directory
    # we will also add the date to the output directory
    out_dir = (config["folders"]["EXPERIMENTS"] +
               args.output_directory_name[0] + os.sep)
    # Create out directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create figures directory
    if not os.path.exists(out_dir + os.path.join('figures', '')):
        os.makedirs(out_dir + os.path.join('figures', ''))
    fig_dir = out_dir + os.path.join('figures', '')

    # Create clusters directory
    if not os.path.exists(out_dir + os.path.join('clusters_data', '')):
        os.makedirs(out_dir + os.path.join('clusters_data', ''))

    # Save a copy of the configuration file into the experiments directory
    copy2(args.config_file[0], out_dir)

    # Load the metadata
    metadata = pd.read_csv(config["data"]["metadata_cl"])
    # Work only with baselines
    metadata = metadata[metadata.VISCODE == 'bl']
    # Make gender numeric
    metadata['PTGENDER'] = metadata['PTGENDER'].astype('category').cat.codes

    # Load the feaures
    features, feat_names = load_data(metadata, config)

    # Prepare the data, normalize
    features.iloc[:, 1:] = features.iloc[:, 1:].apply(
        lambda x: (x - np.mean(x)) / np.std(x), axis=0)

    # First part of the pipeline: create the mixture model
    print("Creating clusters...")

    # List of columns defnining the covariates
    model_cov = metadata.iloc[:, 5:].columns.values.tolist()

    # Remove not used covariates
    model_cov.remove('DX_bl')
    model_cov.remove('APOE4')
    model_cov.remove('PTGENDER')

    # Sanity check of mising data
    metadata.dropna(subset=model_cov, inplace=True)

    # Select only a subset of the data to cluster, and then try to add the
    # tests points into the clustering.

    # CLUSTERING
    t_size = config['data_settings']['test_size']
    rd = config['general']['random_state']
    X_train_i, X_test = train_test_split(metadata, test_size=t_size,
                                         random_state=rd)
    # Normalize
    X_train = X_train_i.copy()
    X_train[model_cov] = X_train_i[model_cov].apply(
        lambda x: (x - np.mean(x)) / np.std(x), axis=0)

    # Cluster space analysis
    # Use the SIMLR feature ranking
    y, S, F, ydata, alpha = compute_simlr(
        np.array(X_train[model_cov]), nclusters)

    # Assign the clustering
    X_train_i['clusters'] = y

    # Compute feature ranking using laplacian scores
    aggR, pval = feat_ranking(S, np.array(X_train[model_cov]))

    # Draw similarity figure
    draw_sim_matrix(S, fig_dir)

    # Save feature ordering on a table
    table_featordering = pd.DataFrame({'aggR': aggR, 'pval': pval})
    table_featordering = table_featordering.sort_index(by='aggR')
    table_featordering["name"] = model_cov
    table_featordering = table_featordering.sort_index(by='pval')
    table_featordering.to_csv(out_dir + 'feat_importance.csv')

    # Draw cluster space
    draw_space(ydata, y, fig_dir, X_train.DX_bl)

    # Feature space analysis
    for c in range(1, nclusters + 1):
        print('Cluster ' + str(c))
        # For each cluster, do some analysis
        X_cluster = X_train_i[X_train_i.clusters == c]
        X_cluster = X_cluster[model_cov + ['DX_bl', 'MMSE']]
        stats = pd.DataFrame(X_cluster.describe())
        stats = stats.loc[['mean', 'std'], :].T
        stats_DX = pd.DataFrame(X_cluster["DX_bl"].value_counts())
        stats.to_csv(out_dir + '/clusters_data/cluster_' + str(c) + '.csv')
        stats_DX.to_csv(
            out_dir + '/clusters_data/clusterdx_' + str(c) + '.csv')

    # Compute univariate tests and lasso randomized tests over the
    # volume features
    # Statistical tests over the importance of each feature in the original
    # space over the metadata
    scores, pval_univ, clusters = compute_univariate_test(features, X_train_i,
                                                          config["univariate"],
                                                          out_dir, feat_names)

    scores_lasso, clusters = compute_randomizedlasso(features, X_train_i,
                                                     config["lasso"], out_dir,
                                                     feat_names)

    # save results in tables
    table_scoresuniv = pd.DataFrame(
        scores.T, index=feat_names, columns=range(1, nclusters + 1))
    table_scoreslasso = pd.DataFrame(
        scores_lasso.T, index=feat_names, columns=range(1, nclusters + 1))
    table_pvaluniv = pd.DataFrame(
        np.array(pval_univ).T, index=feat_names, columns=range(1, nclusters + 1))

    table_scoresuniv.to_csv(out_dir + 'results_scores_univ.csv')
    table_scoreslasso.to_csv(out_dir + 'results_scores_lasso.csv')
    table_pvaluniv.to_csv(out_dir + 'results_pval_univ.csv')

    # Use laplacian scores to see statistically significant differences
    # between subjects
    X_aux = X_train_i[["RID"]]
    X = features.merge(X_aux, 'inner', on='RID')
    X = np.array(X[feat_names])

    # Need to select only the values in the original space for which
    # we have available features
    list_rid_og = features[["RID"]].values.tolist()
    list_rid_new = X_aux.values.tolist()
    indices = np.isin(list_rid_new, list_rid_og)
    S_new = S[np.ix_(indices.flatten(), indices.flatten())]
    aggR, pval = feat_ranking(S_new, X)

    # Order it into a table and save to disk
    table_featordering = pd.DataFrame({'aggR': aggR, 'pval': pval})
    table_featordering = table_featordering.sort_index(by='aggR')
    table_featordering["name"] = feat_names
    table_featordering = table_featordering.sort_index(by='pval')
    table_featordering.to_csv(out_dir + 'feat_importance_MRI.csv')

    print('Procés finished.')
    t1 = time.time()
    print('Time to compute the script: ', t1 - t0)


if __name__ == "__main__":
    main()
