# -*- coding: utf-8 -*-
"""
Main function for SIMLR_AD clustering.

This function implements the following pipeline:
* Loads the data.
* Calculates the clustering using SIMLR library in MATLAB
* Run statistical tests for the respective data and clusters.

Using SIMLR implementation of: Wang, B., Zhu, J., Pierson, E., Ramazzotti, D., &
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
from utils.utils import compute_simlr, compute_cimlr, feat_ranking, estimate_number_clusters_cimlr
from sklearn.model_selection import train_test_split
from utils.stat_utils import compute_randomizedlasso, compute_univariate_test
from utils.fig_utils import draw_space, draw_sim_matrix, draw_twodim, create_weight_maps
from sklearn.decomposition import FastICA
from utils.data_utils import load_all_data, load_covariates

# Parser
def get_parser():
    parser = argparse.ArgumentParser(description='Clustering classification.')
    parser.add_argument("--config_file",
                        type=str, nargs=1, required=True, help='config file')
    parser.add_argument("--clusters", type=str, nargs=1,
                        required=True, help='Number of clusters')
    parser.add_argument("--output_directory_name", type=str, nargs=1,
                        required=True, help='directory where the output will be')
    parser.add_argument("--cimlr", action="store_true", help='Use cimlr')

    return parser


def main(config_file, clusters, output_directory_name, cimlr):
    """
    Compute all the tasks of the simlr-ad procedure.

    First baseline implementation of the model, with automatic
    selection of clusters.
    """
    t0 = time.time()
    # Load the configuration of a given experiment.
    config = configparser.ConfigParser()
    config.read(config_file)

    # Load number of clusters
    nclusters = int(clusters)

    # Load the output directory
    # we will also add the date to the output directory
    out_dir = (config["folders"]["EXPERIMENTS"] +
               output_directory_name + os.sep)
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
    copy2(config_file, out_dir)

    # covariate_data, cov_names, feature_data, feature_names = load_all_data(config["data"]["metadata_cl"], config["data"]["data_ucsd"])
    covariate_data, cov_names = load_covariates(config["data"]["metadata_cl"])
    covariate_data.sort_index(by='RID', inplace=True)

    # First part of the pipeline: create the mixture model
    print("Creating clusters...")

    # CLUSTERING
    t_size = float(config['data_settings']['test_size'])
    rd = int(config['general']['random_state'])

    if cimlr:
        y, S, F, ydata, alpha = compute_cimlr(
            np.array(covariate_data[cov_names]), nclusters)
    else:
        y, S, F, ydata, alpha = compute_simlr(
            np.array(covariate_data[cov_names]), nclusters)

    covariate_data['clusters'] = y

    np.save(out_dir + 'S_matrix', S)
    np.save(out_dir + 'F_matrix', F)
    np.save(out_dir + 'ydata_matrix', ydata)
    np.save(out_dir + 'alpha', alpha)

    # TODO: Need to save all other outputs

    df_cluster = pd.DataFrame(data={
        'PTID': covariate_data['PTID'],
        'DX': covariate_data['DX_bl'],
        'C': y
    })
    df_cluster.to_csv(out_dir + "cluster_data.csv")


    # Compute feature ranking using laplacian scores
    aggR, pval = feat_ranking(S, np.array(covariate_data[cov_names]))

    # Draw similarity figure
    draw_sim_matrix(S, fig_dir)

    # Save feature ordering on a table
    table_featordering = pd.DataFrame({'aggR': aggR, 'pval': pval})
    table_featordering = table_featordering.sort_index(by='aggR')
    table_featordering["name"] = cov_names
    table_featordering = table_featordering.sort_index(by='pval')
    table_featordering.to_csv(out_dir + 'feat_importance.csv')

    # Compute univariate tests and lasso randomized tests over the
    # volume features
    # Statistical tests over the importance of each feature in the original
    # space over the metadata
    """
    scores, pval_univ, clusters = compute_univariate_test(feature_data, covariate_data,
                                                          config["univariate"],
                                                          out_dir, feature_names)

    scores_lasso, clusters = compute_randomizedlasso(feature_data, covariate_data,
                                                     config["lasso"], out_dir,
                                                     feature_names)
    table_scoresuniv = pd.DataFrame(
        scores.T, index=feature_names, columns=range(1, nclusters + 1))
    table_scoreslasso = pd.DataFrame(
        scores_lasso.T, index=feature_names, columns=range(1, nclusters + 1))

    create_weight_maps(table_scoresuniv, feature_names, out_dir, "univ")
    create_weight_maps(table_scoreslasso, feature_names, out_dir, "lasso")
    """
    """
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

    # save results in tables

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
    """
    print('Procés finished.')
    t1 = time.time()
    print('Time to compute the script: ', t1 - t0)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.config_file[0], args.clusters[0], args.output_directory_name[0], args.cimlr)
