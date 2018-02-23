# -*- coding: utf-8 -*-
"""
Main function for SIMLR_AD clustering.

This function implements the following pipeline:
* Loads the data.
* Calculates the clustering using SIMLR library in MATLAB
* Creates a classifier over the samples that belong
to those clusters.
* Visualize the selected features of said classifiers.
* Run statistical tests for the respective data and clusters.
"""

import configparser
import time
import argparse
import os
import datetime
import pandas as pd
import numpy as np
from shutil import copy2
from utils.utils import compute_simlr, feat_ranking
from sklearn.model_selection import train_test_split
from utils.stat_utils import compute_randomizedlasso, compute_univariate_test
from utils.fig_utils import draw_space, create_weight_maps
from utils.data_utils import load_data
import seaborn as sns
import matplotlib.pyplot as plt


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
    curr_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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

    print("Number of features")
    print(len(features))
    # Normalize features
    # Prepare the data, normalize
    features.iloc[:, 1:] = features.iloc[:, 1:].apply(lambda x: (x - np.mean(x)) / np.std(x), axis=0)


    # First part of the pipeline: create the mixture model
    print("Creating clusters...")
    # List of columns defnining the covariates
    model_cov = metadata.iloc[:, 5:].columns.values.tolist()
    model_cov.remove('DX_bl')
    model_cov.remove('APOE4')
    model_cov.remove('PTGENDER')

    # Sanity check of mising data
    metadata.dropna(subset=model_cov, inplace=True)

    # TEMPORAL do some computation
    metadata["participant_id"] = ['sub-ADNI' + x.replace("_", "") for x in metadata.PTID.values]

    metadata[["participant_id"]].to_csv('subjects.csv')


    metadata[model_cov].to_csv('test.csv')
    print("Number of points to cluster :" + str(len(metadata)))

    # Select only a subset of the data to cluster, and then try to add the
    # tests points into the clustering.

    # CLUSTERING
    X_train_i, X_test = train_test_split(metadata, test_size=0.25,
                                         random_state=1714)

    X_train = X_train_i.copy()
    X_train[model_cov] = X_train_i[model_cov].apply(lambda x: (x - np.mean(x)) / np.std(x), axis=0)

    f, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(X_train[model_cov], linewidths=.0, xticklabels=False, yticklabels=False, cbar=False, ax=ax)
    plt.savefig("test.png")
    plt.close()


    # Cluster space analysis
    # Use the SIMLR feature ranking
    y, S, F, ydata, alpha = compute_simlr(np.array(X_train[model_cov]), nclusters)
    aggR, pval = feat_ranking(S, np.array(X_train[model_cov]))

    sns.clustermap(S, col_cluster=True, row_cluster=True, robust=True, method='average', metric='seuclidean', figsize=(20, 20))
    plt.savefig("similarity.png")
    plt.close()

    table_featordering = pd.DataFrame({'aggR': aggR, 'pval': pval})

    table_featordering = table_featordering.sort_index(by='aggR')
    table_featordering["name"] = model_cov
    table_featordering = table_featordering.sort_index(by='pval')

    table_featordering.to_csv(out_dir + 'feat_importance.csv')

    draw_space(ydata, y, out_dir, X_train.DX_bl)
    X_train_i['clusters'] = y
    # feature space analysis
    for c in range(1, nclusters + 1):
        print('Cluster ' + str(c))
        # For each cluster, do some analysis
        X_cluster = X_train_i[X_train_i.clusters == c]
        X_cluster = X_cluster[model_cov + ['DX_bl', 'MMSE']]
        stats = pd.DataFrame(X_cluster.describe())
        stats = stats.loc[['mean','std'], :].T
        stats_DX = pd.DataFrame(X_cluster["DX_bl"].value_counts())
        stats.to_csv(out_dir + '/clusters_data/cluster_' + str(c) + '.csv')
        stats_DX.to_csv(out_dir + '/clusters_data/clusterdx_' + str(c) + '.csv')
    # We could try to do both an univariate test  And a randomized lasso test
    # To compare the selected features in each of the clusters.

    # Can we represent it over brains?
    scores, pval_univ, clusters = compute_univariate_test(features, X_train_i,
                                                     config["univariate"],
                                                     out_dir, feat_names)

    scores_lasso, clusters = compute_randomizedlasso(features, X_train_i,
                                                     config["lasso"], out_dir,
                                                     feat_names)

    # Statistical tests over the importance of each feature in the original
    # space over the metadata

    X_aux = X_train_i[["RID"]]
    X = features.merge(X_aux, 'inner', on='RID')
    X = np.array(X[feat_names])

    ## Need to select only the values in the original space for which
    # we have available features
    list_rid_og = features[["RID"]].values.tolist()
    list_rid_new = X_aux.values.tolist()
    indices = np.isin(list_rid_new, list_rid_og)
    S_new = S[np.ix_(indices.flatten(), indices.flatten())]
    aggR, pval = feat_ranking(S_new, X)
    table_featordering = pd.DataFrame({'aggR': aggR, 'pval': pval})

    table_featordering = table_featordering.sort_index(by='aggR')
    table_featordering["name"] = feat_names
    table_featordering = table_featordering.sort_index(by='pval')

    table_featordering.to_csv(out_dir + 'feat_importance_MRI.csv')

    # save results in tables
    table_scoresuniv = pd.DataFrame(scores.T, index = feat_names, columns=range(1,nclusters+1))
    table_scoreslasso = pd.DataFrame(scores_lasso.T, index = feat_names, columns=range(1,nclusters+1))
    table_pvaluniv = pd.DataFrame(np.array(pval_univ).T, index = feat_names, columns=range(1,nclusters+1))

    table_scoresuniv.to_csv(out_dir + 'results_scores_univ.csv')
    table_scoreslasso.to_csv(out_dir + 'results_scores_lasso.csv')
    table_pvaluniv.to_csv(out_dir + 'results_pval_univ.csv')

    create_weight_maps(scores, feat_names, out_dir, "univ")
    create_weight_maps(scores_lasso, feat_names, out_dir, "lasso")

    # Classify using a linear SVM.

    print('Proces finished.')
    t1 = time.time()
    print('Time to compute the script: ', t1 - t0)

    """


    # pca = PCA(n_components=2)
    # data_reduced = pca.fit_transform(metadata[model_cov])

    # ica = FastICA(n_components=2)
    # data_reduced = ica.fit_transform(metadata[model_cov])

    # import ipdb; ipdb.set_trace()
    # Asisgnation of each cluster to each sample
    metadata["cluster"] = pd.read_csv('labelscluster.csv', header=None).values
    # draw_space(data_reduced, metadata["cluster"], metadata["DX_bl"], fig_dir)
    model_cov = model_cov + ['cluster']
    metadata_og["cluster"] = pd.read_csv('labelscluster.csv', header=None).values
    metadata_og.to_csv(out_dir + "clustering_results.csv", index=False)
    # Save/interpret the clusters in some way?
    # joblib.dump(gm, out_dir + 'gmm_model.pkl')

    ###########################################

    # Second part: create a feature selection procedure for each cluster over
    # features.
    print("Computing performance of those clusters...")
    bt_weights = [[] for x in range(nclusters)]
    for c in range(1,nclusters+1):
        # Select features of those clusters
        selected = metadata[metadata.cluster == c]
        selected_rid = selected[["RID", "DX_bl"]]
        # Only AD or CN
        selected_rid = selected_rid[((selected_rid.DX_bl == 'CN') | (selected_rid.DX_bl == 'AD'))]
        selected_rid = selected_rid.merge(features, 'inner', on='RID')
        # Compute the feature selectionbootstrapping
        X = np.array(selected_rid[feat_names])
        Y = np.array(selected_rid["DX_bl"])
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
        print('Cluster ' + str(c))
        print(str(len(Y)) + ' samples.')
        values, counts = np.unique(Y, return_counts=True)
        print(list(zip(values, counts)))
        # Sanity check here
        # Save the weights, print the figures
        for j in range(nsamples):
            # sample the data
            X_sampled, Y_sampled = balanced_subsample(X, Y)
            # For each iteration, run feature_selection and
            # force the lasso feat selection to select a set of features.
            # Then, we can visualize the mean or the whole weights
            svc = LinearSVC(penalty='l1', loss='squared_hinge', dual=False)
            svc.fit(X_sampled, Y_sampled)
            bt_weights[c-1].append(svc.coef_)

        # Compute metrics in order to check sanity
        # Over the same training set, just to test it.
        svc = LinearSVC()
        svc.fit(X_train, Y_train)
        y_pred = svc.predict(X_test)

        with open(out_dir + 'c_results.txt', 'a') as f:
            f.write(str(compute_metrics_classification(Y_test, y_pred)))


    # try with single classifier
    selected_rid = metadata[["RID", "DX_bl"]]
    selected_rid = selected_rid[((selected_rid.DX_bl == 'CN') | (selected_rid.DX_bl == 'AD'))]
    selected_rid = selected_rid.merge(features, 'inner', on='RID')
    # Compute the feature selectionbootstrapping
    X = np.array(selected_rid[feat_names])
    Y = np.array(selected_rid["DX_bl"])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    svc = LinearSVC(penalty='l1', loss='squared_hinge', dual=False)
    svc.fit(X_train, Y_train)
    y_pred = svc.predict(X_test)
    print(compute_metrics_classification(Y_test, y_pred))

    # Compute figures
    create_weight_maps(bt_weights, feat_names, nclusters, fig_dir)
    np.save(out_dir + "weights_bootstrapping.npy", bt_weights)

    # Save to disk
    print('Proces finished.')
    t1 = time.time()
    print('Time to compute the script: ', t1 - t0)
    """

if __name__ == "__main__":
    main()
