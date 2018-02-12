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
    Compute all the tasks of the LADCE procedure.

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
    model_cov = config["data"]["columns_cl"].split(',')
    metadata.dropna(subset=model_cov, inplace=True)

    metadata[model_cov] = metadata[model_cov].apply(lambda x: (x - np.mean(x)) / np.std(x), axis=0)

    print("Number of points to cluster :" + str(len(metadata)))

    # Select only a subset of the data to cluster, and then try to add the
    # tests points into the clustering.

    # CLUSTERING
    X_train, X_test = train_test_split(metadata, test_size=0.25,
                                       random_state=1714)

    # Cluster space analysis
    # Use the SIMLR feature ranking
    y, S, F, ydata, alpha = compute_simlr(np.array(X_train[model_cov]), nclusters)
    aggR, pval = feat_ranking(S, np.array(X_train[model_cov]))

    draw_space(ydata, y, out_dir, X_train.DX_bl)
    X_train['clusters'] = y
    # feature space analysis

    # We could try to do both an univariate test  And a randomized lasso test
    # To compare the selected features in each of the clusters.

    # Statistical tests over the importance of each feature in the original
    # space over the metadata

    # Can we represent it over brains?
    scores, pval, clusters = compute_univariate_test(features, X_train,
                                                     config["univariate"],
                                                     out_dir, feat_names)

    scores_lasso, clusters = compute_randomizedlasso(features, X_train,
                                                     config["lasso"], out_dir,
                                                     feat_names)
    # save results in tables
    table_scoresuniv = pd.DataFrame(scores.T, index = feat_names, columns=['1', '2', '3', '4' ,'5'])
    table_scoreslasso = pd.DataFrame(scores_lasso.T, index = feat_names, columns=['1', '2', '3', '4' ,'5'])
    table_pvaluniv = pd.DataFrame(np.array(pval).T, index = feat_names, columns=['1', '2', '3', '4' ,'5'])

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
