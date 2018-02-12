"""
Functions to handle data.

Set of functions to handle data and metadata, such as removal of outliers,
loading of features, and so on.
"""

import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd

def remove_outliers(metadata, config):
    """
    Remove outliers from a set of AD patients.

    Following procedure described in Iturria-Medina et al.
    (See suplementary notes) https://www.nature.com/articles/ncomms11934,
    this procedure uses the cognitive scores from each clinical group to remove
    from it the patients that present a low probability of belonging
    to that group.
    This ensures homogeneoity within the group.
    """
    # For each clinical groups
    for dx in config['data_settings']['labels'].split(','):
        # print('doing ' + dx)
        # Select only one clinical groups
        metadata_i = metadata[metadata["DX_bl"].str.contains(dx)]
        # Extract clinical data
        cog_scores = metadata_i[["CDRSB", "ADAS11", "ADAS13", "MMSE",
                                 "RAVLT_immediate"]]
        # Fill missing data
        cog_scores = cog_scores.apply(lambda x: x.fillna(x.mean()), axis=0)
        # Build model
        data = np.array(cog_scores)
        mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=0)
        # Calculate probabilities
        y = multivariate_normal.pdf(data, mean=mean, cov=cov)
        perc = np.percentile(y, 10, axis=0)
        # Add to pandas
        metadata.drop(metadata_i.index[y <= perc], inplace=True)

    return metadata


def load_data(metadata, config):
    """
    Load the data available.

    Load the features of the data described in config.
    """
    # Depending on the data type, the position of the features
    # is different
    data_type = config["data"]["data_type"]
    if data_type == 'UPENN':
        features_base = pd.read_csv(config["data"]["data_upenn"])
        features = features_base.iloc[:, 9:]
        feat_names = features.columns.values
        features.insert(loc=0, column='RID', value=features_base['RID'])
    elif data_type == "UCSD":
        features_base = pd.read_csv(config["data"]["data_ucsd"])
        # select only RID and features
        features = features_base.iloc[:, 10:-1]
        features = features.div(features_base.EICV, axis=0)
        feat_names = features.columns.values
        features.insert(loc=0, column='RID', value=features_base['RID'])
    elif data_type == "EBM":
        features_base = pd.read_csv(config["data"]["data_ebm"])
        features = features_base.iloc[:, 7:]
        feat_names = features.columns.values
        features.insert(loc=0, column='RID', value=features_base['RID'])
    elif data_type == "metadata":
        # select features from ADNIMERGE
        features_base = pd.read_csv(config["data"]["metadata"])
        # Work only with baselines
        features_base = features_base[features_base.VISCODE == 'bl']
        feat_names = ["Ventricles", "Hippocampus", "WholeBrain", "Entorhinal",
                      "Fusiform", "MidTemp"]
        # Remove missing data
        features_base = features_base.dropna(subset=feat_names)
        features = features_base.loc[:, feat_names]
        features = features.div(features_base.ICV, axis=0)
        # Remove ICV
        feat_names = features.columns.values
        features.insert(loc=0, column='RID', value=features_base['RID'])
    else:
        raise ValueError("Invalid data type!")

    return features, feat_names
