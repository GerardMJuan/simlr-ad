"""
Functions to handle data.

Set of functions to handle data and metadata, such as removal of outliers,
loading of features, and so on.

Here we should have all the existing functions for data handling, from the scripts we should only need

"""

import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

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
        features_base = features_base[features_base.VISCODE == 'sc']
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


def load_data_aux(data_path):
    features_base = pd.read_csv(data_path)
    features_base = features_base[features_base.VISCODE == 'sc']
    # select only RID and features
    features = features_base.iloc[:, 10:-1]
    features = features.div(features_base.EICV, axis=0)
    feat_names = features.columns.values
    features.insert(loc=0, column='RID', value=features_base['RID'])
    
    features[feat_names] = features[feat_names].apply(
       lambda x: (x - np.min(x)) / (np.max(x)-np.min(x)), axis=0)

    return features, feat_names


def load_covariates(metadata_path):
    """
    Loads the covariate data.
    
    Loads the covariate data, and returns the dataframe and the names of the features.
    """
    # Load the metadata
    covariate_data = pd.read_csv(metadata_path)
    # Work only with baselines
    covariate_data = covariate_data[covariate_data.VISCODE == 'bl']
    # Make gender numeric
    covariate_data['PTGENDER'] = covariate_data['PTGENDER'].astype('category').cat.codes
    
    # List of columns defnining the covariates
    cov_names = covariate_data.iloc[:, 5:].columns.values.tolist()
    cov_names.remove('DX_bl')
    cov_names.remove('APOE4')
    cov_names.remove('PTGENDER')
    cov_names.remove('AGE')
    cov_names.remove('PTEDUCAT')
    # cov_names.remove('VSBPSYS')
    # cov_names.remove('VSBPDIA')
    # cov_names.remove('BMI')
    """
    variables_to_remove = ['AGE', 'APOE4', 'PTGENDER', 'DX', 'PTEDUCAT']
    
    # Encode DX
    le = LabelEncoder()
    y = le.fit_transform(covariate_data['DX_bl'])
    covariate_data['DX'] = y

    # Regress
    lr = LinearRegression()
    lr.fit(covariate_data[cov_names], covariate_data[variables_to_remove])

    # Get data with variables regressed out
    # covariate_data[cov_names] = lr.intercept_
    for c in lr.coef_:
        print(c.shape)
        covariate_data[cov_names] = covariate_data[cov_names] - covariate_data[cov_names]*c 
    """
    # Sanity check of mising data
    covariate_data.dropna(subset=cov_names, inplace=True)

    # Normalize
    # covariate_data[cov_names] = covariate_data[cov_names].apply(
    #     lambda x: (x - np.mean(x)) / np.std(x), axis=0)

    # Try different types of regularization
    # Transform to [0..1] range
    covariate_data[cov_names] = covariate_data[cov_names].apply(
       lambda x: (x - np.min(x)) / (np.max(x)-np.min(x)), axis=0)

    # Try different types of regularization
    # Transform to [0..1] range
    # metadata[model_cov] = metadata[model_cov].apply(
    #   lambda x: (x - np.min(x)) / (np.max(x)-np.min(x)), axis=0)
    
    # Apply a log transformation
    # metadata[model_cov] = metadata[model_cov].apply(
    #    lambda x: np.log10(x + 1), axis=0)

    return covariate_data, cov_names



def load_all_data(metadata_path, data_path):
    """
    Auxiliary function to call both data loading functions.
    
    This function loads both types of data and returns the already ordered, 
    intersected available data. Will be useless later but for now it is cool
    to use and all that hehe.
    """
    covariate_data, cov_names = load_covariates(metadata_path)
    feature_data, feature_names = load_data_aux(data_path)

    covariate_data.sort_index(by='RID', inplace=True)
    feature_data.sort_index(by='RID', inplace=True)

    # select only samples where both 
    selected_rid = np.intersect1d(feature_data.RID.values, covariate_data.RID.values)
    covariate_data_new = covariate_data.loc[covariate_data.RID.isin(selected_rid)] 
    feature_data_new = feature_data.loc[feature_data.RID.isin(selected_rid)]
    
    return covariate_data_new, cov_names, feature_data_new, feature_names