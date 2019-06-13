"""
Quick script to create long qdec.
"""

import pandas as pd
import glob
import os

# image_dir
img_dir = "/home/gerard/Documents/DATA/Data/CIMLR_LONG_CARS_TEST/"
# Cluster path
cluster_path = "/home/gerard/Documents/EXPERIMENTS/SIMLR-AD/cimlr4long/"
# adnimerge_path
adnimerge_path = "/home/gerard/Documents/DATA/metadata/ADNI/ADNIMERGE.csv"

df_adnimerge = pd.read_csv(adnimerge_path)

# Create lists to save later in the dataframe
fsid_names = []
fsid_bases = []
Cs = []
Dxs = []
years = []
ages = []
educations = []
apoes = []
gender = []

df_cluster = pd.read_csv(cluster_path + 'cluster_data.csv')
df_cluster.reset_index(inplace=True)
df_cluster.head()


# Traverse the directory with the info
files = glob.glob(img_dir + '*.long.*')
for file in files:
    bname = os.path.basename(file)
    i = bname.find(".long.")
    fsid = bname[0:i]
    fsid_base = bname[i+6:]
    ptid = fsid_base[4:7] + '_S_' + fsid_base[8:]
    fsid_names.append(fsid)
    fsid_bases.append(fsid_base)
    C = df_cluster.loc[df_cluster["PTID"] == ptid, 'C'].values[0]
    Dx = df_cluster.loc[df_cluster["PTID"] == ptid, 'DX'].values[0]
    Cs.append(C)
    Dxs.append(Dx)

    # compute the years, from the months on the name
    m = fsid[-2:]
    y = float(m)/12.0

    Years_bl = df_adnimerge.loc[(df_adnimerge["PTID"] == ptid) & (df_adnimerge["Month"] == int(m)), 'Years_bl'].values[0]
    years.append(Years_bl)
    # transform the m to the appropiate thing for adnimerge

    # Add other covariates
    age = df_adnimerge.loc[(df_adnimerge["PTID"] == ptid) & (df_adnimerge["Month"] == int(m)), 'AGE'].values[0]
    age = age + Years_bl
    apoe = df_adnimerge.loc[(df_adnimerge["PTID"] == ptid) & (df_adnimerge["Month"] == int(m)), 'APOE4'].values[0]
    edu = df_adnimerge.loc[(df_adnimerge["PTID"] == ptid) & (df_adnimerge["Month"] == int(m)), 'PTEDUCAT'].values[0]
    g = df_adnimerge.loc[(df_adnimerge["PTID"] == ptid) & (df_adnimerge["Month"] == int(m)), 'PTGENDER'].values[0]

    ages.append(age)
    educations.append(edu)
    apoes.append(apoe)
    gender.append(g)

df_qdec = pd.DataFrame(data={'fsid': fsid_names,
                             'fsid_base': fsid_bases,
                             'years': years,
                             'cluster': Cs,
                             'diagnosis': Dxs,
                             'age': ages,
                             'apoe': apoes,
                             'education': educations,
                             'gender': gender},
                       columns=['fsid', 'fsid_base',
                                'years', 'cluster', 'diagnosis',
                                'age', 'apoe', 'education', 'gender'])

# Convert to numeric
to_numeric = {"diagnosis":     {"CN": 0, "LMCI": 1, "AD": 2},
              "gender": {"Male": 0, "Female": 1}}
df_qdec.replace(to_numeric, inplace=True)
df_qdec.to_csv("qdec_test.txt", sep=' ', index=False)
