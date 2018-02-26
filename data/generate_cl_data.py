"""
Script to combine existing ADNI data
for creating a dataset with clustering
data.
"""
import pandas as pd
import numpy as np

# Load data path
data_path = 'ADNIMERGE.csv'
vitals_path = 'VITALS.csv'
CSF_path = 'UPENNBIOMK_MASTER.csv'
PLASMA = "UPENNPLASMA.csv"
final_path = "CLUSTER_DATA.csv"
CSF_full_path = "adni_csf.csv"
PLASMA_full_path = "adni_plasma.csv"
# Load data
df_data = pd.read_csv(data_path, low_memory=False)
df_vitals = pd.read_csv(vitals_path)
df_csf = pd.read_csv(CSF_path)
df_plasma = pd.read_csv(PLASMA)
df_csf_full = pd.read_csv(CSF_full_path)
df_plasma_full = pd.read_csv(PLASMA_full_path)

# Select columns
info = ["RID", "PTID", "MMSE", "VISCODE", "EXAMDATE", "AGE", "PTGENDER", "APOE4", "DX_bl", "PTEDUCAT"]

df_data = df_data[info].copy()

# Select only baseline data
df_data = df_data[(df_data.VISCODE == "bl")]

# Add corresponding VITALS information to columns
df_vitals = df_vitals.drop_duplicates(subset="RID", keep="first")
df_data = pd.merge(df_data, df_vitals[["RID", "VSWEIGHT", "VSHEIGHT", "VSBPSYS", "VSBPDIA", "VSWTUNIT", "VSHTUNIT"]], how='inner', on="RID")

# Need to convert inches to centimeters
df_data["VSHEIGHT"] = df_data.apply(lambda x: x.VSHEIGHT*2.54 if x.VSHTUNIT == 1 else x.VSHEIGHT, axis=1)

# Need to convert pounds to kilograms
df_data["VSWEIGHT"] = df_data.apply(lambda x: x.VSWEIGHT*0.453 if x.VSWTUNIT == 1 else x.VSWEIGHT, axis=1)

#Drop type of data columns
df_data.drop('VSWTUNIT', axis=1, inplace=True)
df_data.drop('VSHTUNIT', axis=1, inplace=True)

# Calculate BMI
df_data["BMI"] = df_data.apply(lambda x: x.VSWEIGHT/(x.VSHEIGHT * x.VSHEIGHT) * 10000, axis=1)

# Drop -1 and -4
df_data = df_data.replace(to_replace=[-1, -4], value=[np.nan, np.nan]).dropna()

# Delete height and weight, they are already in BMI
del df_data["VSHEIGHT"]
del df_data["VSWEIGHT"]

print(df_data["DX_bl"].describe())

df_data.to_csv('cluster_NOCSF_NOPLASMA.csv', index=False)
# Print some of the statistics
print("Number of subjects: " + str(len(df_data)))
print("Statistics by diagnosis")
print(df_data["DX_bl"].value_counts())
'''
## ADD CSF Biomarkers
# Drop empty columns
df_csf_full = df_csf_full.dropna(axis=1, how='any')

# Add all colums from col 3 to col end-2
df_data = pd.merge(df_data, df_csf_full.iloc[:, :-2], how='inner', on="RID")

del df_data["sampid"]
# Drop na
df_data.dropna()
print("Number of subjects: " + str(len(df_data)))
print("Statistics by diagnosis")
print(df_data["DX_bl"].value_counts())
df_data.to_csv('cluster_NOPLASMA.csv')
'''
## ADD Plasma biomarkers
# Drop empty columns
df_plasma_full = df_plasma_full.replace(to_replace=['.'], value=[np.nan])
df_plasma_full = df_plasma_full.dropna(axis=1, how='any')

# select only baselines
df_plasma_full = df_plasma_full[df_plasma_full.Visit_Code == 'bl']

# Add all colums
df_data = pd.merge(df_data, df_plasma_full.iloc[:, 1:], how='inner', on="RID")
del df_data['Visit_Code']
del df_data['RBM Sample ID']
del df_data['Sample_Received_Date']

# Drop na
df_data.dropna()
print("Number of subjects: " + str(len(df_data)))
print("Statistics by diagnosis")
print(df_data["DX_bl"].value_counts())
df_data.to_csv('cluster_FULL.csv', index=False)

'''
# CSF DATA
# Drop duplicates
df_csf = df_csf[df_csf.VISCODE == "bl"].drop_duplicates(subset="RID", keep="last")

# ADD to data
df_data = pd.merge(df_data, df_csf[["RID", "ABETA"]], how='inner', on="RID")
df_data = pd.merge(df_data, df_csf[["RID", "TAU"]], how='inner', on="RID")
df_data = pd.merge(df_data, df_csf[["RID", "PTAU"]], how='inner', on="RID")

df_data.to_csv('cluster_NOPLASMA.csv')
# Print some of the statistics
print("Number of subjects: " + str(len(df_data)))
print("Statistics by diagnosis")
print(df_data["DX_bl"].value_counts())

# PLASMA data
df_plasma = df_plasma[df_plasma.VISCODE == "bl"].drop_duplicates(subset="RID", keep="last")
#ADD to data
df_data = pd.merge(df_data, df_plasma[["RID", "AB40", "AB42"]], how='inner', on="RID")

# Remove missing data
df_data = df_data.dropna()
df_data.to_csv('cluster_FULL.csv')

# TODO: NORMALIZE, BUT IT IS ALREADY DONE IN THE OTHER FILES.
# Normalize STUDIES LEVEL WELL

# Print some of the statistics
print("Number of subjects: " + str(len(df_data)))
print("Statistics by diagnosis")
print(df_data["DX_bl"].value_counts())
'''
