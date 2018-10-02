"""
Prepare data to run freesurfer group tests.

This script will also be extended to do permutation tests.
"""
import pandas as pd
import io
import os
import shutil
import itertools

freesurfer_dir_new = '/homedtic/gmarti/DATA/Data/SIMLR-AD-FS_Full/'
freesurfer_files = 'freefiles/'

cluster_data = 'cluster_data.csv'
df_cluster = pd.read_csv(cluster_data)

DX = ['CN', 'MCI', 'AD']
C = ["C1", "C2", "C3", "C4"]

fsid_names = [x for x in map(lambda s: 'ADNI' + s[0:3] + 'S' + s[6:], df_cluster.PTID.values)]

df_qdec = pd.DataFrame(data={'fsid': fsid_names,
                            'cluster': df_cluster.C.values,
                            'diagnosis': df_cluster.DX.values},
                       columns = ['fsid', 'cluster', 'diagnosis'])

# Create FSGD file
fsgd_file = 'freefiles/adni_files.fsgd'
with open(fsgd_file, 'w') as f:
    f.write('GroupDescriptorFile 1\n')  # First line, mandatory
    f.write('Title CIMLR-AD 1\n')   # Title, optional
    for x in itertools.product(C, DX):
        f.write('Class ' + x[0] + '-' + x[1] + '\n')   # Title, optional
    for subj in df_qdec.itertuples():
        subj_ptid = subj.fsid
        cluster = 'C' + str(int(subj.cluster))
        dx = subj.diagnosis
        if subj.diagnosis == 'LMCI':
            dx = 'MCI'
        f.write('Input ' + subj_ptid + ' ' + cluster + '-' + dx + '\n')


def write_contrast(name, matrix):
    with open(name, 'w') as f:
        f.write(matrix)  # First line, mandatory

# Create mtx files for different comparisons
# Each file has as positive a concrete group of diagnosis in a cluster,
# And as negative the other gorups of same diagnostic in the rest of clusters
# Columns are C1CN C1MCI C1AD C2CN C2MCI C2AD C3CN C3MCI C3AD C4CN C4MCI C4AD

## Cluster 1
mtx_C1CN = 'freefiles/C1CN.mtx'
contrast = "+1 0 0 -0.33 0 0 -0.33 0 0 -0.33 0 0\n"
write_contrast(mtx_C1CN, contrast)

mtx_C1MCI = 'freefiles/C1MCI.mtx'
contrast = "0 +1 0 0 -0.33 0 0 -0.33 0 0 -0.33 0\n"
write_contrast(mtx_C1MCI, contrast)

mtx_C1AD = 'freefiles/C1AD.mtx'
contrast = "0 0 +1 0 0 -0.33 0 0 -0.33 0 0 -0.33\n"
write_contrast(mtx_C1AD, contrast)

## Cluster 2
mtx_C2CN = 'freefiles/C2CN.mtx'
contrast = "-0.33 0 0 +1 0 0 -0.33 0 0 -0.33 0 0\n"
write_contrast(mtx_C2CN, contrast)

mtx_C2MCI = 'freefiles/C2MCI.mtx'
contrast = "0 -0.33 0 0 +1 0 0 -0.33 0 0 -0.33 0\n"
write_contrast(mtx_C2MCI, contrast)

mtx_C2AD = 'freefiles/C2AD.mtx'
contrast = "0 0 -0.33 0 0 +1 0 0 -0.33 0 0 -0.33\n"
write_contrast(mtx_C2AD, contrast)

## Cluster 3
mtx_C3CN = 'freefiles/C3CN.mtx'
contrast = "-0.33 0 0 -0.33 0 0 +1 0 0 -0.33 0 0\n"
write_contrast(mtx_C3CN, contrast)

mtx_C3MCI = 'freefiles/C3MCI.mtx'
contrast = "0 -0.33 0 0 -0.33 0 0 +1 0 0 -0.33 0\n"
write_contrast(mtx_C3MCI, contrast)

mtx_C3AD = 'freefiles/C3AD.mtx'
contrast = "0 0 -0.33 0 0 -0.33 0 0 +1 0 0 -0.33\n"
write_contrast(mtx_C3AD, contrast)

## Cluster 4
mtx_C4CN = 'freefiles/C4CN.mtx'
contrast = "-0.33 0 0 -0.33 0 0 -0.33 0 0 +1 0 0\n"
write_contrast(mtx_C4CN, contrast)

mtx_C4MCI = 'freefiles/C4MCI.mtx'
contrast = "0 -0.33 0 0 -0.33 0 0 -0.33 0 0 +1 0\n"
write_contrast(mtx_C4MCI, contrast)

mtx_C4AD = 'freefiles/C4AD.mtx'
contrast = "0 0 -0.33 0 0 -0.33 0 0 -0.33 0 0 +1\n"
write_contrast(mtx_C4AD, contrast)
