"""
Create new archive of subject metadata including only:
1. actual patients used in the paper,
2. patients still not processed

The script traverse over the po

"""

import pandas as pd
import os
import shutil

freesurfer_dir = '/homedtic/gmarti/DATA/Data/SIMLR-AD-FS_Full/'
freesurfer_dir_new = '/homedtic/gmarti/DATA/Data/SIMLR-AD-FS/'
bids_dir = '/homedtic/gmarti/DATA/Data/ADNI_BIDS/'
list_of_patients = "subjects_experiment.csv"

# Read the list of patients
df_patients = pd.read_csv(list_of_patients, header=None)
df_patients.columns = ["ID", "PTID"]

fs_status = []
bids_status = []
i = 0

# For each patient
for ptid in df_patients.PTID.values:
    done = False
    # Check in which preprocessing step they are:
    try:
        # Open log file, compare last step
        fs_dir = freesurfer_dir + 'ADNI' + ptid[0:3] + 'S' + ptid[6:] + '/'
        log_file = 'scripts/recon-all-status.log'
        with open(fs_dir + log_file) as f:
            content = f.readlines()
            content = ' '.join(content)
            if "BA_exvivo" in content:
                # All done
                print('Moving ' + fs_dir)
                fs_status.append('Done')
                # shutil.move(fs_dir, freesurfer_dir_new)
                # Move whole directory into the new folder
            elif "Curvature Stats" in content:
                # Needs reconall 3
                fs_status.append('3')
            elif "SubCort Seg" in content:
                # Needs reconall 2, 3
                fs_status.append('2,3')
            else:
                # Else is new case, check manually
                # import shutil
                # shutil.rmtree(fs_dir, ignore_errors=True)
                fs_status.append('Manual check')
    except:
        # If the file is not available, say it
        print('Freesurfer directory not created for ' + ptid)
        fs_status.append('NA')
    # Also check if the patient has an available scan or not
    patient_dir = 'sub-ADNI' + ptid[0:3] + 'S' + ptid[6:]
    bids_ptid = bids_dir + patient_dir + '/ses-M00/anat/' + patient_dir + '_ses-M00_T1w.nii.gz'
    bids_status.append(os.path.exists(bids_ptid))

# Put all the information into a dataframe and print it
df_patients['fs_status'] = fs_status
df_patients['bids_status'] = bids_status
df_patients.to_csv('freesurfer_information.csv')

"""
# Read the processed patients
proc_p = []
with open(processed_patients) as f:
    content = f.readlines()
    for line in content:
        index = line.find(" ADNI")
        proc_p.append(line[index+1:index+13])

with open(processed_patients_2) as f:
    content = f.readlines()
    for line in content:
        index = line.find(" ADNI")
        proc_p.append(line[index+1:index+13])
# Convert them to the same format as PTID
proc_p_f = [x[4:7] + '_S_' + x[8:] for x in proc_p]
print(proc_p_f)

# Select subset of list of patients that are not processed
df_final_list = df_patients[~df_patients.PTID.isin(proc_p_f)]
print(len(df_patients))
print(len(proc_p_f))
print(len(df_final_list))

df_final_list.to_csv('list_patients_subset_missing.csv')
"""
