"""
Functions for the ordering and model optimization.

Set of functions that are useful to order following some rules.
"""
import subprocess
from functools import cmp_to_key
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Set margin value (global)
margin = 10


def eval_baseline(full_data, test_data, config, out_dir):
    """
    Baseline evaluation function.

    This function creates a classification model to compare performance by a
    single score, using a PCA of 1 component.
    """
    if config['train']['train'] != 'False':
        # If we want to train the baselin pipeline
        # Define PCA
        pca = PCA(n_components=1)
        # Get features
        feat = full_data.values.tolist()
        feat_n = np.array([d[3:] for d in feat[:]])
        pca.fit(feat_n)
        # Create return file
        df = pd.DataFrame(test_data, columns=['RID', 'DX', 'AGE'])
        # Get features for test
        feat_t = test_data.values.tolist()
        feat_n_t = np.array([d[3:] for d in feat_t[:]])
        df['scores'] = pca.transform(feat_n_t)
    else:
        # We already have the features
        df = pd.DataFrame(test_data, columns=['RID', 'DX', 'AGE'])
        df['scores'] = test_data.SPARE_AD.values
    return df


def ordering_rule(a, b):
    """
    Ordering function.

    Support function that decides on the order between two samples. Those two
    samples have a field called DX_bl, with value AD
    or CN, and a numeric field called AGE.
    Column DX_bl es 1, column 2 es AGE
    """
    # IMPERATIVE that we use always the second element of a and b
    # The first element will be the features
    a = a[1]
    b = b[1]
    #  That function should take two arguments to be compared and then
    # return a negative value for less-than, return zero if they are equal,
    # or return a positive value for greater-than.
    # If they have the same diagnostic, ordering by age
    if a[1] == b[1]:
        return a[2] - b[2]
    else:
        # If they have different diagnostic
        if a[1] == 'AD':
            return a[2] - b[2] + margin
        elif a[1] == 'CN':
            return a[2] - b[2] - margin
        else:
            return 0


def create_dat_file(data, out_file, ord=True):
    """
    Create dat file for SVMRank.

    Saves to disk a file with the correct format for a SVMRank execution.
    """
    # data = data.values.tolist()
    # shuffle(data)
    # Sort the data
    # Write all the data to file
    print(out_file)
    with open(out_file, 'w') as f:
        # For every subject
        # kf = KFold(n_splits=0)
        qid = 1
        od = 1
        # for _, data_ind in kf.split(data):
        # data_qid = [data[i] for i in data]
        ord_data = data.values.tolist()
        if ord:
            ord_data = sorted(ord_data, key=cmp_to_key(ordering_rule))
        for d in ord_data[:]:
            l = "{0} qid:{1} ".format(od, qid)
            # Now we add the features to the line
            nfeat = 1
            for feat in d[3:]:
                l += '{0}:{1} '.format(nfeat, feat)
                nfeat = nfeat + 1
            # Write file
            f.write(l + '\n')
            od = od + 1
        # qid = qid + 1

    # The file is written
    print('Training file written')
    return [d[:3] for d in ord_data]


def calculate_svmrank(train_data, config, out_dir, m):
    """
    Train svmrank with train data.

    train_data is a pandas DataFrame containing
    """
    # Assign global margin
    global margin
    margin = m
    # Train file
    train_file = out_dir + config['train']['train_out']
    # model file
    model_file = out_dir + config['train']['model_path']

    # Create the training file
    create_dat_file(train_data, train_file)

    # Create empty model out_file
    open(model_file, 'a').close()

    # Create the execution order of svmrank_training
    cmdline = [config['svmrank']['learn']]
    cmdline += ['-c ' + config['train']['C']]
    # Config
    # cmdline += ['-l 2 -p 2 -o 1 -y 3 -v 3']
    # cmdline += ['-t 2 -g ' + str(gamma)]
    cmdline += [train_file]
    cmdline += [model_file]

    # Execute
    print('Run ' + " ".join(cmdline))
    returncode = subprocess.call(" ".join(cmdline), shell=True)
    if returncode != 0:
        raise ValueError("Error training!", returncode)
    # Return model_file
    return model_file


def eval_svmrank(final_data, model_file, config, out_dir, m, name):
    """
    Evaluate data with trained svmrank.

    Train data with model_path and return scores assigned.
    """
    # Assign global margin
    global margin
    margin = m
    # Test file
    test_file = out_dir + config['train']['test_out']
    # Prediction file
    pred_file = out_dir + config['train']['pred_file']

    # Create testing file
    ordered_final_data = create_dat_file(final_data, test_file, ord=False)

    # Create empty pred file
    open(pred_file, 'a').close()

    # Create the execution order of svmrank_training
    cmdline = [config['svmrank']['eval']]
    cmdline += [test_file]
    cmdline += [model_file]
    cmdline += [pred_file]
    # Execute
    proc = subprocess.Popen(" ".join(cmdline), stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    if err:
        raise ValueError("Error testing!", err)

    # Save output of evaluation to file
    # Commented for now, as it is not needed
    # output_file = out_dir + name + '.txt'
    # with open(output_file, "w") as text_file:
    #     print(out, file=text_file)

    # Recover scores
    scores = np.loadtxt(pred_file)

    # Create return file
    df = pd.DataFrame(ordered_final_data, columns=['RID', 'DX', 'AGE'])
    df['scores'] = scores
    return df
