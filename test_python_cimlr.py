"""
Test script for CIMLR.

This CIMLR loads some basic test data and runs the algorithm
Ideally, I should use a known dataset so that I can reproduce it
in  MATLAB and see that it works exactly the same
"""

from cimlr_python.cimlr import CIMLR
import scipy.io
from sklearn.preprocessing import minmax_scale
import numpy as np
import pandas as pd

def main():

    np.random.seed(1714)
    """Load the data nd run the algorithm

    Run with the same data as the matlab example
    """
    # Load the data
    data = scipy.io.loadmat("MATLAB/data/gliomas_multi_omic_data.mat")
    data = data['gliomas_multi_omic_data']
    
    # Put the data in the section
    # Just repeat the preprocessing on the TEST MATLAB file
    alldata = []
    # Point mutations
    alldata.append([data[0, 0]["point_mutations"]])
    # cna linear
    cna_linear = data[0, 0]["cna_linear"]
    cna_linear[cna_linear > 10] = 10
    cna_linear[cna_linear < -10] = -10
    alldata.append([cna_linear])
    # Methylation
    alldata.append([data[0, 0]["methylation"]])
    # Expression
    expression = data[0, 0]["expression"]
    expression[expression > 10] = 10
    expression[expression < -10] = -10
    alldata.append([expression]) 
    # Now, normalize
    i = 0
    for [l] in alldata:
        # l = minmax_scale(l)
        l = (l - np.amin(l, axis=0)) / (np.amax(l, axis=0) - np.amin(l, axis=0))
        l[np.isnan(l)] = 0.5
        alldata[i] = [l]
        i += 1
    
    # RUN CIMLR
    model = CIMLR(C=3, k=10)
    model.fit(alldata)

    S = model.S
    c_data = model.y

    y_true = pd.read_csv("true_results.csv", header=None)
    y_true = y_true.values.squeeze()
    
    # Ha de ser el mateix grup, així que es pot comparar a qualsevol de los combinacions
    # de labels
    # l que farem serà, agafar els indexs de cada cluster i comparar les ocurrencies,
    # dins de cada cluster, de les labels de sortida
    idx_a, = np.where(y_true==1)
    idx_b, = np.where(y_true==2)
    idx_c, = np.where(y_true==3)
    
    from collections import Counter
    label_a, count_a = Counter(c_data[idx_a]).most_common(1)[0]
    label_b, count_b = Counter(c_data[idx_b]).most_common(1)[0]
    label_c, count_c = Counter(c_data[idx_c]).most_common(1)[0]

    per_a = count_a / len(idx_a)
    per_b = count_b / len(idx_b)
    per_c = count_c / len(idx_c)

    # Fem la comparacio entre output de CIMLR i output de pyCIMLR
    per = np.mean([per_a, per_b, per_c])
    print('Percentage of correct ones: ' + str(per*100))


if __name__ == "__main__":
    # parser = get_parser()
    # args = parser.parse_args()
    # main(args.config_file[0], args.clusters[0], args.output_directory_name[0], args.cimlr)
    main()
