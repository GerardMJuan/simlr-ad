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

def main():
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
        l = (l - np.min(l, axis=0)) / (np.max(l, axis=0) - np.min(l, axis=0))
        l[np.isnan(l)] = 0.5
        alldata[i] = [l]
        i += 1

    # RUN CIMLR
    model = CIMLR(C=3, k=10)
    model.fit(alldata)

    # Test the results


if __name__ == "__main__":
    # parser = get_parser()
    # args = parser.parse_args()
    # main(args.config_file[0], args.clusters[0], args.output_directory_name[0], args.cimlr)
    main()
