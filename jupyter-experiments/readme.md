## Experiment files
The Jupyter notebook contained here can be used to both reproduce the results in the paper and generate the figures. The files are:
* **bootstrap.ipynb** : Contains the stability tests for the clusters, using the jaccard distance.
* **clustering_visualization.ipynb** : Contains scripts to generate figures of the distance matrices and the embeddings.
* **data_statistics.ipynb** : Computes the statistics of different subpopulations.
* **permutation_test.ipynb** : Calculates the permutation tests of the different values.
* **test_roi.ipynb** : Generates an overlay figure of the brain with the statistical tests (unused).

Those files need either to have already generated a clustering using the standalone simlr-ad.py script, or generating one on the go. In this case,
an appropiate configuration file and parameters needs to be set.