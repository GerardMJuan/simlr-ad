## Experiments for cortical structures
This folder contains all necessary files to reproduce the cortical structures experiments. Each of the Jupyter files
contain a corresponding experiment. Each files can generate the results for both cortical thickness and cortical volume:

 - cortical_exp_1.ipynb: contains the experiment on cortical structure for each cluster group with the rest.
 - cortical_exp_2.ipynb: contains the experiment on cortical structure betweeen diagnostic groups at each presentation.
 - cortical_exp_3.ipynb: contains the experiment on cortical structure between different diagnostic groups on the whole population.
 - cortical_exp_4.ipynb: contains the experiment on cortical structure between different diagnosis stages across presentations.

Those files are actually really similar and share a bit of code, but for the sake of reproducibility they are in completely separate
files. This way, each of the experiments can be replicated separately.

# Requeriments
Freesurfer 6.0 and fsPALM are required to perform the experiments, with the subjects directory of freesurfer pointing to a folder containing the processed files of the subjects.
