## SIMLR_AD
Repository of the code implementing non-supervised clustering procedure [1]
for Alzheimer's disease patients subtyping.


## Requeriments
Python 2.7+ is required. Matlab engine is also required.
Packages:

## Data
Data used is from ADNI database:

patients-dtic2018.csv is the list of patients identificators used in the results presented
at ETIC PhD Workshop 2018. Data is available at ( ), authorization needed. The files used are:


## Instructions:
1. Place the corresponding data in the data/ directory.

2. Define a config file with the experiment parameters. An already existing file, named config_dtic2018.ini, was used
to generate the experiments presented at ETIC PhD Workshop 2018.

3. Execute simlr-ad.py. Example execution:

python simlr-ad.py --config_file configs/baseline_linux.ini --clusters 3 --output_directory_name test_poster

4. folder figures_generator/ and poster.ppptx are the files necessary to
reproduce the poster at ETIC PhD Workshop 2018.

## References
[1]
[2]
