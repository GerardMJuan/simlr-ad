%%%% First test script of cortical analysis %%%%%

% Set important paths
freesurfer_base = strcat('export FREESURFER_HOME=/home/gerard/Documents/LIB/freesurfer && ',...
                         'source $FREESURFER_HOME/SetUpFreeSurfer.sh && export SUBJECTS_DIR=/home/gerard/Documents/DATA/Data/CIMLR_LONG_CARS_TEST');

fslmehome = sprintf('%s/matlab/','/home/gerard/Documents/LIB/freesurfer');
addpath(genpath(fslmehome)); 
                     
fsdir = '/home/gerard/Documents/LIB/freesurfer/';
qdec = 'qdec_test.txt';

% Create preprocessing steps
preproc_l = ['mris_preproc' ' --qdec-long ' qdec ' --target fsaverage --hemi lh --meas thickness --out lh.thickness.mgh'];
preproc_r = ['mris_preproc' ' --qdec-long ' qdec ' --target fsaverage --hemi rh --meas thickness --out rh.thickness.mgh'];

surf2_l = 'mri_surf2surf --hemi lh --s fsaverage --sval lh.thickness.mgh --tval lh.thickness_sm10.mgh --fwhm-trg 10 --cortex  --noreshape';
surf2_r = 'mri_surf2surf --hemi rh --s fsaverage --sval rh.thickness.mgh --tval rh.thickness_sm10.mgh --fwhm-trg 10 --cortex  --noreshape';

% mri_preproc
system([freesurfer_base ' && ' preproc_l], '-echo');
system([freesurfer_base ' && ' preproc_r], '-echo');

% surf2surf
system([freesurfer_base ' && ' surf2_l], '-echo');
system([freesurfer_base ' && ' surf2_r], '-echo');

asegstats2table --qdec-long qdec_test.dat --stats aseg.stats --tablefile aseg.table.txt
aparcstats2table --qdec-long qdec_test.dat --stats aparc.stats --tablefile aparc.table.txt

mris_preproc --qdec-long qdec_test.dat --target fsaverage --hemi lh --meas thickness --out lh.thickness.stack.mgh
mri_surf2surf --hemi lh --s fsaverage --sval lh.thickness.stack.mgh --tval lh.thickness.stack.fwhm10.mgh --fwhm-trg 10 --cortex --noreshape

mris_preproc --qdec-long qdec_test.dat --target fsaverage --hemi rh --meas thickness --out rh.thickness.stack.mgh
mri_surf2surf --hemi rh --s fsaverage --sval rh.thickness.stack.mgh --tval rh.thickness.stack.fwhm10.mgh --fwhm-trg 10 --cortex --noreshape


