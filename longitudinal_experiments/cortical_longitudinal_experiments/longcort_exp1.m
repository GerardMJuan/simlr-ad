%%%% First test script of cortical analysis %%%%%
%% Using mixed-effects models
                     
fsdir = '/home/gerard/Documents/LIB/freesurfer/';
qdec = 'qdec_test.txt';

%% Load data
[Y,mri] = fs_read_Y('lh.thickness_sm10.mgh');
lhsphere = fs_read_surf([fsdir 'subjects/fsaverage/surf/lh.sphere']);
lhcortex = fs_read_label([fsdir 'subjects/fsaverage/label/lh.cortex.label']);

% Set important paths
fslmehome = sprintf('%s/matlab/','/home/gerard/Documents/LIB/freesurfer');
addpath(genpath(fslmehome)); 

%% Read qdec
Qdec = fReadQdec(qdec);

% Data sorting (needed)
Qdec = rmQdecCol(Qdec,1);  % (removes first column)
sID = Qdec(2:end,1);   % (grabs the subjects' IDs)
Qdec = rmQdecCol(Qdec,1);  % (removes the subjects'ID column)
M = Qdec2num(Qdec);  % (converts to a numeric matrix)
[M,Y,ni] = sortData(M,1,Y,sID);  % (sorts the data)

clusters = M(:,2);
dx = M(:,3);

c1CN = double(clusters==1.0 & dx==0);
c1MCI = double(clusters==1.0 & dx==1);
c1AD = double(clusters==1.0 & dx==2);

c2CN = double(clusters==2.0 & dx==0);
c2MCI = double(clusters==2.0 & dx==1);
c2AD = double(clusters==2.0 & dx==2);

c3CN = double(clusters==3.0 & dx==0);
c3MCI = double(clusters==3.0 & dx==1);
c3AD = double(clusters==3.0 & dx==2);

c4CN = double(clusters==4.0 & dx==0);
c4MCI = double(clusters==4.0 & dx==1);
c4AD = double(clusters==4.0 & dx==2);

% Create model
% simple linear model containing all the info

% need to include time in all the columns

t = M(:,1);

X = [ones(length(M),1) c1CN c1CN.*t c1MCI c1MCI.*t c1AD c1AD.*t...
    c2CN c2CN.*t c2MCI c2MCI.*t c2AD c2AD.*t...
    c3CN c3CN.*t c3MCI c3MCI.*t c3AD c3AD.*t...
    c4CN c4CN.*t c4MCI c4MCI.*t c4AD c4AD.*t M(:, 4:end)];

% Compute likelihood
[lhTh0,lhRe] = lme_mass_fit_EMinit(X,[1 2],Y,ni,lhcortex,3);
[lhRgs,lhRgMeans] = lme_mass_RgGrow(lhsphere,lhRe,lhTh0,lhcortex,1.8,95);

% visually compare
surf.faces =  lhsphere.tri;
surf.vertices =  lhsphere.coord';

figure; p1 = patch(surf);
set(p1,'facecolor','interp','edgecolor','none','facevertexcdata',lhTh0(1,:)');

figure; p2 = patch(surf); set(p2,'facecolor','interp','edgecolor','none','facevertexcdata',lhRgMeans(1,:)');

% Fit spatiotemporal LME model,  for one or two random effects, and check if its correct
% left
lhstats = lme_mass_fit_Rgw(X,[1 2],Y,ni,lhTh0,lhRgs,lhsphere);

lhTh0_1RF = lme_mass_fit_EMinit(X,[1],Y,ni,lhcortex,3);
lhstats_1RF = lme_mass_fit_Rgw(X,[1],Y,ni,lhTh0_1RF,lhRgs,lhsphere);

LR_pval = lme_mass_LR(lh_stats,lhstats_1RF,1);

dvtx = lme_mass_FDR2(LR_pval,ones(1,length(LR_pval)),lhcortex,0.05,0);

% right

% Lists of contrasts
each row should have 15 positions. Testing, for each 
CM.C = [zeros(3,3)  [1 0 0 0 0; -1 0 1 0 0; 0 0 -1 0 1] zeros(3,5)];

F_lhstats = lme_mass_F(lhstats,CM);

% Run contrast following test

contrast_list= {
[zeros(2) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 zeros(4)],
[zeros(2) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 zeros(4)],
[zeros(2) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 zeros(4)],
[zeros(2) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 zeros(4)],
[zeros(2) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 zeros(4)],
[zeros(2) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 zeros(4)],
[zeros(2) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 zeros(4)],
[zeros(2) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 zeros(4)],
[zeros(2) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 zeros(4)],
[zeros(2) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 zeros(4)],
[zeros(2) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 zeros(4)],
[zeros(2) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 zeros(4)],
}

names_list = {
    

}


for i = 1:length(contrast_list)
    
    

