%%%% First test script of cortical analysis %%%%%
%% Using mass univariate, depending on the result we can use 
fsdir = '/home/gerard/Documents/LIB/freesurfer/';
qdec = 'qdec_test.txt';

% Set important paths
fslmehome = sprintf('%s/matlab/','/home/gerard/Documents/LIB/freesurfer');
addpath(genpath(fslmehome)); 


%% Read the table

[aseg,asegrows,asegcols] =  fast_ldtable('aseg.long.table');
asegcols=cellstr(asegcols); % convert column names into cell string

% select the interested structures
% Podem fer left i right hippocampus, left i right ventricles?

what = 'Left-Hippocampus';
id=find(strcmp(what,asegcols)==1);
Y = aseg(:,id);

id=find(strcmp('EstimatedTotalIntraCranialVol',asegcols)==1);
etiv = aseg(:,id);

%% Read qdec
% Get them together and sort the data

Qdec = fReadQdec('qdec_full.dat');
Qdec = rmQdecCol(Qdec,1);
sID = Qdec(2:end,1);
Qdec = rmQdecCol(Qdec,1);
M = Qdec2num(Qdec);
M = M(:,1:7);

[M,Y,ni] = sortData(M,1,Y,sID);

%% Do tests
% Equal to test 1


clusters = M(:,2);
dx = M(:,3);

c1 = double(clusters==1.0);
c2 = double(clusters==2.0);
c3 = double(clusters==3.0);
c4 = double(clusters==4.0);

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

%% Test
i = 2.0;
c = double(clusters==i);
rest = double(clusters~=i);

% Create model
% simple linear model containing all the info

% need to include time in all the columns

t = M(:,1);

X = [ones(length(M),1) t c c.*t rest rest.*t M(:, 3:end) etiv];

% lme_lowessPlot(M(:,1),Y(:,1),0.70,M(:,2));

%& Do the analysis

% one effect
total_hipp_vol_stats = lme_fit_FS(X,[1 2],Y(:,1),ni);

% two effects
total_hipp_vol_stats_1RF = lme_fit_FS(X,[1],Y(:,1),ni);

% Comparison between effects
lr = lme_LR(total_hipp_vol_stats.lreml,total_hipp_vol_stats_1RF.lreml,1);


%% Fer els tests

% Test 1
% Each cluster group with respect to the rest.
% Not that informative

% Test 2
% Betweeen diagnostic groups for each presentation (cluster).

% Test 3
% Between diagnostic groups on the whole population (normal)

% Test 4
% Between jo que se ja mateume


% Contrast matrix
C = [zeros(3,3) [1 0 0 0 0; -1 0 1 0 0; 0 0 -1 0 1] zeros(3,6)];

% Final test
F_C = lme_F(total_hipp_vol_stats,C);