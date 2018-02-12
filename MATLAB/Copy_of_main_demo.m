% Load data


% true_labs = grp2idx(DX_bl);
C = 5; %%% number of clusters
rng('default'); %%% for reproducibility
[y, S, F, ydata,alpha] = SIMLR(metadatanorm1,C,10);
%%% report NMI values
% NMI_i = Cal_NMI(y,true_labs);
%fprintf(['The NMI value for dataset ' dataset{i} ' is %f\n'], NMI_i);
%%% visualization
%y_label = litekmeans(F,C,'Replicates',50);
figure;
gscatter(ydata(:,1),ydata(:,2),y);
csvwrite("labelscluster.csv", y); 