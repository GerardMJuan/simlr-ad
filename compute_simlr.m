function [y, S, F, ydata, alpha] = compute_simlr(X, nclusters)
%% Function that, given data and the number of clusters, returns the clustering
%% done by SIMLR
addpath('MATLAB/src')
X = double(X);
rng('default'); %%% for reproducibility
nclusters = double(nclusters);
[y, S, F, ydata,alpha] = SIMLR(X,nclusters,10);
end
