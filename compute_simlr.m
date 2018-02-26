function [y, S, F, ydata, alpha] = compute_simlr(X, nclusters)
%% Function that, given data and the number of clusters, returns the clustering
%% done by SIMLR
addpath('MATLAB/src')
X = double(X);
nclusters = double(nclusters);
rng('default'); %%% for reproducibility
[y, S, F, ydata,alpha] = SIMLR(X,nclusters,10);
end
