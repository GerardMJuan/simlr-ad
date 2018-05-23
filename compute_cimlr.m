function [y, S, F, ydata, alphaK] = compute_cimlr(X, nclusters)
%% Function that, given data and the number of clusters, returns the clustering
%% done by CIMLR
addpath('MATLAB/src');
addpath('MATLAB');
addpath('MATLAB/kernels2/');
X = double(X);
for i= 1:size(X,2)
    fulldata{i} = X(:,i);
end
rng('default'); %%% for reproducibility
nclusters = double(nclusters);
[y, S, F, ydata, alphaK, timeOurs, converge, LF] = CIMLR(fulldata,nclusters,10);
end
