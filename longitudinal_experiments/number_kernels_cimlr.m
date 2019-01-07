function [K1, K2] = number_kernels_cimlr(X, list)
%% Function that, given data and the number of clusters, returns the clustering
%% done by CIMLR
addpath('/home/gerard/Documents/CODE/SIMLR-AD/MATLAB/src')
addpath('/home/gerard/Documents/CODE/SIMLR-AD/MATLAB')
X = double(X);
for i= 1:size(X,2)
    fulldata{i} = X(:,i);
end
rng('default'); %%% for reproducibility
[K1, K2] = Estimate_Number_of_Clusters_CIMLR(fulldata, list);
end
