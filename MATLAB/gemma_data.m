% This script performs the whole CIMLR analysis

clear
clc
close all

addpath('data')
addpath('src')

% import and normalize the data
load('mydatabase');

data = Tpatients;

% normalize min-max
for i = 1:size(data, 2)
    data(:,i) = (data(:,i) - min(data(:,i))) / (max(data(:,i)) - min(data(:,i)));
    data(isnan(data(:,i)),i) = 0.5;
end

rng(32655,'twister'); %%% for reproducibility
k = randperm(17344);
small_data = data(k(1:1734),:);

% Try without first columns
for i = 1:size(data, 2)
    alldata{i} = data(:,i)
end

alldata = alldata(3:end)

% estimate the best number of clusters
NUMC = 2:15;
[K1, K2] = Estimate_Number_of_Clusters_CIMLR(alldata,NUMC);

subplot(1,2,1)
plot(NUMC,K1,'b-s','LineWidth',4);
title('Relative Quality')
subplot(1,2,2)
plot(NUMC,K2,'r-o','LineWidth',4);
title('Adjusted Quality')


% perform CIMLR with the estimated best number of clusters
C = 3; %%% best number of clusters
rng(43556,'twister'); %%% for reproducibility
[y, S, F, ydata] = CIMLR(alldata,C,10);
 
csvwrite('true_results.csv',y)

 % visualization
figure;
gscatter(ydata(:,1),ydata(:,2),y);
 
% perform CIMLR Feature Ranking
%rng(12844,'twister'); %%% for reproducibility
%mydata = [alldata{1} alldata{2} alldata{3} alldata{4}];
%[aggR,pval] = CIMLR_Feature_Ranking(S,mydata);
