% This script performs the whole CIMLR analysis for a dataset of 282 patients 
% affected by lower-grade gliomas. 
% Original data published in Cancer Genome Atlas Research Network. "Comprehensive, 
% integrative genomic analysis of diffuse lower-grade gliomas." New England 
% Journal of Medicine 372.26 (2015): 2481-2498. 
rng(1714,'twister'); %%% for reproducibility

clear
clc
close all

addpath('data')
addpath('src')

% import and normalize the data
load('gliomas_multi_omic_data');
tumor_data = gliomas_multi_omic_data;
alldata{1} = tumor_data.point_mutations;
cna_linear = tumor_data.cna_linear;
cna_linear(cna_linear>10) = 10;
cna_linear(cna_linear<-10) = -10;
alldata{2} = cna_linear;
alldata{3} = tumor_data.methylation;
expression = tumor_data.expression;
expression(expression>10) = 10;
expression(expression<-10) = -10;
alldata{4} = expression;

for i = 1:size(alldata{1},2)
    alldata{1}(:,i) = (alldata{1}(:,i) - min(alldata{1}(:,i))) / (max(alldata{1}(:,i)) - min(alldata{1}(:,i)));
    alldata{1}(isnan(alldata{1}(:,i)),i) = 0.5;
    alldata{2}(:,i) = (alldata{2}(:,i) - min(alldata{2}(:,i))) / (max(alldata{2}(:,i)) - min(alldata{2}(:,i)));
    alldata{2}(isnan(alldata{2}(:,i)),i) = 0.5;
    alldata{3}(:,i) = (alldata{3}(:,i) - min(alldata{3}(:,i))) / (max(alldata{3}(:,i)) - min(alldata{3}(:,i)));
    alldata{3}(isnan(alldata{3}(:,i)),i) = 0.5;
    alldata{4}(:,i) = (alldata{4}(:,i) - min(alldata{4}(:,i))) / (max(alldata{4}(:,i)) - min(alldata{4}(:,i)));
    alldata{4}(isnan(alldata{4}(:,i)),i) = 0.5;
end

% estimate the best number of clusters
NUMC = 2:15;
% [K1, K2] = Estimate_Number_of_Clusters_CIMLR(alldata,NUMC);

% perform CIMLR with the estimated best number of clusters
C = 3; %%% best number of clusters
 [y, S, F, ydata] = CIMLR(alldata,C,10);
 
csvwrite('/media/gerard/ae63733d-113b-4d0e-80f1-2307f4418fd01/gerard/Documents/CODE/SIMLR-AD/true_results.csv',y)

 % visualization
figure;
gscatter(ydata(:,1),ydata(:,2),y);
 
% perform CIMLR Feature Ranking
%rng(12844,'twister'); %%% for reproducibility
%mydata = [alldata{1} alldata{2} alldata{3} alldata{4}];
%[aggR,pval] = CIMLR_Feature_Ranking(S,mydata);