clc;
close all;
clear all;

%% Training Side
% AND gate data with target matrix -- [4X2] 4 observation with 2 features
data = [0 0;
        0 1;
        1 0;
        1 1];
target = [-1;-1;-1;1]; % [0 0 0 1] is representd as [-1 -1 -1 1] 

%train an binary adaboost model
[~,admodel]=adaboost('train',data,target,500); % admodel is the trained model. save it at end for doing testing
save('admodel.mat','admodel');

% train performance
Group=adaboost('apply',data,admodel) % give the data to model for checking its training level
perf=sum(Group==target)/size(Group,1) % performance in the range of 0 to 1

%% Testing Side
% for testing load the trained model
load('admodel.mat');

testdata = [0 0]; % take 1 new unknown observation and give to trained model
Group=adaboost('apply',testdata,admodel)