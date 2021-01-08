clc;
close all;
clear all;

%% Training Side
% random three class data with target matrix -- [9X3] 9 observation with 3 features
data = [10 0 0;
        10 0 1;
        10 1 0;
        2 10 0;
        2 10 1;
        2 11 0;
        5 0 10;
        5 0 11;
        5 1 10];
target = [1;1;1;2;2;2;3;3;3];

admodel = cell(3,1);
classes = unique(target);
%This example has 3 classes but adaboost is an binary classifier so train an binary adaboost model for each class
for j = 1:numel(classes)
    indx(target~=j)=-1;indx(target==j)=1; % Create binary class target for each classifier
    [~,admodel{j}]=adaboost('train',data,indx,500); % admodel is the trained model. save it at end for doing testing
end
save('admodel.mat','admodel');

% train performance
Scores = zeros(size(data,1),numel(classes));
for j = 1:numel(classes)
    label=adaboost('apply',data,admodel{j});
    Scores(:,j) = label; % Second column contains positive-class scores
end
Scores
[~,maxScore] = max(Scores,[],2)

perf=sum(maxScore==target)/size(maxScore,1) % performance in the range of 0 to 1

%% Testing Side
% for testing load the trained model
load('admodel.mat');
numclasses = 3;
testdata = [10 0 0]; % take 1 new unknown observation and give to trained model
for j = 1:numclasses
    label=adaboost('apply',testdata,admodel{j});
    Group(:,j) = label;
end
Group
[~,maxGroup] = max(Group,[],2)