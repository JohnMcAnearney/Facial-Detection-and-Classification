clear all
close all

%% training
tic
allImagesLoc = 'face_train_all.cdataset';
imagesAndLabels = loadFaceImages(allImagesLoc);
allImages = imagesAndLabels.images;
allLabels = imagesAndLabels.labels;

%Divide the image set into three groups
%Two will be used for training and the other testing
%G3 is the testing set
cvIndices  = crossvalind('kFold',imagesAndLabels.labels,3);

ImagesG1 = [];
ImagesG2  = [];
ImagesG3 = [];
LabelsG1 = [];
LabelsG2  = [];
LabelsG3 = [];

for i =1:size(cvIndices,1)
   if(cvIndices(i,:) == 1)
       ImagesG1 = [ImagesG1; allImages(i,:)];
       LabelsG1 = [LabelsG1; allLabels(i,:)];
   end
    if(cvIndices(i,:) == 2)
       ImagesG2 = [ImagesG2; allImages(i,:)];
       LabelsG2 = [LabelsG2; allLabels(i,:)];
   end
    if(cvIndices(i,:) == 3)
       ImagesG3 = [ImagesG3; allImages(i,:)];
       LabelsG3 = [LabelsG3; allLabels(i,:)];
   end    
end

%Initilise combination of training groups

% trainingMatxG1G2 = [];
trainingMatxG2G3 = [];
% trainingMatxG1G3 = [];

% trainingMatxG1G2Labels =[];
trainingMatxG2G3Labels = [];
% trainingMatxG1G3Labels = [];

% trainingMatxG1G2 = [trainingMatxG1G2;ImagesG1];
% trainingMatxG1G2 = [trainingMatxG1G2;ImagesG2];
% trainingMatxG1G2Labels = [trainingMatxG1G2Labels;LabelsG1];
% trainingMatxG1G2Labels = [trainingMatxG1G2Labels;LabelsG2];

trainingMatxG2G3 = [trainingMatxG2G3;ImagesG2];
trainingMatxG2G3 = [trainingMatxG2G3;ImagesG3];
trainingMatxG2G3Labels = [trainingMatxG2G3Labels;LabelsG2];
trainingMatxG2G3Labels = [trainingMatxG2G3Labels;LabelsG3];

% trainingMatxG1G3 = [trainingMatxG1G3;ImagesG1];
% trainingMatxG1G3 = [trainingMatxG1G3;ImagesG3];
% trainingMatxG1G3Labels = [trainingMatxG1G3Labels;LabelsG1];
% trainingMatxG1G3Labels = [trainingMatxG1G3Labels;LabelsG3];


% Preform training
net = patternnet(10);

%Select training group to train
%[net, tr] = train(net, trainingMatxG1G2', trainingMatxG1G2Labels');
[net, tr] = train(net, trainingMatxG2G3', trainingMatxG2G3Labels'); 
% [net, tr] = train(net, trainingMatxG1G3', trainingMatxG1G3Labels'); 
nntraintool;

%% testing

% Uncomment group which is not used in training for testing

% testImages = ImagesG1;
testImages = ImagesG2;
%testImages = ImagesG3;

% testLabels = LabelsG1;
testLabels = LabelsG2;
% testLabels = LabelsG3;

testResult = net(testImages');
testResultIndexes = vec2ind(testResult);
[correct, incorrect] = confusion(testResult, testLabels');
plotconfusion;
plotroc(testResult, testLabels');


%% Evaluation

% Created function to preform calculation and generate graphs/figures
evaluation(testLabels, testResult, testEdgeImages);
