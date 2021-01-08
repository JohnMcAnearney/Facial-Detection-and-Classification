clear all
close all
%% Pre-processing
timing = [];
tic

% Loading testing labels and testing images;
trainingImagesLoc = 'face_train.cdataset';
imagesAndLabels = loadFaceImages(trainingImagesLoc);
trainingImages = imagesAndLabels.images;
trainingLabels = imagesAndLabels.labels;
trainingImages = preprocessing(trainingImages);
toc
preEnd = toc;
timing = [timing, preEnd];

trainingImages = preprocessing(trainingImages);

%%Training
tic
%Preform SVM training
modelSVM = SVMtraining(trainingImages, imagesAndLabels.labels, 3);

toc
trainEnd = toc;
timing = [timing, trainEnd];

%% testing
tic
testStart = tic;

% Loading testing labels and testing images;
testingImagesLoc = 'face_test.cdataset';
testImagesAndLabels = loadFaceImages(testingImagesLoc);
testImages = testImagesAndLabels.images;
testLabels = testImagesAndLabels.labels;

% Preform testing 
for i=1:size(testImages,1)
    testnumber= testImages(i,:);
    classificationResult(i,1) = SVMTesting(testnumber,modelSVM);
end

toc
testEnd = toc;
timing = [timing, testEnd];

%% Evaluation
% Created function to preform calculation and generate graphs/figures
evaluation(testLabels, classificationResult, testImages, timing);
