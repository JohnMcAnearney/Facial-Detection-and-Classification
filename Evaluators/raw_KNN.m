clear all
close all
%% Training
%These timings are here just to keep the evaluation function at the bottom
%happy.
timing = [];
tic
toc
preEnd = toc;
timing = [timing, preEnd];

tic
%Preform KNN training with data from Raw pixels
modelKNN = KNNtraining(3, 0);
toc
trainEnd = toc;
timing = [timing, trainEnd];

%% Testing

tic
% Loading testing labels and testing images;
testingImagesLoc = 'face_test.cdataset';
testImagesAndLabels = loadFaceImages(testingImagesLoc);
testImages = testImagesAndLabels.images;
testLabels = testImagesAndLabels.labels;

%Preprocess images
%testImages = preprocessing(testImages);

%Preform KNN testing with assoictaed Raw pixels data from testing images
for i=1:size(testImages,1)
    testnumber= testImages(i,:);
    classificationResult(i,1) = KNNTesting(testnumber,modelKNN, 5);
end

toc
testEnd = toc;
timing = [timing, testEnd];

%% Evaluation
% Created function to preform calculation and generate graphs/figures
evaluation(testLabels, classificationResult, testImages, timing);