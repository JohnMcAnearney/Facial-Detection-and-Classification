clear all
close all

%% Trying to implement LDA, was unsuccessful. 

%% Pre-Processing
timing = [];
tic
trainStart = tic;

trainingImagesLoc = 'face_train.cdataset';
imagesAndLabels = loadFaceImages(trainingImagesLoc);
trainingImages = imagesAndLabels.images;
trainingLabels = imagesAndLabels.labels;
reshapedImages=[];

% Apply LDA
unit8Im = uint8(trainingImages);
[eigenVectors, eigenValues, meanX, Xlda] = LDA(trainingLabels, [], trainingImages);
toc
preEnd = toc;
timing = [timing, preEnd];
%% Training
tic
unit8Im = uint8(Xlda);
modelSVM = SVMtraining(trainingImages,unit8Im);
toc
trainEnd = toc;
timing = [timing, trainEnd];
%% testing
tic
testStart = tic;

testingImagesLoc = 'face_test.cdataset';
testImagesAndLabels = loadFaceImages(testingImagesLoc);
testImages = testImagesAndLabels.images;
testLabels = testImagesAndLabels.labels;

for i=1:size(testImages,1)
    image = testImages(i,:);
    A = uint8(image) - meanX;
    testXlda = A'*eigenVectors';
    classificationResult(i,1) = SVMTesting(testXlda, modelSVM);
end

toc
testEnd = toc;
timing = [timing, testEnd];
%% Evaluation
% Created function to preform calculation and generate graphs/figures
evaluation(testLabels, classificationResult, testImages, timing);
