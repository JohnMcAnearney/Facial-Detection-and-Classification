clear all
close all

%% Pre-Processing
timing = [];
tic
trainStart = tic;

trainingImagesLoc = 'face_train.cdataset';
imagesAndLabels = loadFaceImages(trainingImagesLoc);
trainingImages = imagesAndLabels.images;
trainingLabels = imagesAndLabels.labels;

%Preprocess Images
trainingImages = preprocessing(trainingImages);
% Apply PCA
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (trainingImages);

toc
preEnd = toc;
timing = [timing, preEnd];
%% Training
%due to time constraints, just hard coding training. But, for KNN, no
%training is necessary really as all it does is set neighbours and labels
tic
modelKNN.neighbours = Xpca;
modelKNN.labels = trainingLabels;
toc
trainEnd = toc
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
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult(i,1) = KNNTesting(testXpca, modelKNN, 1);
end

toc
testEnd = toc;
timing = [timing, testEnd];
%% Evaluation
% Created function to preform calculation and generate graphs/figures
evaluation(testLabels, classificationResult, testImages, timing);
