clear all
close all
addpath '../loadFaceImages';
addpath '../convertToEdge';
addpath '../evaluation';
%% PreProcessing
timing = [];
tic

trainingImagesLoc = 'face_train.cdataset';
imagesAndLabels = loadFaceImages(trainingImagesLoc);
trainingImages = imagesAndLabels.images;
trainingLabels = imagesAndLabels.labels;

edgeImages = convertToEdge(trainingImages, 'canny');
edgeMtx.labels = imagesAndLabels.labels;
edgeMtx.images = edgeImages;

% Apply PCA
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (edgeImages);

toc
preEnd = toc;
timing = [timing, preEnd];
%% Training
tic
%Preform KNN training with data from Edge
modelKNN.neighbours = Xpca;
modelKNN.labels = edgeMtx.labels;
toc
trainEnd = toc;
timing = [timing, trainEnd];
%% testing

tic
% Loading testing labels and testing images;
testingImagesLoc = 'face_test.cdataset';
testImagesAndLabels = loadFaceImages(testingImagesLoc);
testImages = testImagesAndLabels.images;
testLabels = testImagesAndLabels.labels;

%Get the assoicated edge vector for every testing image
testEdgeImages = convertToEdge(testImages, 'canny'); 
testEdgeMtx.images = testEdgeImages;

%Preform SVM testing with assoictaed edge data from testing images
for i=1:size(testEdgeMtx.images,1)
    image = testEdgeMtx.images(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult(i,1) = KNNTesting(testXpca,modelKNN, 1);
end

toc
testEnd = toc;
timing = [timing, testEnd];
%% Evaluation
% Created function to preform calculation and generate graphs/figures
evaluation(testLabels, classificationResult, testEdgeImages, timing);
