
clear all
close all

%% Pre-processing
timing = [];
tic;

trainingImagesLoc = 'face_train.cdataset';
imagesAndLabels = loadFaceImages(trainingImagesLoc);
trainingImages = imagesAndLabels.images;
trainingLabels = imagesAndLabels.labels;

%Get the assoicated hog vector for every training image
gaborImages = convertToGabor(trainingImages);

garMtx.labels = trainingLabels;
garMtx.images = gaborImages;

% Apply PCA
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (gaborImages);

toc
preEnd = toc;
timing = [timing, preEnd];

%% Training
tic
%Preform SVM training with data from HOG
modelKNN.neighbours = garMtx.images;
modelKNN.labels = garMtx.labels;

toc
trainEnd = toc;
timing = [timing, trainEnd];

%% testing
tic;
testStart = tic;

% Loading testing labels and testing images;
testingImagesLoc = 'face_test.cdataset';
testImagesAndLabels = loadFaceImages(testingImagesLoc);
testImages = testImagesAndLabels.images;
testLabels = testImagesAndLabels.labels;

%Get the assoicated hog vector for every testing image
testGarImages = convertToGabor(testImages);
testGarMtx.images = testGarImages;

%Preform SVM testing with assoictaed HOG data from testing images
for i=1:size(testGarMtx.images,1)
    image = testGarMtx.images(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult(i,1) = KNNTesting(testXpca,modelKNN, 1);
end

toc;
testEnd = toc;
timing = [timing, testEnd];
%% Evaluation;
evaluation(testLabels, classificationResult, testImages, timing);


