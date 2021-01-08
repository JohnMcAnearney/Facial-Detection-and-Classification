clear all
close all

%% Pre-Processing
timing = [];
tic

trainingImagesLoc = 'face_train.cdataset';
imagesAndLabels = loadFaceImages(trainingImagesLoc);
trainingImages = imagesAndLabels.images;
trainingLabels = imagesAndLabels.labels;

%Get the assoicated hog vector for every training image
hogImages = convertToHog(trainingImages);

hogMtx.labels = trainingLabels;
hogMtx.images = hogImages;

% Apply PCA
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (hogImages);

toc
preEnd = toc;
timing = [timing, preEnd];
%% Training
tic
%Preform SVM training with data from HOG
modelSVM = SVMtraining(Xpca, hogMtx.labels, 1);
toc
trainEnd = toc
timing = [timing, trainEnd];
%% testing
tic
testStart = tic
% Loading testing labels and testing images;
testingImagesLoc = 'face_test.cdataset';
testImagesAndLabels = loadFaceImages(testingImagesLoc);
testImages = testImagesAndLabels.images;
testLabels = testImagesAndLabels.labels;

%Get the assoicated hog vector for every testing image
testHogImages = convertToHog(testImages);
testHogMtx = [];
testHogMtx.labels = trainingLabels;
testHogMtx.images = testHogImages;

%Preform SVM testing with assoictaed HOG data from testing images with PCA
for i=1:size(testHogMtx.images,1)
    image = testHogMtx.images(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult(i,1) = SVMTesting(testXpca,modelSVM);
end
toc
testEnd = toc;
timing = [timing, testEnd];

%% Evaluation
% Created function to preform calculation and generate graphs/figures
evaluation(testLabels, classificationResult, testHogImages, timing);