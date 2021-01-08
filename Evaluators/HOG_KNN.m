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
%Preform KNN training with data from HOG
modelKNN = KNNtraining(1, 1);
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
testImages = preprocessing(testImages);

%Get the assoicated hog vector for every testing image
testHogImages = convertToHog(testImages);
testHogMtx = [];
testHogMtx.images = testHogImages;

%Preform KNN testing with assoictaed HOG data from testing images
for i=1:size(testHogMtx.images,1)
    testnumber= testHogMtx.images(i,:);
    classificationResult(i,1) = KNNTesting(testnumber,modelKNN, 3);
end

toc
testEnd = toc;
timing = [timing, testEnd];

%% Evaluation
% Created function to preform calculation and generate graphs/figures
evaluation(testLabels, classificationResult, testHogImages, timing);