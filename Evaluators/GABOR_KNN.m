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
%Preform KNN training with data from Gabor 
modelKNN = KNNtraining(4, 0);
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

%Get the assoicated hog vector for every testing image
testGaborImages = convertToGabor(testImages);
testGaborMtx = [];
testGaborMtx.images = testGaborImages;

%Preform KNN testing with assoictaed Raw pixels data from testing images
for i=1:size(testGaborMtx.images,1)
    testnumber= testGaborMtx.images(i,:);
    classificationResult(i,1) = KNNTesting(testnumber,modelKNN, 1);
end

toc
testEnd = toc;
timing = [timing, testEnd];

%% Evaluation
% Created function to preform calculation and generate graphs/figures
evaluation(testLabels, classificationResult, testGaborImages, timing);