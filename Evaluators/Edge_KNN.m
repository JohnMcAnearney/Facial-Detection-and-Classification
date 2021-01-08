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
%Preform KNN training with data from EDGE
modelKNN = KNNtraining(2, 1);
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

%Get the assoicated edge vector for every testing image
testEdgeImages = convertToEdge(testImages, 'canny'); 

testEdgeMtx.images = testEdgeImages;

%Preform KNN testing with assoictaed edge data from testing images
for i=1:size(testEdgeMtx.images,1)
    %edIm = reshape(testEdgeMtx.images(i,:), 27, []);
    testnumber= testEdgeMtx.images(i,:);
    classificationResult(i,1) = KNNTesting(testnumber,modelKNN, 1);
end

toc
testEnd = toc;
timing = [timing, testEnd];

%% Evaluation
% Created function to preform calculation and generate graphs/figures
evaluation(testLabels, classificationResult, testEdgeImages, timing);


