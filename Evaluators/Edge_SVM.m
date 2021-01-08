clear all
close all
%% PreProcessing
timing = [];
tic

trainingImagesLoc = 'face_train.cdataset';
imagesAndLabels = loadFaceImages(trainingImagesLoc);
trainingImages = imagesAndLabels.images;
trainingLabels = imagesAndLabels.labels;

%trainingImages = preprocessing(trainingImages);

edgeImages = convertToEdge(trainingImages, 'canny');
edgeMtx.labels = imagesAndLabels.labels;
edgeMtx.images = edgeImages;

toc
preEnd = toc;
timing = [timing, preEnd];
%% Training
tic
%Preform SVM training with data from HOG
modelSVM = SVMtraining(edgeMtx.images, edgeMtx.labels, 2);
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


%testImages = preprocessing(testImages);

%Get the assoicated edge vector for every testing image with Canny feature
%extrator
testEdgeImages = convertToEdge(testImages, 'canny'); 

testEdgeMtx.images = testEdgeImages;

%Preform SVM testing with assoictaed edge data from testing images
for i=1:size(testEdgeMtx.images,1)
    testnumber= testEdgeMtx.images(i,:);
    classificationResult(i,1) = SVMTesting(testnumber,modelSVM);
end

toc
testEnd = toc;
timing = [timing, testEnd];
%% Evaluation
% Created function to preform calculation and generate graphs/figures
evaluation(testLabels, classificationResult, testEdgeImages, timing);
