clear all
close all

%% training
timing = [];
tic
trainStart = tic;

allImagesLoc = 'face_train_all.cdataset';
testImagesLoc = 'face_test.cdataset';
imagesAndLabels = loadFaceImages(allImagesLoc);
testimagesAndLabels = loadFaceImages(testImagesLoc);
allImages = imagesAndLabels.images;
allLabels = imagesAndLabels.labels;
allTI = testimagesAndLabels.images;

net = patternnet(10);

%Select training group to train
reformatedAllImages = allImages';
reformatedAllLabels = allLabels';
reformatedAllT = allTI;
[net, tr] = train(net, allImages, allTI); 
nntraintool;

toc
trainEnd = toc
timing = [timing, trainEnd];
%% testing
tic
testStart = tic;

testImages = reformatedAllImages(:, tr.testInd);
testLabels = allTI(:, tr.testInd);

testResults = net(testImages);
testIndices = vec2ind(testResults);
[correct, incorrect] = confusion(testIndices, testLabels);
plotconfusion;
plotroc(testResults, testLabels);

toc
testEnd = toc;
timing = [timing, testEnd];
%% Evaluation

% Created function to preform calculation and generate graphs/figures
evaluation(testLabels, testResults, testImages, timing);
