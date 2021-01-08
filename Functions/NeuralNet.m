clear all
close all

%% training
timing = [];
tic
trainStart = tic;

allImagesLoc = 'face_train_all.cdataset';
imagesAndLabels = loadFaceImages(allImagesLoc);
allImages = imagesAndLabels.images;
allLabels = imagesAndLabels.labels;

% setting layer to 15;
net = patternnet(15);

labels = zeros(910, 2);
for i=1:size(allLabels)
   if(allLabels(i,1) == 1)
        labels(i,1) = 1; 
   else
        labels(i,2) = 1; 
   end
end
size(allImages)
size(labels)
[net, tr] = train(net, allImages', labels'); 
nntraintool;

toc
trainEnd = toc;
timing = [timing, trainEnd];
%% testing
tic
testStart = tic;
testImagesFlipped = allImages';
testLabelsFlipped = labels';

testImages = testImagesFlipped(:, tr.testInd);
testLabels = testLabelsFlipped(:, tr.testInd);

testResults = net(testImages);
testResult = vec2ind(testResults);
testResultFlipped = testResult';
results = [];
for i=1:size(testResult')
   if(testResultFlipped(i,1) == 2)
        results(i,1) = 1; 
   else
        results(i,1) = 0; 
   end
end

figure, plotroc(testLabels, testResults);
confusionchart(testLabels', results');
testEnd = toc;
timing = [timing, testEnd];
%% Evaluation

% Created function to preform calculation and generate graphs/figures
evaluation(testLabels, testResults, testImages, timing);
