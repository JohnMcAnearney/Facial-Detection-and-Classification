clear all
close all

%% PreProcessing
% Loading all images
allImagesLoc = 'face_train_all.cdataset';
images = loadFaceImages(allImagesLoc);

% Breaking images into sets; 
[trainingG1G2, trainingG1G3, trainingG2G3, testingG1, testingG2, testingG3 ] = crossValidationSelection(images.images, images.labels);


%% ROUND 1 - trainingG1G2, testingG3;

%% Training #1 
%no training needed as KNN jsut sets labels and images
modelKNN.neighbours = trainingG1G2.images;
modelKNN.labels = trainingG1G2.labels;

%% testing #1
for i=1:size(testingG3.images,1)
    testnumber= testingG3.images(i,:);
    classificationResult1(i,1) = KNNTesting(testnumber,modelKNN, 3);
end
%% Evaluation #1
comparison = (testingG3.labels==classificationResult1);
accuracy1 = sum(comparison)/length(comparison);






%% ROUND 2  - trainingG1G3, testingG2
%no training needed as KNN jsut sets labels and images
modelKNN.neighbours = trainingG1G3.images;
modelKNN.labels = trainingG1G3.labels;

%% Testing #2
for i=1:size(testingG2.images,1)
    testnumber= testingG2.images(i,:);
    classificationResult2(i,1) = KNNTesting(testnumber,modelKNN, 3);
end

%% Evaluation #2
comparison = (testingG2.labels==classificationResult2);
accuracy2 = sum(comparison)/length(comparison);







%% ROUND 3  - trainingG2G3, testingG1
%% Training #3
%no training needed as KNN jsut sets labels and images
modelKNN.neighbours = trainingG2G3.images;
modelKNN.labels = trainingG2G3.labels;
%% Testing #3
for i=1:size(testingG1.images,1)
    testnumber= testingG1.images(i,:);
    classificationResult3(i,1) = KNNTesting(testnumber,modelKNN, 3);
end
%% Evaluation #3
comparison = (testingG1.labels==classificationResult3);
accuracy3 = sum(comparison)/length(comparison);



%% AVERAGING ROUNDS
average = (accuracy1 + accuracy2 + accuracy3) / 3;