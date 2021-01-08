clear all
close all
addpath '../loadFaceImages';
addpath '../convertToEdge';
addpath '../evaluation';
%% PreProcessing
% Loading all images
allImagesLoc = 'face_train_all.cdataset';
images = loadFaceImages(allImagesLoc);

% Breaking images into sets; 
[trainingG1G2, trainingG1G3, trainingG2G3, testingG1, testingG2, testingG3 ] = crossValidationSelection(images.images, images.labels);




%% ROUND 1 - trainingG1G2, testingG3;
edgeTrainingImages = convertToEdge(trainingG1G2.images, 'prewitt');
edgeTrainingLabels = trainingG1G2.labels;

edgeTestingImages = convertToEdge(testingG3.images, 'prewitt');
edgeTestingLabels = testingG3.labels;

[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (edgeTrainingImages);

%% Training #1 
%Preform KNN training with data from Edge extractor
modelKNN.neighbours = Xpca;
modelKNN.labels = edgeTrainingLabels;

%% testing #1
for i=1:size(edgeTestingImages,1)
    image = edgeTestingImages(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult1(i,1) = KNNTesting(testXpca,modelKNN, 1);
end
%% Evaluation #1
comparison = (edgeTestingLabels==classificationResult1);
accuracy1 = sum(comparison)/length(comparison);






%% ROUND 2  - trainingG1G3, testingG2
edgeTrainingImages = convertToEdge(trainingG1G3.images, 'prewitt');
edgeTrainingLabels = trainingG1G3.labels;

edgeTestingImages = convertToEdge(testingG2.images, 'prewitt');
edgeTestingLabels = testingG2.labels;

[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (edgeTrainingImages);

%% Training #2
modelKNN.neighbours = Xpca;
modelKNN.labels = edgeTrainingLabels;

%% Testing #2
for i=1:size(edgeTestingImages,1)
    image = edgeTestingImages(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult2(i,1) = KNNTesting(testXpca,modelKNN, 3);
end

%% Evaluation #2
comparison = (edgeTestingLabels==classificationResult2);
accuracy2 = sum(comparison)/length(comparison);







%% ROUND 3  - trainingG2G3, testingG1
edgeTrainingImages = convertToEdge(trainingG2G3.images, 'prewitt');
edgeTrainingLabels = trainingG2G3.labels;

edgeTestingImages = convertToEdge(testingG1.images, 'prewitt');
edgeTestingLabels = testingG1.labels;

[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (edgeTrainingImages);

%% Training #3
modelKNN.neighbours = Xpca;
modelKNN.labels = edgeTrainingLabels;

%% Testing #3
for i=1:size(edgeTestingImages,1)
    image = edgeTestingImages(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult3(i,1) = KNNTesting(testXpca,modelKNN, 3);
end
%% Evaluation #3
comparison = (edgeTestingLabels==classificationResult3);
accuracy3 = sum(comparison)/length(comparison);





%% AVERAGING ROUNDS
average = (accuracy1 + accuracy2 + accuracy3) / 3;