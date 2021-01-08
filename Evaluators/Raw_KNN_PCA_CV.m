clear all
close all

%% PreProcessing
% Loading all images
allImagesLoc = 'face_train_all.cdataset';
images = loadFaceImages(allImagesLoc);

% Breaking images into sets; 
[trainingG1G2, trainingG1G3, trainingG2G3, testingG1, testingG2, testingG3 ] = crossValidationSelection(images.images, images.labels);



%% ROUND 1 - trainingG1G2, testingG3;
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (trainingG1G2.images);
trainingLabels = trainingG1G2.labels;
testingImages = testingG3.images;
testingLabels = testingG3.labels;

%% Training #1 
modelKNN.neighbours = Xpca;
modelKNN.labels = trainingLabels;

%% Testing #1
for i=1:size(testingImages,1)
    image = testingImages(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult1(i,1) = KNNTesting(testXpca, modelKNN, 3);
end
%% Evaluation #1
comparison = (testingLabels==classificationResult1);
accuracy1 = sum(comparison)/length(comparison);






%% ROUND 2  - trainingG1G3, testingG2
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (trainingG1G3.images);
trainingLabels = trainingG1G3.labels;
testingImages = testingG2.images;
testingLabels = testingG2.labels;

%% Training #2 
modelKNN.neighbours = Xpca;
modelKNN.labels = trainingLabels;

%% testing #2
for i=1:size(testingImages,1)
    image = testingImages(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult2(i,1) = KNNTesting(testXpca, modelKNN, 3);
end
%% Evaluation #2
comparison = (testingLabels==classificationResult2);
accuracy2 = sum(comparison)/length(comparison);








%% ROUND 3  - trainingG2G3, testingG1
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (trainingG2G3.images);
trainingLabels = trainingG2G3.labels;
testingImages = testingG1.images;
testingLabels = testingG1.labels;

%% Training #3
modelKNN.neighbours = Xpca;
modelKNN.labels = trainingLabels;

%% testing #3
for i=1:size(testingImages,1)
    image = testingImages(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult3(i,1) = KNNTesting(testXpca, modelKNN, 3);
end
%% Evaluation #3
comparison = (testingLabels==classificationResult3);
accuracy3 = sum(comparison)/length(comparison);




%% AVERAGING ROUNDS
average = (accuracy1 + accuracy2 + accuracy3) / 3;