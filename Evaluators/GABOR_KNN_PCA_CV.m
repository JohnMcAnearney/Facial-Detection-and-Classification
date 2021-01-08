clear all
close all

%% PreProcessing
% Loading all images
allImagesLoc = 'face_train_all.cdataset';
images = loadFaceImages(allImagesLoc);

% Breaking images into sets; 
[trainingG1G2, trainingG1G3, trainingG2G3, testingG1, testingG2, testingG3 ] = crossValidationSelection(images.images, images.labels);

%% ROUND 1 - trainingG1G2, testingG3;
gaborTrainingImages = convertToGabor(trainingG1G2.images);
gaborTrainingLabels = trainingG1G2.labels;

gaborTestingImages = convertToGabor(testingG3.images);
gaborTestingLabels = testingG3.labels;

[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (gaborTrainingImages);

%% Training #1 
modelKNN.neighbours = Xpca;
modelKNN.labels = gaborTrainingLabels;

%% testing #1
for i=1:size(gaborTestingImages,1)
    image = gaborTestingImages(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult1(i,1) = KNNTesting(testXpca,modelKNN, 1);
end
%% Evaluation #1
comparison = (gaborTestingLabels==classificationResult1);
accuracy1 = sum(comparison)/length(comparison);




%% ROUND 2  - trainingG1G3, testingG2
gaborTrainingImages = convertToGabor(trainingG1G3.images);
gaborTrainingLabels = trainingG1G3.labels;

gaborTestingImages = convertToGabor(testingG2.images);
gaborTestingLabels = testingG2.labels;

[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (gaborTrainingImages);

%% Training #2
modelKNN.neighbours = Xpca;
modelKNN.labels = gaborTrainingLabels;

%% Testing #2
for i=1:size(gaborTestingImages,1)
    image = gaborTestingImages(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult2(i,1) = KNNTesting(testXpca,modelKNN, 1);
end

%% Evaluation #2
comparison = (gaborTestingLabels==classificationResult2);
accuracy2 = sum(comparison)/length(comparison);







%% ROUND 3  - trainingG2G3, testingG1
gaborTrainingImages = convertToGabor(trainingG2G3.images);
gaborTrainingLabels = trainingG2G3.labels;

gaborTestingImages = convertToGabor(testingG1.images);
gaborTestingLabels = testingG1.labels;

%% Training #3
modelKNN.neighbours = Xpca;
modelKNN.labels = gaborTrainingLabels;

[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (gaborTrainingImages);

%% Testing #3
for i=1:size(gaborTestingImages,1)
    image = gaborTestingImages(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult3(i,1) = KNNTesting(testXpca,modelKNN, 1);
end
%% Evaluation #3
comparison = (gaborTestingLabels==classificationResult3);
accuracy3 = sum(comparison)/length(comparison);






%% AVERAGING ROUNDS
average = (accuracy1 + accuracy2 + accuracy3) / 3;