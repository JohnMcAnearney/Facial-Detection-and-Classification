clear all
close all
%% PreProcessing
% Loading all images
allImagesLoc = 'face_train_all.cdataset';
images = loadFaceImages(allImagesLoc);

% Breaking images into groups for the various of training and testing with kfold; 
[trainingG1G2, trainingG1G3, trainingG2G3, testingG1, testingG2, testingG3 ] = crossValidationSelection(images.images, images.labels);


%% Running 3 rounds of SVM training and testing with Edge extrated data to get average accuracy

%% ROUND 1 - trainingG1G2, testingG3;
edgeTrainingImages = convertToEdge(trainingG1G2.images, 'canny');
edgeTrainingLabels = trainingG1G2.labels;

edgeTestingImages = convertToEdge(testingG3.images, 'canny');
edgeTestingLabels = testingG3.labels;

[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (edgeTrainingImages);

%% Training #1 
%Preform SVM training with data from Edge extractopm
modelSVM = SVMtraining(Xpca, edgeTrainingLabels, 2);

%% testing #1
for i=1:size(edgeTestingImages,1)
    image = edgeTestingImages(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult1(i,1) = SVMTesting(testXpca,modelSVM);
end
%% Evaluation #1
comparison = (edgeTestingLabels==classificationResult1);
accuracy1 = sum(comparison)/length(comparison);






%% ROUND 2  - trainingG1G3, testingG2
edgeTrainingImages = convertToEdge(trainingG1G3.images, 'canny');
edgeTrainingLabels = trainingG1G3.labels;

edgeTestingImages = convertToEdge(testingG2.images, 'canny');
edgeTestingLabels = testingG2.labels;

[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (edgeTrainingImages);

%% Training #2
modelSVM = SVMtraining(Xpca, edgeTrainingLabels, 2);

%% Testing #2
for i=1:size(edgeTestingImages,1)
    image = edgeTestingImages(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult2(i,1) = SVMTesting(testXpca,modelSVM);
end

%% Evaluation #2
comparison = (edgeTestingLabels==classificationResult2);
accuracy2 = sum(comparison)/length(comparison);







%% ROUND 3  - trainingG2G3, testingG1
edgeTrainingImages = convertToEdge(trainingG2G3.images, 'canny');
edgeTrainingLabels = trainingG2G3.labels;

edgeTestingImages = convertToEdge(testingG1.images, 'canny');
edgeTestingLabels = testingG1.labels;

[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (edgeTrainingImages);

%% Training #3
modelSVM = SVMtraining(Xpca, edgeTrainingLabels, 2);

%% Testing #3
for i=1:size(edgeTestingImages,1)
    image = edgeTestingImages(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult3(i,1) = SVMTesting(testXpca,modelSVM);
end
%% Evaluation #3
comparison = (edgeTestingLabels==classificationResult3);
accuracy3 = sum(comparison)/length(comparison);





%% AVERAGING ROUNDS
average = (accuracy1 + accuracy2 + accuracy3) / 3;