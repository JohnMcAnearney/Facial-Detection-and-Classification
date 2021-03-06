clear all
close all

%% PreProcessing
% Loading all images
allImagesLoc = 'face_train_all.cdataset';
images = loadFaceImages(allImagesLoc);

% Breaking images into groups for the various of training and testing with kfold; 
[trainingG1G2, trainingG1G3, trainingG2G3, testingG1, testingG2, testingG3 ] = crossValidationSelection(images.images, images.labels);


%% Running 3 rounds of SVM training and testing with HOG extrated data to get average accuracy

%% ROUND 1 - trainingG1G2, testingG3;
hogTrainingImages = convertToHog(trainingG1G2.images);
hogTrainingLabels = trainingG1G2.labels;

hogTestingImages = convertToHog(testingG3.images);
hogTestingLabels = testingG3.labels;
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (hogTrainingImages);

%% Training #1 
%Preform SVM training with data from hog extractopm
modelSVM = SVMtraining(Xpca, hogTrainingLabels, 1);

%% testing #1
for i=1:size(hogTestingImages,1)
    image = hogTestingImages(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult1(i,1) = SVMTesting(testXpca,modelSVM);
end
%% Evaluation #1
comparison = (hogTestingLabels==classificationResult1);
accuracy1 = sum(comparison)/length(comparison);






%% ROUND 2  - trainingG1G3, testingG2
hogTrainingImages = convertToHog(trainingG1G3.images);
hogTrainingLabels = trainingG1G3.labels;

hogTestingImages = convertToHog(testingG2.images);
hogTestingLabels = testingG2.labels;
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (hogTrainingImages);

%% Training #2
modelSVM = SVMtraining(Xpca, hogTrainingLabels, 1);

%% Testing #2
for i=1:size(hogTestingImages,1)
    image = hogTestingImages(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult2(i,1) = SVMTesting(testXpca,modelSVM);
end

%% Evaluation #2
comparison = (hogTestingLabels==classificationResult2);
accuracy2 = sum(comparison)/length(comparison);







%% ROUND 3  - trainingG2G3, testingG1
hogTrainingImages = convertToHog(trainingG2G3.images);
hogTrainingLabels = trainingG2G3.labels;

hogTestingImages = convertToHog(testingG1.images);
hogTestingLabels = testingG1.labels;

[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (hogTrainingImages);
%% Training #3
modelSVM = SVMtraining(Xpca, hogTrainingLabels, 1);

%% Testing #3
for i=1:size(hogTestingImages,1)
    image = hogTestingImages(i,:);
    A = image - meanX;
    testXpca = A'*eigenvalues';
    classificationResult3(i,1) = SVMTesting(testXpca,modelSVM);
end
%% Evaluation #3
comparison = (hogTestingLabels==classificationResult3);
accuracy3 = sum(comparison)/length(comparison);





%% AVERAGING ROUNDS
average = (accuracy1 + accuracy2 + accuracy3) / 3;