clear all
close all

%% PreProcessing
% Loading all images
allImagesLoc = 'face_train_all.cdataset';
images = loadFaceImages(allImagesLoc);

% Breaking images into groups for the various of training and testing with kfold; 
[trainingG1G2, trainingG1G3, trainingG2G3, testingG1, testingG2, testingG3 ] = crossValidationSelection(images.images, images.labels);

%calculations for ROC
probTPF = [];
probFPR= [];

%% Running 3 rounds of SVM training and testing with HOG extrated data to get average accuracy

%% ROUND 1 - trainingG1G2, testingG3;
hogTrainingImages = convertToHog(trainingG1G2.images);
hogTrainingLabels = trainingG1G2.labels;

hogTestingImages = convertToHog(testingG3.images);
hogTestingLabels = testingG3.labels;

%% Training #1 
%Preform SVM training with data from hog extractopm
modelSVM = SVMtraining(hogTrainingImages, hogTrainingLabels, 1);

%% testing #1
for i=1:size(hogTestingImages,1)
    testnumber= hogTestingImages(i,:);
    classificationResult1(i,1) = SVMTesting(testnumber,modelSVM);
end
%% Evaluation #1
comparison = (hogTestingLabels==classificationResult1);
accuracy1 = sum(comparison)/length(comparison);
[FPR1, TPR1] = errorProbalityCalculator(classificationResult1,hogTestingLabels);
probTPF = [probTPF; TPR1];
probFPR = [probFPR; FPR1];





%% ROUND 2  - trainingG1G3, testingG2
hogTrainingImages = convertToHog(trainingG1G3.images);
hogTrainingLabels = trainingG1G3.labels;

hogTestingImages = convertToHog(testingG2.images);
hogTestingLabels = testingG2.labels;
%% Training #2
modelSVM = SVMtraining(hogTrainingImages, hogTrainingLabels, 1);

%% Testing #2
for i=1:size(hogTestingImages,1)
    testnumber= hogTestingImages(i,:);
    classificationResult2(i,1) = SVMTesting(testnumber,modelSVM);
end

%% Evaluation #2
comparison = (hogTestingLabels==classificationResult2);
accuracy2 = sum(comparison)/length(comparison);
[FPR2, TPR2] = errorProbalityCalculator(classificationResult2,hogTestingLabels);
probTPF = [probTPF; TPR2];
probFPR = [probFPR; FPR2];






%% ROUND 3  - trainingG2G3, testingG1
hogTrainingImages = convertToHog(trainingG2G3.images);
hogTrainingLabels = trainingG2G3.labels;

hogTestingImages = convertToHog(testingG1.images);
hogTestingLabels = testingG1.labels;

%% Training #3
modelSVM = SVMtraining(hogTrainingImages, hogTrainingLabels, 1);

%% Testing #3
for i=1:size(hogTestingImages,1)
    testnumber= hogTestingImages(i,:);
    classificationResult3(i,1) = SVMTesting(testnumber,modelSVM);
end
%% Evaluation #3
comparison = (hogTestingLabels==classificationResult3);
accuracy3 = sum(comparison)/length(comparison);
[FPR3, TPR3] = errorProbalityCalculator(classificationResult3,hogTestingLabels);
probTPF = [probTPF; TPR3];
probFPR = [probFPR; FPR3];




%% AVERAGING ROUNDS
average = (accuracy1 + accuracy2 + accuracy3) / 3;