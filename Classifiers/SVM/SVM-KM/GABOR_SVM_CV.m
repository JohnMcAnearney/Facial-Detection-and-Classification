clear all
close all

%% PreProcessing
% Loading all images
allImagesLoc = 'face_train_all.cdataset';
images = loadFaceImages(allImagesLoc);

% Breaking images into sets; 
[trainingG1G2, trainingG1G3, trainingG2G3, testingG1, testingG2, testingG3 ] = crossValidationSelection(images.images, images.labels);





%% ROUND 1 - trainingG1G2, testingG3;
gaborTrainingImages = convertToGarbor(trainingG1G2.images);
gaborTrainingLabels = trainingG1G2.labels;

gaborTestingImages = convertToGarbor(testingG3.images);
gaborTestingLabels = testingG3.labels;

%% Training #1 
modelSVM = SVMtraining(gaborTrainingImages, gaborTrainingLabels, 4);

%% testing #1
for i=1:size(gaborTestingImages,1)
    edIm = reshape(gaborTestingImages(i,:), 27, []);
    testnumber= gaborTestingImages(i,:);
    classificationResult1(i,1) = SVMTesting(testnumber,modelSVM);
end
%% Evaluation #1
comparison = (gaborTestingLabels==classificationResult1);
accuracy1 = sum(comparison)/length(comparison);




%% ROUND 2  - trainingG1G3, testingG2
gaborTrainingImages = convertToGarbor(trainingG1G3.images);
gaborTrainingLabels = trainingG1G3.labels;

gaborTestingImages = convertToGarbor(testingG2.images);
gaborTestingLabels = testingG2.labels;
%% Training #2
modelSVM = SVMtraining(gaborTrainingImages, gaborTrainingLabels, 4);

%% Testing #2
for i=1:size(gaborTestingImages,1)
    edIm = reshape(gaborTestingImages(i,:), 27, []);
    testnumber= gaborTestingImages(i,:);
    classificationResult2(i,1) = SVMTesting(testnumber,modelSVM);
end

%% Evaluation #2
comparison = (gaborTestingLabels==classificationResult2);
accuracy2 = sum(comparison)/length(comparison);







%% ROUND 3  - trainingG2G3, testingG1
gaborTrainingImages = convertToGarbor(trainingG2G3.images, 'canny');
gaborTrainingLabels = trainingG2G3.labels;

gaborTestingImages = convertToGarbor(testingG1.images, 'canny');
gaborTestingLabels = testingG1.labels;

%% Training #3
modelSVM = SVMtraining(gaborTrainingImages, gaborTrainingLabels, 4);

%% Testing #3
for i=1:size(gaborTestingImages,1)
    edIm = reshape(gaborTestingImages(i,:), 27, []);
    testnumber= gaborTestingImages(i,:);
    classificationResult3(i,1) = SVMTesting(testnumber,modelSVM);
end
%% Evaluation #3
comparison = (gaborTestingLabels==classificationResult3);
accuracy3 = sum(comparison)/length(comparison);






%% AVERAGING ROUNDS
average = (accuracy1 + accuracy2 + accuracy3) / 3;