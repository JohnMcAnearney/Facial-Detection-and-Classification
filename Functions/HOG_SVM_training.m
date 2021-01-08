function [modelSVM] = HOG_SVM_training()
%UNTITLED3 Summary of this function goes here

%   Detailed explanation goes here
trainingImagesLoc = 'face_train.cdataset';
imagesAndLabels = loadFaceImages(trainingImagesLoc);
trainingImages = imagesAndLabels.images;
trainingLabels = imagesAndLabels.labels;

hogImages = convertToHog(trainingImages);
hogMtx.labels = imagesAndLabels.labels;
hogMtx.images = hogImages;

%Preform SVM training with data from HOG
modelSVM = SVMtraining(hogMtx.images, hogMtx.labels, 1);
end

