function [modelSVM] = Edge_SVM_training()
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
trainingImagesLoc = 'face_train.cdataset';
imagesAndLabels = loadFaceImages(trainingImagesLoc);
trainingImages = imagesAndLabels.images;
trainingLabels = imagesAndLabels.labels;

edgeImages = convertToEdge(trainingImages, 'canny');
edgeMtx.labels = imagesAndLabels.labels;
edgeMtx.images = edgeImages;

%Preform SVM training with data from HOG
modelSVM = SVMtraining(edgeMtx.images, edgeMtx.labels, 2);
end

