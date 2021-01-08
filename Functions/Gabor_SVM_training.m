function [modelSVM] = Gabor_SVM_training()
%UNTITLED3 Summary of this function goes here

%   Detailed explanation goes here
trainingImagesLoc = 'face_train.cdataset';
imagesAndLabels = loadFaceImages(trainingImagesLoc);
trainingImages = imagesAndLabels.images;
trainingLabels = imagesAndLabels.labels;

%Get the assoicated hog vector for every training image
gaborImages = convertToGabor(trainingImages);

garMtx.labels = trainingLabels;
garMtx.images = gaborImages;

%Preform SVM training with data from HOG
modelSVM = SVMtraining(garMtx.images, garMtx.labels, 4);
end

