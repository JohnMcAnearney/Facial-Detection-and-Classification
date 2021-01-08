function [trainingG1G2, trainingG1G3, trainingG2G3, testingG1, testingG2, testingG3 ] = crossValidationSelection(images, labels, type)
cvIndices  = crossvalind('kFold',labels,3);

%Divide the image set into three groups
%Two will be used for training and the other testing
%G3 is the testing set
ImagesG1 = [];
ImagesG2  = [];
ImagesG3 = [];
LabelsG1 = [];
LabelsG2  = [];
LabelsG3 = [];

for i =1:size(cvIndices,1)
   if(cvIndices(i,:) == 1)
       ImagesG1 = [ImagesG1; images(i,:)];
       LabelsG1 = [LabelsG1; labels(i,:)];
   end
    if(cvIndices(i,:) == 2)
       ImagesG2 = [ImagesG2; images(i,:)];
       LabelsG2 = [LabelsG2; labels(i,:)];
   end
    if(cvIndices(i,:) == 3)
       ImagesG3 = [ImagesG3; images(i,:)];
       LabelsG3 = [LabelsG3; labels(i,:)];
   end    
end

trainingG1G2Images = [];
trainingG1G2Labels = [];
trainingG1G2Images = [trainingG1G2Images; ImagesG1];
trainingG1G2Images = [trainingG1G2Images; ImagesG2];
trainingG1G2Labels = [trainingG1G2Labels; LabelsG1];
trainingG1G2Labels = [trainingG1G2Labels; LabelsG2];

trainingG1G3Images = [];
trainingG1G3Labels = [];
trainingG1G3Images = [trainingG1G3Images; ImagesG1];
trainingG1G3Images = [trainingG1G3Images; ImagesG3];
trainingG1G3Labels = [trainingG1G3Labels; LabelsG1];
trainingG1G3Labels = [trainingG1G3Labels; LabelsG3];

trainingG2G3Images = [];
trainingG2G3Labels = [];
trainingG2G3Images = [trainingG2G3Images; ImagesG2];
trainingG2G3Images = [trainingG2G3Images; ImagesG3];
trainingG2G3Labels = [trainingG2G3Labels; LabelsG2];
trainingG2G3Labels = [trainingG2G3Labels; LabelsG3];

trainingG1G2 = [];
trainingG1G2.images = trainingG1G2Images;
trainingG1G2.labels = trainingG1G2Labels;

trainingG1G3 = [];
trainingG1G3.images = trainingG1G3Images;
trainingG1G3.labels = trainingG1G3Labels;

trainingG2G3 = [];
trainingG2G3.images = trainingG2G3Images;
trainingG2G3.labels = trainingG2G3Labels;

testingG1 = [];
testingG1.images = ImagesG1;
testingG1.labels = LabelsG1;

testingG2 = [];
testingG2.images = ImagesG2;
testingG2.labels = LabelsG2;

testingG3 = [];
testingG3.images = ImagesG3;
testingG3.labels = LabelsG3;
end

