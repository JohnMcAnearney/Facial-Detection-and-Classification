
function modelNN = KNNtraining(feature_extractor, preprocessFlag)
%type represents feature extractor 1 = HOG, 2 = EDGE, 3 = raw pixels, 4 =
%Gabor

%%Training and setup
%load images and labels
%loading here due to haw detector is setup.
trainingImagesLoc = 'face_train.cdataset';
imagesAndLabels = loadFaceImages(trainingImagesLoc);
trainingImages = imagesAndLabels.images;
trainingLabels = imagesAndLabels.labels;

if preprocessFlag == 1
   %Applying preprocessing for Images if set
    trainingImages = preprocessing(trainingImages); 
end

if feature_extractor == 1
    % apply HOG feature extraction on images
    hogImages = convertToHog(trainingImages);
    trainingImages = hogImages;
    
elseif feature_extractor == 2
    % apply Edge feature extraction on images
    edgeImages = convertToEdge(trainingImages, 'canny');
    trainingImages = edgeImages;
elseif feature_extractor == 3
    %raw images
    
elseif feature_extractor == 4
    % apply Gabor feature extraction on images
    gaborImages = convertToGabor(trainingImages);
    trainingImages = gaborImages;
end


modelNN.neighbours=trainingImages;
modelNN.labels=trainingLabels;

end
