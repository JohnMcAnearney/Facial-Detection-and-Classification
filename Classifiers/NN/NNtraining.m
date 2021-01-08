function modelNN = NNtraining(type, preprocessFlag)
%type represents feature extractor 1 = HOG, 2 = HOG, 3 = raw pixels
 
%%Training and setup
%load images and labels
trainingImagesLoc = 'face_train.cdataset';
imagesAndLabels = loadFaceImages(trainingImagesLoc);
trainingImages = imagesAndLabels.images;
trainingLabels = imagesAndLabels.labels;

if preprocessFlag == 1
   %Applying preprocessing for Images if set
    trainingImages = preprocessing(trainingImages); 
end

if type == 1
    % apply HOG feature extraction on images
    hogImages = [];
    for i=1:size(trainingImages,1)
        image= trainingImages(i,:);
        reShaped = reshape(image, 27, []);
        hogIm = hog_feature_vector(reShaped);
        %revertReshape = reshape(hogIm, 1, []);
        hogImages= [hogImages; hogIm];
    end
    trainingImages = hogImages;
    
elseif type == 2
    % apply Edge feature extraction on images
    edgeImages = [];

    for i=1:size(trainingImages,1)
        image= trainingImages(i,:);
        reshaped = reshape(image, 27, []);
        edgeIm = edge(reshaped, 'prewitt'); 
        revertReshape = reshape(edgeIm, [1, 486]);
        edgeImages = [edgeImages; revertReshape];
    end
    trainingImages = edgeImages;
else
    %raw images
end



modelNN.neighbours=trainingImages;
modelNN.labels=trainingLabels;

end
