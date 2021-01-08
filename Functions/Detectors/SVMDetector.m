function boundingBoxes = SVMDetector(image, window_size, feature_extractor)
%Detector for SVM, image location passed in as a string or an array
% window_size(1) = length, windows_size(2) = width/height
%feature_extractor represents feature extractor 1 = HOG, 2 = EDGE, 3 = raw pixels

image = imread(image);
[columns, rows] = size(image);
number_boxes = 0;



%Training
if feature_extractor == 1
    %Train for HOG
    model = HOG_SVM_training();
    
elseif feature_extractor == 2
    %Train for EDGE
    model = Edge_SVM_training();
    
elseif feature_extractor == 3
    %Train for Raw
    model = SVMtraining(0,0,feature_extractor);
    
else
    %Train for Gabor
    model = Gabor_SVM_training();
end

tic
%Initialize required variables
results = zeros(number_boxes,1);
boundingBoxes = [];
result_counter = 1;
boundingboxes_counter = 1;

for x=1: rows-window_size(1)
    for y=1: columns-window_size(2)
        window = [x, y, window_size(1), window_size(2)];
        %crop part of the image and store it for testing
        cropped_image = imcrop(image, [x, y, window_size(1)-1,window_size(2)-1 ]);
        %resize to training data size if required
        if window_size(1) ~= 18 || window_size(2) ~= 27
            cropped_image = imresize(cropped_image, [18,27]);
        end
        %determine which feature extractor is chosen and test the image
        %using feature extractor pre processing
        if feature_extractor == 1
            hogImage = hog_feature_vector(cropped_image);
            results(result_counter) = SVMTesting(hogImage, model);
            
        elseif feature_extractor == 2
            edgeImage = convertToEdge_single_image(cropped_image,'canny',size(cropped_image));
            results(result_counter) = SVMTesting(edgeImage, model);
        
        elseif feature_extractor == 3
            cropped_image = reshape(cropped_image, 1, []);
            cropped_image = double(cropped_image);
            results(result_counter) = SVMTesting(cropped_image, model);
            
        elseif feature_extractor == 4
            reShaped = reshape(cropped_image, 27, []);
            reShapedUint8 = uint8(reShaped);
            gaborImage = gabor_feature_vector(reShapedUint8);
            results(result_counter) = SVMTesting(gaborImage, model);
        end
        
        %If the classifier thinks the current cropped image is a face,
        %store the box 
        if results(result_counter) == 1
            boundingBoxes(boundingboxes_counter,1:4) = window;
            boundingboxes_counter = boundingboxes_counter + 1;
        %imshow(cropped_image);
        end
        result_counter = result_counter + 1;
    end
end
time_taken_to_detect = toc
end

