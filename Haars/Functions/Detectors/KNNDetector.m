function boundingBoxes = KNNDetector(image, window_size, feature_extractor)
%Detector for KNN, image location passed in as a string 
% window_size(1) = length, windows_size(2) = width/height
%feature_extraction represents feature extractor 1 = HOG, 2 = EDGE, 3 = raw pixels

image = imread(image);

[columns, rows] = size(image);
number_boxes = 0;

%Training
model = KNNtraining(feature_extractor,0);


for i=1: rows-window_size(1)
    for j=1: rows-window_size(2)
        number_boxes = number_boxes + 1;
    end
end

results = zeros(number_boxes,1);
boundingBoxes = zeros(number_boxes, 5);
result_counter = 1;
boundingboxes_counter = 1;
for x=1: rows-window_size(1)
    for y=1: columns-window_size(2)
        window = [x, y, window_size(1)-1, window_size(2)-1];
        cropped_image = imcrop(image, window);
        
        if feature_extractor == 1
            hogImage = hog_feature_vector(cropped_image);
            [prediction, confidence] = KNNTesting(hogImage, model, 11);
            
        elseif feature_extractor == 2
             edgeImage = convertToEdge_single_image(cropped_image,'canny',size(cropped_image));
            [prediction, confidence] = KNNTesting(edgeImage, model, 11);
        
        else
            [prediction, confidence] = KNNTesting(cropped_image, model, 11);
            
        end
        
        results(result_counter, 1:2) = [prediction, confidence];
        
        if results(result_counter) == 1
            boundingBoxes(boundingboxes_counter,1:4) = window;
            boundingBoxes(boundingboxes_counter,5) = confidence;
            boundingboxes_counter = boundingboxes_counter + 1;
        %imshow(cropped_image);
        
        result_counter = result_counter + 1;
        end
    end
end
end

