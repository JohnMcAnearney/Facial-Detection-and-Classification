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
else
    model = SVMtraining(0,0);
end

for i=1: rows-window_size(1)
    for j=1: rows-window_size(2)
        number_boxes = number_boxes + 1;
    end
end

results = zeros(number_boxes,1);
boundingBoxes = zeros(number_boxes, 2);
result_counter = 1;
boundingboxes_counter = 1;
for x=1: rows-window_size(1)
    for y=1: columns-window_size(2)
        window = [x, y, window_size(1)-1, window_size(2)-1];
        cropped_image = imcrop(image, window);
        
        if feature_extractor == 1
            hogImage = hog_feature_vector(cropped_image);
            results(result_counter) = SVMTesting(hogImage, model);
            
        elseif feature_extractor == 2
            edgeImage = convertToEdge_single_image(cropped_image,'canny',size(cropped_image));
            results(result_counter) = SVMTesting(edgeImage, model);
            
        else
            cropped_image = reshape(cropped_image, 1, []);
            results(result_counter) = SVMTesting(cropped_image, model);
        end
        
        if results(result_counter) == 1
            boundingBoxes(boundingboxes_counter,1:4) = window;
            boundingboxes_counter = boundingboxes_counter + 1;
        %imshow(cropped_image);
        
        result_counter = result_counter + 1;
        end
    end
end
end

