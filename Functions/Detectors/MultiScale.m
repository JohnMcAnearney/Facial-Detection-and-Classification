function [most_detected_boxes, scale] = MultiScale(image,windowSize,detector,scale)
%Run multi scale window sizes on detectors
%   Detailed explanation goes here

%read Image
image_array = imread(image);
%Get image size
[height, width] = size(image_array);

%Get widht and height scales
ScaleHeight = height/windowSize(1);
ScaleWidth = width/windowSize(2);

if(ScaleHeight < ScaleWidth ) 
    StartScale =  ScaleHeight; 
else
    StartScale = ScaleWidth;
end

%Calculate maximum of search scale itterations
itt=ceil(log(1/StartScale)/log(scale));

%Perform face detection, looping through all image scales
most_detected_boxes = [];
current_number_of_faces_detected = 0;
best_scale = 0;
for i=1:itt
    %Set scale
    Scale =StartScale*scale^(i-1); 
    
    %Set Width and Height of window size
    w = floor(windowSize(1)*Scale);
    h = floor(windowSize(2)*Scale);
    
    %determine which detector to use, store the scale results which
    %contains the most positive detections
    if detector == 1
        bounding_boxes = SVMDetector(image, [w,h], 1);
        [rows,columns] = size(bounding_boxes);
        number_of_face_detected = rows;
        if number_of_face_detected > current_number_of_faces_detected
            current_number_of_faces_detected = number_of_face_detected;
            most_detected_boxes = bounding_boxes;
            best_scale = Scale;
        end
        
    elseif detector == 2
        bounding_boxes = KNNDetector(image, [w,h], 1);
        [rows,columns] = size(bounding_boxes);
        number_of_face_detected = rows;
        if number_of_face_detected > current_number_of_faces_detected
            current_number_of_faces_detected = number_of_face_detected;
            most_detected_boxes = bounding_boxes;
            best_scale = Scale;
    end
    end
end

end

