function [true_positives,false_positives,true_negatives,false_negatives, accuracy] = Detector_Evaluator(image, boundingBoxes, window_size)
%Calculate True positives, False positives, True Negatives, False Negatives 
%   Detailed explanation goes here
imageArray = imread(image);
[imageRows, imageColumns] = size(imageArray);
[rows,columns] = size(boundingBoxes);
number_of_boxes = rows;
threshold_is_face = 0.7;

%Set faces as rectangles in an array from each final image
if image == 'im1.jpg'
    number_of_faces = 7;
    array_of_faces = zeros(7,4);
    array_of_faces(1,1:4) = [7,19,22,33];
    array_of_faces(2,1:4) = [26,14,19,26];
    array_of_faces(3,1:4) = [46,24,20,26];
    array_of_faces(4,1:4) = [63,11,19,29];
    array_of_faces(5,1:4) = [81,26,19,25];
    array_of_faces(6,1:4) = [99,14,20,22];
    array_of_faces(7,1:4) = [118,24,22,26];
    
elseif image == 'im2.jpg'
    number_of_faces = 15;
    array_of_faces(1,1:4) = [3,3,25,30];
    array_of_faces(2,1:4) = [28,3,25,30];
    array_of_faces(3,1:4) = [53,3,25,30];
    array_of_faces(4,1:4) = [78,3,25,30];
    array_of_faces(5,1:4) = [103,3,25,30];
    
    array_of_faces(6,1:4) = [3,34,25,30];
    array_of_faces(7,1:4) = [28,34,25,30];
    array_of_faces(8,1:4) = [78,34,25,30];
    array_of_faces(9,1:4) = [78,34,25,30];
    array_of_faces(10,1:4) = [103,34,25,30];
    
    array_of_faces(11,1:4) = [3,63,25,30];
    array_of_faces(12,1:4) = [28,63,25,30];
    array_of_faces(13,1:4) = [53,63,25,30];
    array_of_faces(14,1:4) = [78,63,25,30];
    array_of_faces(15,1:4) = [103,63,25,30];
    
    
elseif image == 'im3.jpg'
    number_of_faces = 8;
    array_of_faces(1,1:4) = [25,77,22,23];
    array_of_faces(2,1:4) = [37,44,21,25];
    array_of_faces(3,1:4) = [16,33,22,24];
    array_of_faces(4,1:4) = [62,51,21,22];
    array_of_faces(5,1:4) = [68,21,19,21];
    array_of_faces(6,1:4) = [92,30,21,27];
    array_of_faces(7,1:4) = [92,76,19,21];
    array_of_faces(8,1:4) = [116,57,20,24];
    
elseif image == 'im4.jpg'
    number_of_faces = 57;
    array_of_faces(1,1:4) = [16,186,22,30];
    array_of_faces(2,1:4) = [24,114,23,25];
    array_of_faces(3,1:4) = [50,86,19,27];
    array_of_faces(4,1:4) = [61,71,22,24];
    array_of_faces(5,1:4) = [43,53,22,24];
    array_of_faces(6,1:4) = [67,33,17,20];
    array_of_faces(7,1:4) = [103,46,19,22];
    array_of_faces(8,1:4) = [106,89,24,25];
    array_of_faces(9,1:4) = [89,117,22,23];
    array_of_faces(10,1:4) = [80,140,19,30];
    array_of_faces(11,1:4) = [113,187,22,24];
    array_of_faces(12,1:4) = [139,155,22,25];
    array_of_faces(13,1:4) = [145,119,21,25];
    array_of_faces(14,1:4) = [138,83,22,26];
    array_of_faces(15,1:4) = [155,51,22,23];
    array_of_faces(16,1:4) = [140,30,19,22];
    array_of_faces(17,1:4) = [176,117,17,20];
    array_of_faces(18,1:4) = [206,178,27,29];
    array_of_faces(19,1:4) = [208,112,24,25];
    array_of_faces(20,1:4) = [192,82,19,25];
    array_of_faces(21,1:4) = [207,61,19,22];
    array_of_faces(22,1:4) = [205,19,19,23];
    array_of_faces(23,1:4) = [234,45,17,22];
    array_of_faces(24,1:4) = [254,65,22,24];
    array_of_faces(25,1:4) = [256,98,22,27];
    array_of_faces(26,1:4) = [272,140,19,27];
    array_of_faces(27,1:4) = [301,184,23,28];
    array_of_faces(28,1:4) = [303,127,23,26];
    array_of_faces(29,1:4) = [292,74,15,24];
    array_of_faces(30,1:4) = [283,50,17,22];
    array_of_faces(31,1:4) = [265,24,17,23];
    array_of_faces(32,1:4) = [320,24,20,20];
    array_of_faces(33,1:4) = [334,58,19,23];
    array_of_faces(34,1:4) = [321,87,21,29];
    array_of_faces(35,1:4) = [338,149,19,25];
    array_of_faces(36,1:4) = [373,187,24,29];
    array_of_faces(37,1:4) = [390,143,19,29];
    array_of_faces(38,1:4) = [362,128,22,27];
    array_of_faces(39,1:4) = [386,111,21,27];
    array_of_faces(40,1:4) = [364,75,20,24];
    array_of_faces(41,1:4) = [381,65,18,21];
    array_of_faces(42,1:4) = [378,31,21,22];
    array_of_faces(43,1:4) = [404,30,20,21];
    array_of_faces(44,1:4) = [408,64,19,20];
    array_of_faces(45,1:4) = [439,92,21,23];
    array_of_faces(46,1:4) = [447,126,23,25];
    array_of_faces(47,1:4) = [451,181,24,28];
    array_of_faces(48,1:4) = [470,147,21,29];
    array_of_faces(49,1:4) = [529,192,26,29];
    array_of_faces(50,1:4) = [549,145,23,27];
    array_of_faces(51,1:4) = [510,117,24,25];
    array_of_faces(52,1:4) = [496,97,17,24];
    array_of_faces(53,1:4) = [519,73,20,27];
    array_of_faces(54,1:4) = [487,43,20,23];
    array_of_faces(55,1:4) = [455,66,18,25];
    array_of_faces(56,1:4) = [436,46,19,23];
    array_of_faces(57,1:4) = [465,31,18,23];
    
end

%showDetectorResults(image,array_of_faces, false, false)


%Detect True positives and False positives
true_positives_array = [];
true_positive_counter = 1;

box_counter1 = 0;
%Iterate through all boxes and check if the box intersects with the face
%box over the threshold value
for i=1:number_of_boxes
    for j=1:number_of_faces
        box_counter1 = box_counter1 +1;
        intersection = rectint(array_of_faces(j,1:4),boundingBoxes(i,1:4));
        bounded_box_area = boundingBoxes(i,3) * boundingBoxes(i,4);
        if intersection/bounded_box_area > threshold_is_face
            true_positives_array_size = size(true_positives_array);
            if true_positives_array_size(1) == 0
                true_positives_array(true_positive_counter, 1:4) = boundingBoxes(i,1:4);
                true_positive_counter = true_positive_counter + 1;
            else
                %duplicate boxes being stored prevention
                duplicate = false;
                for k=1: true_positives_array_size(1)
                    is_equal_results = true_positives_array(k,1:4) == boundingBoxes(i,1:4);
                    if ~ismember(false,is_equal_results)
                        duplicate = true;
                    end
                end
                %if not duplicate store the box as it overlaps with a face
                %box over the threshold value
                if duplicate == false
                    true_positives_array(true_positive_counter, 1:4) = boundingBoxes(i,1:4);
                    true_positive_counter = true_positive_counter + 1;
                end
            end
            
        end
    end
end


array = true_positives_array;
%Calulate True positives and False positives
true_positives = true_positive_counter-1;

false_positives = number_of_boxes - true_positives;

%Calculate max number of windows
number_of_possible_boxes = 0;
no_of_windows_face_intersection = 0;
true_faces_array = [];
for x=1:imageColumns-window_size(1)
    for y=1: imageRows-window_size(2)
        number_of_possible_boxes = number_of_possible_boxes + 1;
        %Calculate max number of windows that intersect with the faces
        %for false negative calculation
        for i=1:number_of_faces
            intersection = rectint([x,y,window_size(1), window_size(2)], array_of_faces(i,1:4));
            bounded_box_area = boundingBoxes(i,3) * boundingBoxes(i,4);
            bounding_box_area = window_size(1)* window_size(2);
            if intersection/bounded_box_area > threshold_is_face
                no_of_windows_face_intersection = no_of_windows_face_intersection + 1;
                true_faces_array(no_of_windows_face_intersection, 1:4) = [x,y,window_size(1), window_size(2)];
            end
        end
    end
end

%Calculate False Negatives
false_negatives = no_of_windows_face_intersection - true_positives;

%Calculate True Negatives
true_negatives = number_of_possible_boxes - number_of_boxes;

%Accuracy 
accuracy = (true_negatives + true_positives)/number_of_possible_boxes;
end

