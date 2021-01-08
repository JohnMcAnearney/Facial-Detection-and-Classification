function result_array = non_maxima_supression(Objects,threshold, confidence)
%UNTITLED Summary of this function goes here

%hold index of boxes to remove
temp_array = [];
index_counter = 1;
size_objects = size(Objects);

for i=1:size_objects(1)-1
    for j=i+1:size_objects(1)
        intersection = rectint(Objects(i,1:4),Objects(j,1:4));
        boundbox_area = Objects(j,3)* Objects(j,4);
        if intersection/boundbox_area > threshold
            if confidence == true
                if Objects(i,5) > Objects(j,5)
                    temp_array(index_counter) = i;
                    index_counter = index_counter + 1;
                else
                    temp_array(index_counter) = j;
                    index_counter = index_counter + 1;
                end
            end
        else
            %if i is the same rectangle as j, do nothing 
        end
    end
end

%delete bounding boxes
index_counter = 1;
size_temp_array = size(temp_array);
result_array = [];
result_array_counter = 1;
for i=1: size_objects(1)
    is_member = ismember(i,temp_array);
    if is_member == false
        result_array(result_array_counter,1:4) = Objects(i,1:4);
        result_array_counter = result_array_counter + 1;
    end
end

end

