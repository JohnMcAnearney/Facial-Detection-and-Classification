function Objects = non_maxima_supression(Objects,threshold, confidence)
%UNTITLED Summary of this function goes here

%hold index of boxes to remove
index_array = [];
index_counter = 1;
[rows, columns] = size(Objects);
area_counter = 1;

for i=1:rows-1
    for j=i:rows
        if ~isempty(Objects(i)) || ~isempty(Objects(j))
            if i ~= j
                intersection = rectint(Objects(i,1:4),Objects(j,1:4));
                boundbox_area = Objects(j,3)* Objects(j,4);
                if intersection/boundbox_area > threshold && ~ismember(j,index_array)
                    if confidence == true
                        if Objects(i,5) > Objects(j,5)
                            index_array(index_counter) = j;
                        else
                            index_array(index_counter) = i;
                        end
                    end
                    index_array(index_counter) = j;
                    index_counter = index_counter + 1;
                end
            else
            %if i is the same rectangle as j, do nothing 
            end
        end
    end
end
%delete bounding boxes
temp_array = [];
index_counter = 1;
[rows, columns] = size(index_array);
for i=1:columns
    if ~ismember(i,index_array)
        temp_array(index_counter,1:4) = Objects(i,1:4);
        index_counter = index_counter + 1;
    end
end
Objects = temp_array;
end

