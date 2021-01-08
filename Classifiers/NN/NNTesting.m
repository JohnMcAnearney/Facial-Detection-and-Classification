function [prediction,confidence] = NNTesting(testImage, modelNN, 1)
    confidence = 0;
    [rows, columns] = size(testImage);
    if rows ~= 27 || columns ~= 18
        testImage = imresize(testImage,[27,18]);
    end
    reShaped = reshape(testImage, 27, []);
    testImage = hog_feature_vector(reShaped);

    %go calculate the euclidian value set
    eucSet = dEuc(testImage, modelNN);
    
    %get the closest point
    %a = closest euc value, index = index of that value
    [a, index]= min(eucSet(eucSet>0));  
    
    %store closest point in array of size K + 1
    closestArray = zeros(1, K+1);                       
    closestArray(1) = a;
    %store corresponding index of that closest point
    closestIndexArray = zeros(1, K+1);
    closestIndexArray(1) = index;
    
    %Get the closest neighbours to closestPoint
    for x=2:(K+1)
        [next, indexNext]= min(eucSet(eucSet>closestArray(x-1)));
        closestArray(x) = next;
        closestIndexArray(x) = indexNext;
    end
    
    %At this point vote
    %If there are more 0's than 1's in the group of voters, we set
    %0, if there are more ones in the circle, we set 1
    % 0 = no face, 1 = face
    
    minusOneCount = 0;
    oneCount = 0;
    
    for n=1:size(closestIndexArray, 2)
        val = modelNN.labels(closestIndexArray(n));
        switch val 
            case -1 
                minusOneCount = minusOneCount+1;
            case 1
                oneCount = oneCount+1;
        end
    end
    
        if(minusOneCount > oneCount)
            prediction = -1;
        else 
            prediction = 1;
            confidence = minusOneCount / oneCount;
        end
    
end

