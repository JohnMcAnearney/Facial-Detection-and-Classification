function showDetectorResults(image,boundingBoxes, includeNMS, NMSConfidence)
%Draw bounding boxes on the image indicateding a face

if includeNMS == true
    boundingBoxes = non_maxima_supression(boundingBoxes, 0.5, NMSConfidence);
end

figure,imshow(image), hold on;
% Show the detected objects
if(~isempty(boundingBoxes))
    for n=1:size(boundingBoxes,1)
        x1=boundingBoxes(n,1); y1=boundingBoxes(n,2);
        x2=x1+boundingBoxes(n,3); y2=y1+boundingBoxes(n,4);
        
        
        plot([x1 x1 x2 x2 x1],[y1 y2 y2 y1 y1], 'b');
    end
end
end

