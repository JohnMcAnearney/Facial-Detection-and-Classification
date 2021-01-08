function [enhancedImageSet] = preprocessing(inputImageSet)

%%Preprocessing funciton that will apply all preprocessing to images

enhancedImageSet = zeros(size(inputImageSet, 1), size(inputImageSet, 2));

%Contrast enhancement
for i=1:size(inputImageSet,1)
    image= inputImageSet(i,:);
    %convert to 8-bit so imadjust can adjust the images contrast
    eightBitImage = uint8(image);
    %enhance contrast
    adjustedImage = imadjust(eightBitImage);
    for j=1:size(inputImageSet, 2)
        enhancedImageSet(i, j) = adjustedImage(1, j);
    end
end

%Image brightness - can also enhance brightness before changing contrast, 
%just change the enhancedImageSet to inputImageSet and call before contrast enhancement 

for i=1:size(enhancedImageSet,1)
    totalImagePixels = size(enhancedImageSet,2);
    %get count of each pixel value
    [GC, GR] = groupcounts(enhancedImageSet(i,:)');
    
    %get count of each value 10 and under 
    sumCount = 0;
    for j=1:10
        if GR(j) < 11
            sumCount = sumCount + GC(j);
        end 
    end
    %if its a dark image, increase brightness slightly
    if sumCount > totalImagePixels / 10
        enhancedImageSet(i,:) = enhancedImageSet(i,:) + 5;
    end
end
end

