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

end

