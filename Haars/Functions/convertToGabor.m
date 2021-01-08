function [gaborImages] = convertToGabor(images)
gaborImages = [];
for i=1:size(images,1)
    image= images(i,:);
    reShaped = reshape(image, 27, []);
    reShapedUint8 = uint8(reShaped);
    gaborIm = gabor_feature_vector(reShapedUint8);
    gaborImages= [gaborImages; gaborIm];
end
end

