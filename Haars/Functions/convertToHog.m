function [hogImages] = convertToHog(images)
hogImages = [];
for i=1:size(images,1)
    image= images(i,:);
    reShaped = reshape(image, 27, []);
    hogIm = hog_feature_vector(reShaped);
    hogImages= [hogImages; hogIm];
end
end

