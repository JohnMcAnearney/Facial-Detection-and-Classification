function [edgeImages] = convertToEdge(images, type)
edgeImages = [];
for i=1:size(images,1)
    image= images(i,:);
    reshaped = reshape(image, 27, []);
    switch type
        case 'canny'
              edgeIm = edge(reshaped, 'canny'); 
        case 'sobel'
              edgeIm = edge(reshaped, 'sobel'); 
        case 'Prewitt'
              edgeIm = edge(reshaped, 'Prewitt'); 
        case 'Roberts'
              edgeIm = edge(reshaped, 'Roberts'); 
        otherwise 
              edgeIm = edge(reshaped, 'canny');
    end
    revertReshape = reshape(edgeIm, [1, 486]);
    edgeImages = [edgeImages; revertReshape];
end
end
