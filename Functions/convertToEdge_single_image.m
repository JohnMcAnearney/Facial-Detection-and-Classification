function [edgeImages] = convertToEdge_single_image(image, type, size_of_image)
edgeImages = [];
reshaped = reshape(image, size_of_image(2), []);
[rows, columns] = size(image);
size_of_vector = rows * columns;
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
revertReshape = reshape(edgeIm, [1, size_of_vector]);
edgeImages = [edgeImages; revertReshape];

end

