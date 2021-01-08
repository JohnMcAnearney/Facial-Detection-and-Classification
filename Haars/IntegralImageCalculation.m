function Iout = IntegralImageCalculation(imageArray)

% Calculate the integral image

imageXCounter = 1;
imageYCounter = 1;
sum = 0;
sizeOfMatrix = size(imageArray);
integralImage = zeros(sizeOfMatrix(1), sizeOfMatrix(2),'double');

temp_Image = cumsum(double(imageArray));

for i=1:sizeOfMatrix(1)
    for j=1:sizeOfMatrix(2)
        sum = 0;
        imageXCounter = i;
        imageYCounter = j;
            while imageYCounter>=1
                sum = sum + double(temp_Image(imageXCounter, imageYCounter));
                imageYCounter = imageYCounter - 1;
            end
        integralImage(i, j) = sum;
    end
end
Iout = integralImage;
end