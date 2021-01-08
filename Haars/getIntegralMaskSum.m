function [integralMaskSum] = getIntegralMaskSum(image, x, y, width, height)
%Return sum when mask applied onto an image
%   Detailed explanation goes here
integralImage =  calculatePaddedIntegral(image);
if x-width <=0 || y-width <= 0
    integralMaskSum = integralImage(x,y);
    return
end
t1 = integralImage(x,y);
t2 = integralImage(x-height, y);
t3 = integralImage(x,y-width);
t4 = integralImage(x - width,y-height);
integralMaskSum = t1 - t2 - t3 + t4;

end

