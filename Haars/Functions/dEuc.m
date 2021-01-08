function dEuc = EuclideanDistance(sample1,sample2)

singleImageDataStoreToCalcEuc = zeros(1, size(sample1, 2));
dEuc = zeros(1, size(sample2.neighbours, 1));  %each row is a new image, so array to store euc value of each image

% for every image in training set modelNN
for x=1:size(sample2.neighbours, 1)
    
%for every value in the row of the testImage calculate first part of the
%euclidain value
    for i=1:size(sample1, 2)
        V = sample1(1, i) - sample2.neighbours(x, i);
        dEucSquare = (V * V);
        %why does matlab not let us do +=?????
        singleImageDataStoreToCalcEuc(i) = dEucSquare;
    end
    dEucTotal = sum(singleImageDataStoreToCalcEuc);
    dEucimage = sqrt(dEucTotal);
    
    %store this euc value in the eucList
    dEuc(x) = dEucimage;
end
%now return the euclidain value for it 
    
end

