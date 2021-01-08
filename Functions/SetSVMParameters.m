function svmParameters = SetSVMParameters(feature_extractor)
%Set the best combination of SVM Parameters according to the feature
%extractor
%Feature Extractor: option: 1 = HOG, 2 = EDGE, 3 = raw pixels, 4 = Gabor 

       
    if feature_extractor == 1
        %Best HOG   
        svmParameters.lambda = 1e-20;  
        svmParameters.C = inf;
        svmParameters.kerneloption=11;
        svmParameters.kernel='poly';
    
    elseif feature_extractor == 2
        %Best Edge
        svmParameters.lambda = 1e-20;  
        svmParameters.C = inf;
        svmParameters.kerneloption=4;
        svmParameters.kernel='gaussian';
    
    elseif feature_extractor == 3
        %Best Raw   
        svmParameters.lambda = 1e-20;  
        svmParameters.C = inf;
        svmParameters.kerneloption=5;
        svmParameters.kernel='gaussian';
    
    elseif feature_extractor == 4
        %Best Gabor   
        svmParameters.lambda = 1e-20;  
        svmParameters.C = inf;
        svmParameters.kerneloption=2;
        svmParameters.kernel='gaussian';
    end
end

