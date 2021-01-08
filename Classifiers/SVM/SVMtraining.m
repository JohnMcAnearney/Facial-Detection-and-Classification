function model = SVMtraining(images, labels, feature_extractor)

%Feature Extractor: option: 1 = HOG, 2 = EDGE, 3 = raw pixels, 4 = Gabor
svm_parameters = SetSVMParameters(feature_extractor);
    
%Set parameters
C = svm_parameters.C;
lambda = svm_parameters.lambda;
kernel = svm_parameters.kernel;
kerneloption = svm_parameters.kerneloption;

%Check if images are empty, if so perform training without a feature
%extractor
if images == 0 
    trainingImagesLoc = 'face_train.cdataset';
    imagesAndLabels = loadFaceImages(trainingImagesLoc);
    images = imagesAndLabels.images;
    labels = imagesAndLabels.labels;
end

% first we check if the problem is binary classification or multiclass
if max(labels)<2
    %binary classification
    model.type='binary';
    
    %SVM software requires labels -1 or 1 for the binary problem
    labels(labels==0)=-1;

    % Calculate the support vectors
    [xsup,w,w0,pos,tps,alpha] = svmclass(images,labels,C,lambda,kernel,kerneloption,1); 

    % create a structure encapsulating all teh variables composing the model
    model.xsup = xsup;
    model.w = w;
    model.w0 = w0;

    model.param.kerneloption=kerneloption;
    model.param.kernel=kernel;
    
    
else
    %multiple class classification
     model.type='multiclass';
    
    %SVM software requires labels from 1 to N for the multi-class problem
    labels = labels+1;
    nbclass=max(labels);
    
    %Initilaise and setup SVM parameters
    lambda = 1e-20;  
    C = inf;
    kerneloption= 4;
    kernel='gaussian';
    
    % Calculate the support vectors
    [xsup,w,b,nbsv]=svmmulticlassoneagainstall(images,labels,nbclass,C,lambda,kernel,kerneloption,1);
    
    % create a structure encapsulating all teh variables composing the model
    model.xsup = xsup;
    model.w = w;
    model.b = b;
    model.nbsv = nbsv;

    model.param.kerneloption=kerneloption;
    model.param.kernel=kernel;
    
end



end
