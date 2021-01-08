clear
clear all

%Initialize image
image = 'im1.jpg';
window_size=[18,27];
tic
% Multi Scale detection of given image and feature extractor
%Parameters:
%image = image location
%window_size = [width,height]
%Detector: 1 = SVM, 2:KNN by default these detectors use HOG as the feature
%extractor as HOG proved to be the quickest and most accurate
%Scale: Scale decrease constant
[boundingBoxes, scale] = MultiScale(image,window_size,1, 1/1.2);

%Usage of KNNDetector paramaters:
%image = image location
%window size of sliding window as a 1x2 vector
%Feature Extractor: option: 1 = HOG, 2 = EDGE
%NON MAXIMUM SUPRESSION WORKS WITH KNN ONLY.
%boundingBoxes = KNNDetector(image,window_size,1);


%Usage of SVMDetector paramaters:
%image = image location
%window size of sliding window as a 1x2 vector
%Feature Extractor: option: 1 = HOG, 2 = EDGE, 3 = Raw pixels, 4 = gabor
%FYI: Non maximum supression does not work with SVM as we were not sure how
%to extract confidence from SVM. NON MAXIMUM SUPRESSION WORKS WITH KNN.
%boundingBoxes = SVMDetector(image,window_size,1);


%Usage of ShowDetectorResults
%image = image location
%boundingBoxes = array of boxes that are faces
% true/false = enable non maximum supression
%true/false = let NMS function know to expect confidence or not (KNN only)
%Set last two parameters to true for Non maximum supression when using
%KNN!
showDetectorResults(image, boundingBoxes, false, false); 
detection_plus_display_time_taken = toc

%Detector Evaluator
%Get true_positives,false_positives,true_negatives,false_negatives
%from given image, boundingBoxes, window_size
[true_positives,false_positives,true_negatives,false_negatives,accuracy] = Detector_Evaluator(image, boundingBoxes, window_size);
true_positives
false_positives
true_negatives
false_negatives
accuracy
%Haars face recognition
%ObjectDetection ('1.jpg','haarcascade_frontalface_alt.mat');
