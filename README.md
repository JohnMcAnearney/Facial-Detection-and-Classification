First; Add all folders to path by #Right click folder >> 'Add to Path'>> 'Selected Folders and subfolders'#
       Shift select all folders and follow above instructions to complete faster. 

To use, run main.m. 
For SVM and KNN detectors input args are (imagetobeTested, windowSize, featureExtractorChoice)
Further explanation of parameter uses are explained in main.m

Run SVM+HOG+Detector example:
uncomment lines in main.m: 
boundingBoxes = SVMDetector(image,window_size,2);
showDetectorResults(image, boundingBoxes, false, false); 
This will output all of the faces detected in the image.

There are things that can be changed via flags in the detectors
KNNTraining allows you to choose to have image pre-processing or not and choose which feature extractor to use. 
KNNTesting, you set the K value in the args. The K value is the number of neighbours youre going to check. 


// Run individual tests without detector implemented got to any file in 'Evaluators'. Syntax; Edge_SVM_CV run SVM classifier with
Edge Feature extrartor wit Cross validation applied. 
