# Facial-Detection-and-Classification
Facial detection and classification program using MatLab. 

This was a collaboration project between myself and 2 others in my university group. In this we implemented feature extractors of Edge Extraction, HOG, GABOR and use of Raw Pixels. Alongside 3 classifiers KNN, NN and SVM. Our accuracies were reported in our writeup. My classifiers ranged in accuracy from around 70% to the highest being 94.1% accuracy in classifying if the test image was a face or not. 

I was responsible for coding the KNN and NN classifiers. The KNN classifier works using a euclidean distance calculation for each image in the training images. Basically checking each pixel's value to its closest pixel value and then asking the training data "is this a face?" if so, it learns that for the testing. 

When testing, the algorithm then decides if the test image is a face by itself and then it checks its closest neighbours if they are faces and they vote. It is the outcome of this vote that determines whether the test image is a face. 

Whilst programming I realised that with KNN you dont even need to train the data due to how the algorithm works and so performance literally doubled as i had 50% the could to execute. 

# How to use
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

