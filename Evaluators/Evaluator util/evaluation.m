function evaluation(testLabels, classificationResult, testImages, timing)

comparison = (testLabels==classificationResult);

% Gather Type I and Type II errors
tp = 0;
tn = 0;
fp = 0;
fn = 0;
for i=1:size(testLabels,1)
   if(classificationResult(i,1) == 1)
       if(testLabels(i,1) == -1)
        fp = fp +1;
       end
       if(testLabels(i,1) == 1)
        tp = tp +1;
       end
   end
   if(classificationResult(i,1) == -1)
       if testLabels(i,1) == 1 
        fn = fn +1;
       end
       if(testLabels(i,1) == -1)
        tn = tn +1;
       end
   end
end

%CalculateTimings
preProcessingTime = timing(1,2);
trainTime = timing(1,2);
testTime = timing(1, 3);
totalTime = preProcessingTime + trainTime + testTime;

%Calculate rates - check again
Accuracy = sum(comparison)/length(comparison);
correct = sum(comparison);
incorrect = length(comparison) - correct;
numberOfImages = length(comparison);
totalNegative = numberOfImages - correct;
trueNegative = tn;
truePositive = tp;
errorRate = (fn+fp)/numberOfImages;
sensitivty = truePositive/ (truePositive+fn);
precision =  truePositive/ (truePositive+fp);
specificity = trueNegative / (trueNegative + fp);
FMeasure = (2*(precision*sensitivty))/ (precision + sensitivty);
FalseAlarm = 1-specificity;
[FPR, TPR, ~, AUC] = perfcurve(testLabels, classificationResult, 1);

% Table to display rate calculations
Type = {'Accuracy';'FN'; 'FP';'errorRate';'sensitivty';'precision';'specificity';'FMeasure';'FalseAlarm';'AUC';'PreProcessingTime';'TrainTime';'TestTime';'TotalTime'};
Result = [Accuracy;fn;fp;errorRate;sensitivty;precision;specificity;FMeasure;FalseAlarm;AUC;trainTime;testTime;preProcessingTime; totalTime];
T = table(Type, num2cell(Result));
uitable('Data', T{:,:}, 'ColumnName', T.Properties.VariableNames,'RowName', T.Properties.RowNames, 'Unit', 'Normalized');

% Graph for Type I and Type II errors
errorGraphData =[];
errorGraphData= [fn; fp];
figure('Name', 'Error Graph', 'Color', 'white')
X = categorical({'Type I','Type II'});
X = reordercats(X,{'Type I','Type II'});
b = bar(X, errorGraphData, 'b');
b.FaceColor = 'flat';
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

% Graph to display accuracy
barGraphData =[];
barGraphData= [correct; incorrect];

figure('Name', 'Accuracy', 'Color', 'white')
X = categorical({'Correct','Incorrect'});
X = reordercats(X,{'Correct','Incorrect'});
b = bar(X, barGraphData, 'b');
b.FaceColor = 'flat';
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

% Plot roc
plot(FPR, TPR);

% Confusion Matrix
figure, confusionchart(testLabels, classificationResult);
%PCA used to display test images;
[U,S,X_reduce] = pca(testImages,3);
imean=mean(testImages,1);
X_reduce=(testImages-ones(size(testImages,1),1)*imean)*U(:,1:3);
figure, hold on
title('Test  PCA');
colours= ['r.'; 'g.'; 'b.'; 'k.'; 'y.'; 'c.'; 'm.'; 'r+'; 'g+'; 'b+'; 'k+'; 'y+'; 'c+'; 'm+'];
count=0;
for i=min(classificationResult):max(classificationResult)
    count = count+1;
    indexes = find (classificationResult == i);
    plot3(X_reduce(indexes,1),X_reduce(indexes,2),X_reduce(indexes,3),colours(count,:))
end
end
