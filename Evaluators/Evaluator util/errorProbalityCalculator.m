function [FPR, TPR] = errorProbalityCalculator(classificationResult,labels)
tp = 0;
tn = 0;
fp = 0;
fn = 0;
for i=1:size(labels,1)
   if(classificationResult(i,1) == 1)
       if(labels(i,1) == -1)
        fp = fp +1;
       end
       if(labels(i,1) == 1)
        tp = tp +1;
       end
   end
   if(classificationResult(i,1) == -1)
       if labels(i,1) == 1 
        fn = fn +1;
       end
       if(labels(i,1) == -1)
        tn = tn +1;
       end
   end
end
TPR = tp/(tp+fp);
FPR = tn/(tn+fp);
end

