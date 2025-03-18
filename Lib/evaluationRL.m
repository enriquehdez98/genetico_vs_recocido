function [Sensitivity,Specificity] = evaluation(Y,Ybayes)
%This function calculates the specificity (SP) and sensitivity (SE)
% of a classifier from the labels of the original data set and its predictions.

a=sum(Ybayes==1 & Ybayes==Y);
b=sum(Ybayes==1 & Ybayes~=Y);
c=sum(Ybayes==0 & Ybayes~=Y);
d=sum(Ybayes==0 & Ybayes==Y);
Sensitivity=a/(a+c);
Specificity=d/(b+d);
end

