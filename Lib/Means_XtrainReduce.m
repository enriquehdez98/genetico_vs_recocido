function [m1r,m2r,Xr] = Means_XtrainReduce(nums,m1,m2,X)
%% This function reduces the dimensions of the original database and returns
%m1r=Matrix with the mean of each of the characteristics study group
%m2r=Matrix with the mean of each of the control group characteristics
%Xr=Database reduced to length(nums) characteristics
m1r=zeros(1,length(nums));
m2r=zeros(1,length(nums));
[a,b]=size(X);
Xr=zeros(length(nums),a);
for f=1:length(nums)
    m1r(f)=m1(nums(f));
    m2r(f)=m2(nums(f));
    Xr(f,:)=X(:,nums(f));
end
m1r=m1r';
m2r=m2r';
end

