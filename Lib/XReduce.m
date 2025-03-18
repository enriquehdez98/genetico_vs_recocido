function [Xr] = XReduce(nums,X)
%% This function reduces the dimensions of the original database and returns
%Xr=Database reduced to length(nums) characteristics
[a,b]=size(X);
Xr=zeros(length(nums),b);
for f=1:length(nums)
    Xr(f,:)=X(nums(f),:);
end

end