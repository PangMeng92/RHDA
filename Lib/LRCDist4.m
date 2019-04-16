function [label,Vote] = LRCDist4(Y,Yt,trainlabels,lambda,k)
if ~exist('lambda','var')
    lambda = 0.01;
end
Labels = unique(trainlabels);
nClass = length(Labels);
%Distance=zeros(nClass,1);
Distance= cell(1,nClass);
testimg=cell2mat(Yt(1));
[dim,smp]=size(testimg);
DistanceV=zeros(smp,1);
Distance_Temp=zeros(nClass,1);
Vote=zeros(nClass,1);
for i = 1:nClass
    testimg=cell2mat(Yt(i));
    trainimg=cell2mat(Y(i));
    for j=1:smp
        testblock=testimg(:,j);
         distance_V=zeros(smp,1);
        for t=1:smp
            V=testblock-trainimg(:,t);
             distm=norm(V,2);
             distance_V(t)=distm;
        end
         [sorted,neighborhood] = sort(distance_V,'ascend' );
             Xi= trainimg(:,neighborhood(1:k));
                 imghat = Xi*((Xi'*Xi+lambda*eye(size(Xi,2)))\Xi'*testblock);
             tmpdist = norm(testblock-imghat,2);
             DistanceV(j)=tmpdist;
    end
    Distance(i)={DistanceV};
    
%   tmpdist = norm(testimg-trainimg,2);
   
end

for ii=1:smp
    for jj=1:nClass
        Temp=cell2mat(Distance(jj));
        Distance_Temp(jj)=Temp(ii);
    end
    [sorted,neighborhood] = sort(Distance_Temp,'ascend' );
   Vote(neighborhood(1:1))=Vote(neighborhood(1:1))+1;   
end
[sorted,neighborhood] = sort(Vote,'descend' );
 label=neighborhood(1:5);