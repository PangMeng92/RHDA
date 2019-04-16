clear all
clc
  

%load AR_SunGlasses_SSPPb.mat
load AR_Scarf_SSPPb.mat;


dim=size(Iv,1);
class=size(Iv,3);
elltrain=1; % training sample;
elltest=3; % test sampe;
TotalTrain = class * elltrain;
TotalTest = class * elltest;

% fea: Rows of vectors of data points. Each row is x_i
Itrain = zeros(dim,TotalTrain);
 for i=1:class
    for j=1:elltrain
         Itrain(:,j+(i-1)*elltrain) = Iv(:,j,i);
    end
 end
 trainlabels = constructlabel(class, elltrain);
 
 
Itest = zeros(dim,TotalTest);
for i=1:class
    for j=1:elltest
        Itest(:,j+(i-1)*elltest) = Iv(:,j+elltrain,i);
    end
end
testlabels = constructlabel(class,elltest);

%   Itrain = Itrain ./ repmat(sqrt(sum(Itrain .* Itrain )),[size(Itrain ,1) 1]); % unit norm 2
%   Itest = Itest ./ repmat(sqrt(sum(Itest .* Itest)),[size(Itest ,1) 1]); % unit norm 2

par.dim = dim;
par.tr_num = TotalTrain;  % The number of training samples
par.tt_num = TotalTest;   % The number of testing samples




%% Begin Robust Heterogeneous Discriminative Analysis (RHDA)
options = [];
options.gnd=trainlabels;
options.blocksize =8;
options.neighbor = 1;
options.t=1;


Yt=cell(1,par.tr_num);
options.elltest=elltest;
[ItestNew,testlabelsNew, IndexTest] = partition_block(Itest, options);


[WProj, ItrainNew] = DMME(Itrain, options);
[WProj2, ItrainNew2] = DSME(Itrain, options);   

%[WProj, WProj2, ItrainNew]=RHDA(Itrain, options);

%% Obtain low-dimensional representations of each manifold 
Y=cell(1,par.tr_num);

for i=1: par.tr_num
    WProj_Temp=cell2mat(WProj(i));
    ItrainNew_Temp=cell2mat(ItrainNew(i));
    Y_Temp=WProj_Temp'*ItrainNew_Temp;
    Y(i)={Y_Temp};
end

%% Obtain low-dimensional representations in a shared subspace
blocksize=options.blocksize;
blockdim=blocksize^2;
BlockSmp= dim/blockdim;
YS=cell(1,BlockSmp);
YSt=cell(1,par.tt_num);

options.WProj_DSME=cell2mat(WProj2(1));

 for ii=1:par.tt_num
      WProj_Temp=cell2mat(WProj2(1)); 
      ItestNew_Temp=cell2mat(ItestNew(ii));
      Y_Temp2=WProj_Temp'*ItestNew_Temp;
      YSt(ii) = {Y_Temp2};
 end

%% Recognition Phase
Miss_NUM1=0;
Miss_NUM2=0;

k=2;
lambda=0.001;
for i=1: par.tt_num
    Itest_Temp=cell2mat(ItestNew(i));
    for j=1:par.tr_num
    WProj_Temp=cell2mat(WProj(j));
    Y_Temp=WProj_Temp'*Itest_Temp;
    Yt(j)={Y_Temp};
    end
    [label_Multi, Vote_Multi] = LRCDist4(Y,Yt,trainlabels,lambda,k);   % patch-to-manifold
    [label_Single, Vote_Single] = CRCDist_neighbor(Itrain,YSt(i),trainlabels,BlockSmp,lambda,IndexTest,options);  % patch-to-patch
    Vote_Sum = Vote_Single+Vote_Multi;   % joint majority voting
    [sorted,neighborhood] = sort(Vote_Sum,'descend' );
    label=neighborhood(1:5);
    
    if  ~ismember(testlabels(i),label(1))
         Miss_NUM1=Miss_NUM1+1;
    end
     if  ~ismember(testlabels(i),label)
          Miss_NUM2=Miss_NUM2+1;
     end    
    Yt(:)=[];  
end

Miss_NUM1
Miss_NUM2
Recognition_rate_top1 = (par.tt_num-Miss_NUM1)/par.tt_num 
Recognition_rate_top5 = (par.tt_num-Miss_NUM2)/par.tt_num 



