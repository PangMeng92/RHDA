function [eigvector, X_new] = DSME(X, options)
% DSME: Discriminative Single-Manifold Embedding 
%
%       [eigvector, label] = DSME(X, options)
% 
%             Input:
%               X       - Data matrix. Each row vector of fea is a data point.
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%                            ReducedDim   -  The dimensionality of the
%                                            reduced subspace. If 0,
%                                            all the dimensions will be
%                                            kept. Default is 0. 
%                         gnd     -     Label information  of given data 
%                         blocksize  -  Size of each partitioned image block
%                         neighbor  -   Neighboring patches
%
%             Output:
%               eigvector - Set of projection basis
%               label  - Distinguish different projection basis
% 
% 

[nFea, nSmp] = size(X);

%innerK=3;
%interK=100;

%eigvector =[];
eigvector =cell(1,nSmp);
label=[];

if (~exist('options','var'))
   options = [];
else
   if ~strcmpi(class(options),'struct') 
       error('parameter error!');
   end
end

if ~isfield(options,'blocksize')
    block_size = 2; 
else
    block_size = options.blocksize; 
end

if ~isfield(options,'neighbor')
    neighbor = 0; 
else
    neighbor = options.neighbor; 
end

%%  Divide image into several blocks
nSize = sqrt(nFea);

slide_len=block_size;
imgNew=[];
imgNew2=[];
X_new=cell(1,nSmp);
trainlabels_cell=cell(1,nSmp);
for i =1: nSmp
    img = reshape(X(:,i), nSize, nSize);
    img = img';
    [img_x,img_y]=size(img);
for ix=block_size/2:slide_len:img_x-block_size/2
    for jy=block_size/2:slide_len:img_y-block_size/2
        current_block=img((ix-block_size/2+1):(ix+block_size/2),(jy-block_size/2+1):(jy+block_size/2));
        dct_coeff=reshape(current_block,block_size^2,1);
        imgNew=[imgNew, dct_coeff];
        imgNew2=[imgNew2, dct_coeff];
    end
end
        X_new(i)={imgNew2};
        trainlabels_cell(i)={i};
        imgNew2=[];
end

X_old = X;
X = imgNew;

% Construct new category label
trainlabels=options.gnd;
classnum=length(unique(trainlabels));
NumPerClass=length(trainlabels)/classnum;
elltrain=(nSize/block_size)^2;
trainlabelsNew=constructlabel(classnum,elltrain*NumPerClass);


%% Calculate within-manifold  sparse reconstruction weights  Ww, and and between-manifold adjacency weights Wb (DSMMCL)
% Sparse reconstruction weights
X=NormalizeFea(X);
[Fea,Smp] = size(X);
options.gnd=trainlabelsNew;
Ww=constructDSNPE(X,trainlabelsNew);

Label = unique(trainlabelsNew);
nLabel = length(Label);

%% Calculate between-manifold adjacency weights 
interK=nSmp; % 500
X=NormalizeFea(X);
[Fea,Smp] = size(X);
Label = unique(trainlabelsNew);
nLabel = length(Label);

X2=NormalizeFea(X);
D = zeros(Smp);
        for i=1:Smp-1
            for j=i+1:Smp
                D(i,j) = norm(X2(:,i) - X2(:,j));
            end
        end
D = D+D';
D = exp(-D.^2/options.t);

Wb = ones(Smp,Smp);
Wa=zeros(Smp,Smp);
G=zeros(Smp,Smp);

for idx=1:nLabel
    classIdx = find(trainlabelsNew==Label(idx));
    Wb(classIdx,classIdx) = 0;
    Wa(classIdx,classIdx)=1;
end


 
if interK > 0
    Dw = EuDist2(X',[],0);
    Dw = Dw+Wa*(max(max(Dw))+10);
    [dump idx] = sort(Dw,2); % sort each row
    for i=1:Smp
        G(i,idx(i,1:interK)) = 1;
    end
    %G = max(G,G');    
    Wb = Wb.*G;
end

%% Calculate  projection basis 
Dw = diag(sum(Ww,2));
Db = diag(sum(Wb,2));

Mw = (Dw-Ww)'*(Dw-Ww);
Mb = (Db-Wb);

WPrime = X* Mw * X';
DPrime = X* Mb * X';


DPrime = max(DPrime,DPrime');
WPrime = max(WPrime,WPrime');

for ii =1:nSmp
    
[W, eigvalue] = eig(WPrime,DPrime);
%[W, eigvalue] = eig(WPrime);
eigvalue = diag(eigvalue);
   
[junk, index] = sort(eigvalue);

eigvalue = eigvalue(index);
W = W(:,index);
  index_N=find(eigvalue>0);
  label_num=length(index_N);
% index_N2=index_N(1:label_num);  
  index_N2=index(1:60);
  W = W(:,index_N2);
eigvector(ii)={W};
end



