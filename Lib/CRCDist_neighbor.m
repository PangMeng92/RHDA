 function [label,Vote] = CRCDist_neighbor(X,Yt,trainlabels,BlockSmp,lambda,Index,options);
if ~exist('lambda','var')
    lambda = 0.01;
end

if ~isfield(options,'blocksize')
    block_size = 2; 
else
    block_size = options.blocksize; 
end

if ~isfield(options,'WProj_DSME')
    WProj_DSME = 1; 
else
    WProj_DSME = options.WProj_DSME; 
end

if ~isfield(options,'neighbor')
    neighbor = 1; 
else
    neighbor = options.neighbor; 
end

%%  Divide gallery image into several blocks, and find the neighoring patches of the position (pox_x, pos_y)
[nFea, nSmp] = size(X);
nSize = sqrt(nFea);
Image_row_NUM=nSize;
Image_column_NUM=nSize;
slide_len=block_size;
imgTest = cell2mat(Yt);
Labels = unique(trainlabels);
nClass = length(Labels);
Vote=zeros(nClass,1);
Distance=zeros(nClass,1);
dist = 10^20;



% imgGallery=[];
% X_new=cell(1,nSmp);
for ii =1: BlockSmp
    pos_x=Index(ii,1);
    pos_y=Index(ii,2);
    traingnd=[];
% square patches
pm=1;
po=block_size-pm;
rowtem1=pos_x-neighbor;
rowtem2=pos_x+neighbor;
coltem1=pos_y-neighbor;
coltem2=pos_y+neighbor;
tr_dat=[];

if rowtem1<1
    rowtem1=1;
end

if coltem1<1
    coltem1=1;
end

if rowtem2>Image_row_NUM-po
    rowtem2=Image_row_NUM-po;
end

if coltem2>Image_column_NUM-po
    coltem2=Image_column_NUM-po;
end

for i =1: nSmp
    img = reshape(X(:,i), nSize, nSize);
    img = img';
    [img_x,img_y]=size(img);
    s=0;
    patch=[];
    C4=[];
 for j=rowtem1:neighbor:rowtem2
         for k=coltem1:neighbor:coltem2
           s=s+1;
           C4= img((pm*(j-1)+1):(pm*j+po),(pm*(k-1)+1):(pm*k+po));  %patch for training set
           Temp=reshape(C4,block_size*block_size,1);
           patch(:,s)=WProj_DSME'*Temp;
         end  
 end
          tr_dat=[tr_dat,patch];
          traingnd=[traingnd;ones(s,1)*trainlabels(i)]; 
end
block = imgTest(:,ii); 
    [W_all, gamma_x, total_iter, total_time] = BPDN_homotopy_function(tr_dat, block, lambda, 50);
    %W_all = ((tr_dat'*tr_dat+lambda*eye(size(tr_dat,2)))\tr_dat'*block);
    for jj=1:nClass  
      idx=find(traingnd==trainlabels(jj));
      W=W_all(idx);
      tr_Temp=tr_dat(:,idx);
     blockhat = block*max(W);
    tmpdist = norm(block-blockhat,1);
    Distance(jj) = tmpdist;
    end
[sorted,neighborhood] = sort(Distance,'ascend' );
Vote(neighborhood(1:1))=Vote(neighborhood(1:1))+1;   
end
 [sorted,neighborhood] = sort(Vote,'descend' );
 label=neighborhood(1:5);

