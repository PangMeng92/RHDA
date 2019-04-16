function [X_new, labelsNew,index] = partition_block(X, options)

[nFea, nSmp] = size(X);
if ~isfield(options,'blocksize')
    block_size = 2; 
else
    block_size = options.blocksize; 
end

%%  Divide image into several blocks
nSize = sqrt(nFea);

slide_len=block_size;
imgNew2=[];
imgNew=[];
X_new=cell(1,nSmp);
X_new2=[];
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
X_new(i)={imgNew};
% X_new2=[X_new2,imgNew]
imgNew=[];
end

patch_row=img_x/slide_len;
patch_column=img_y/slide_len;
for ii=1: patch_row
    for jj=1: patch_column
        s=patch_row*(ii-1)+jj;
        index(s,1:2)=[slide_len*(ii-1)+1,slide_len*(jj-1)+1];
    end
end
        

X_old = X;
X = imgNew2;

%% Plot subfigures of divided blocks
% num= nSize/block_size;
% dim=block_size^2;
% for i=1:num*num
% %subplot(num,num,i);
% %A=X(:,i)
% A=X(:,i+2*num*num);
% A=reshape(A,sqrt(dim),sqrt(dim));
% %A=A'; 
% A=mat2gray(A);
% figure; imshow(A); 
% end
% figure;
% Construct new category label
labels=options.gnd;
classnum=length(unique(labels));
ell=(nSize/block_size)^2*options.elltest;
labelsNew=constructlabel(classnum,ell);
