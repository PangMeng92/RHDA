%%Construct W based on Sparse Representation(SR)
%%W: output weight matrix
%%X: sample sets, row: feature, col: samples

function [W] = constructDSNPE(X,trLab)% X:training sample; trLab: the label of training sample

[D,N] = size(X);

%%---normalization---%%

%%---centralization---%%
aveX = X - X*ones(N)/N;


%%---solve for sparse reconstruction weights---%%
 path(path, './Optimization');
W = zeros(N,N);
for i = 1:N
    y = [aveX(:,i);1];%% for invariance to translations
    DifferentIndex = find(trLab ~= trLab(i));
    DifferentIndex = [DifferentIndex,i];
    temp = 1:length(trLab);
    temp(DifferentIndex)=[];
    SameIndex = temp;
    A = [[aveX(:,SameIndex'); ones(1,length(SameIndex))] [eye(D); zeros(1,D)]];%% for invariance to translations 生成论文中的公式(16)中的矩阵

    %w0 = rand(D+N-1,1)+1;
    w0 = A\y;
    %wi = l1eq_pd(w0, A, [], y, 1e-1);%%%%%%%%%---!!!!!!!!!!---%%%%%%%%%
     wi = BPDN_homotopy_function(A, y, 0.01, 100);
    W(SameIndex',i) = wi(1:length(SameIndex));

    fprintf('No.%d sample is done\n',i)
end

