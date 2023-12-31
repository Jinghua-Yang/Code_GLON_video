function [M, Out_Si_TT] = TT_FFDnet_BM3D( data, known, nway, opts)
maxit = opts.maxit; 
tol   = opts.tol;
rho   = opts.rho;
alpha = weightTC(nway);

opts.alpha = alpha;
%% Initialization
N = length(nway);
M = initialization_M(nway,known,data);

ranktube = [5 5 5];
%     [X,Y] = initialMatrix(N,nway,ranktube);
%     save X.mat X
%     save Y.mat Y

load X.mat
load Y.mat



dimL = zeros(1,N-1);
dimR = zeros(1,N-1);
IL = 1;
for k = 1:N-1
    dimL(k) = IL*nway(k);
    dimR(k) = prod(nway)/dimL(k);
    IL = dimL(k);
end

X0 = X; Y0 = Y;  M0 = M;
X = cell(1,N-1); Y = cell(1,N-1);

N = length(nway);
k = 1;
relerr = [];
relerr(1) = 1;

max_rank = [50 50 50];
rank_inc = 5;

for i=1:3
    MM{i}=reshape(X0{i}*Y0{i},nway); 
    MMi = MM{i};
    resrank{i}=norm(M0(:)- MMi(:));
end
%% Start Time measure
t0=tic;
while relerr(k) > tol
    k = k+1;
    Mlast = M;
    
    %% update (X,Y)
    for n = 1:N-1
        M_Temp = reshape(M0,[dimL(n) dimR(n)]);
        X{n}   = (alpha(n)*M_Temp*Y0{n}' + rho*X0{n})*pinv( alpha(n)*Y0{n}*Y0{n}' + rho*eye(size(Y0{n}*Y0{n}')));
        Y{n}   = pinv(alpha(n)*X{n}'*X{n} + rho*eye(size(X{n}'*X{n})))*(alpha(n)*X{n}'*M_Temp +rho*Y0{n});
    end
    
    %% update M by ADMM
    [M] = FFDnet_BM3D_M(data,known,nway,N,X,Y,M0,opts);
    
    % Calculate relative error
    relerr(k) = abs(norm(M(:)-Mlast(:)) / norm(Mlast(:)));

    X0=X; Y0=Y; M0=M;
    %% check stopping criterion
    if k > maxit ||  relerr(k) < tol  
        break 
    end
    %% update Rank  
    for i=1:3
       resold{i}=resrank{i};
       MM{i}=reshape(X{i}*Y{i},nway); 
       MMi = MM{i};
       MMi(known)=M0(known);
       resrank{i}=norm(M0(:)-MMi(:));
       ifrank{i} = abs(1-resrank{i}/resold{i});
       nowrank=[size(X{1},1),size(X{2},2),size(X{3},2)];
       if ifrank{i}<0.01 && nowrank(i)<max_rank(i)
          [X{i},Y{i}]=rank_inc_adaptive(X{i},Y{i},rank_inc);
       end
    end
end
%% Stop Time measure
time = toc(t0);
Out_Si_TT.time = time;
Out_Si_TT.relerr = relerr;
end

function [A,X]=rank_inc_adaptive(A,X,rank_inc)
    % increase the estimated rank
    for ii = 1:rank_inc
        rdnx = rand(size(X,1),1);
        rdna = rand(1,size(A,2));
        X = [X,rdnx];
        A = [A;rdna];
    end
end
   
function [X0,Y0] = initialMatrix(N,Nway,ranktube)
X0 = cell(1,N-1);Y0 = cell(1,N-1);
dimL = zeros(1,N-1);
dimR = zeros(1,N-1);
IL = 1;
for k = 1:N-1
    dimL(k) = IL*Nway(k);
    dimR(k) = prod(Nway)/dimL(k);
    %
    X0{k} = randn(dimL(k),ranktube(k));
    Y0{k} = randn(ranktube(k),dimR(k));
    %uniform distribution on the unit sphere
    X0{k} = bsxfun(@rdivide,X0{k},sqrt(sum(X0{k}.^2,1)));
    Y0{k} = bsxfun(@rdivide,Y0{k},sqrt(sum(Y0{k}.^2,2)));
    %
    IL = dimL(k);
end
end