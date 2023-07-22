function [M, Out_Si_TT] = SiLRTC_TT( data, known, opts )

    maxit = opts.maxit; 
    tol   = opts.tol;
    X0    = opts.X0;
%     Nway  = size( image2vdt256(X0) );
    Nway  = size(X0);
    r     = opts.r;
    
    % Initialization
    M     = initialization_M(Nway,known,data);
    N     = length(Nway);
    
    alpha = weightTC(Nway); 
    beta  = r*alpha;
    
    Out_Si_TT = [];
    
    dimL = zeros(1,N-1);
    dimR = zeros(1,N-1);
    IL = 1;
    for m = 1:N-1
        dimL(m) = IL*Nway(m);
        dimR(m) = prod(Nway)/dimL(m);
        IL = dimL(m);
    end 
    
    X = cell(1,N-1);
    k = 1;
    relerr = [];
    relerr(1) = 1;
 
    % Start Time measure
    t0=tic;
    while relerr(k) > tol
        k = k+1;
        Mlast = M;

       %% update X
        for n = 1:N-1
            X{n}  = SVT( reshape( M, [dimL(n) dimR(n)] ), alpha(n)/beta(n));
            X{n}  = reshape( X{n}, Nway);
        end
        
        %% update M
        M = 0;
        for n = 1:N-1
            M  = M + beta(n)*X{n};
        end
        M = M./(sum(beta));
        M(known) = data;
        
        %% Calculate relative error
        relerr(k) = abs(norm(M(:)-Mlast(:)) / norm(Mlast(:)));
     
        %% check stopping criterion
        if k > maxit || (relerr(k)-relerr(k-1) > 0)
            break
        end
    end
    % Stop Time measure
    time = toc(t0);
    Out_Si_TT.time = time;
    Out_Si_TT.relerr = relerr;
end