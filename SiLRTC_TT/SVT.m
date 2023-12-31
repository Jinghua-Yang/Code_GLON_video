function [ X ] = SVT( A, tau)

    [U0,Sigma0,V0] = svd( full(A), 'econ' );  %%%%%%%%mex file SVD
    Sigma0 = diag(Sigma0);
    S      = soft(Sigma0, tau);
    r      = sum( S>0 );
    U      = U0(:,1:r);
    V      = V0(:,1:r);
    X      = U*diag(S(1:r))*V';
end