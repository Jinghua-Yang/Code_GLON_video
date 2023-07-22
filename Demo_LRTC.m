clc; clear all; close all;
addpath(genpath(cd));
rand('seed',213412); 

EN_HaLRTC   = 1;
EN_SiLRTCTT = 1;
EN_tSVD     = 1;
EN_TMacTT   = 1;
EN_KBR      = 1;
EN_TT_TV    = 1;
EN_GLON      = 1;  % Ours

methodname  = {'HaLRTC','SiLTRC-TT','tSVD','TMac-TT','KBR','TT-TV','GLON'};

load 'carphone_qcif.mat'
X0   = double(X(:,:,:,1:300));
X0   = min( 255, max( X0, 0 ));
nway = size(X0);
[n1 n2 n3 n4] = size(X0);

name = {'carphone'};

for SR = [0.1 0.2 0.05 ]
%% Generate known data
% Nway = size( image2vdt256(X0) );
P = round(SR*prod(nway));      % prod·µ»Ø³Ë»ý
Known = randsample(prod(nway),P);
[Known,~] = sort(Known);

%% Ket Augmentation
% Xtrue = image2vdt256(X0);
%% Missing data
Xkn          = X0(Known);
Xmiss        = zeros(nway);
Xmiss(Known) = Xkn;
    
imname=[num2str(name{1}),'_tensor0','_SR_',num2str(SR),'.mat'];
save(imname,'X0','Xmiss','Xkn','Known');

%% HaLRTC
j = 1;
if EN_HaLRTC
    %%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);
    
    opts = [];
    alpha      = ones(ndims(X0),1);
    opts.alpha = alpha/sum(alpha); 
    opts.tol   = 1e-4; 
    opts.maxit = 200; 
    opts.rho   = 1.1; 
    opts.max_beta = 1e10;
    opts.X0       = X0;
    
    beta = [5*1e-5, 1e-4, 5*1e-4, 1e-3, 5*1e-3];
    for n = 1:length(beta)
        opts.beta = beta(n);
        t0=tic;
        X = HaLRTC( Xkn, Known, opts );
        X    = min( 255, max( X, 0 ));
        time=toc(t0);
                         psnr_vector = zeros(1,n4);
                         ssim_vector = zeros(1,n4);
     
                         for ii=1:n4
                             A = double(X0(:,:,:,ii)); B =double( X(:,:,:,ii));
                            
                             psnrVector = zeros(1,n3);
                             for jj = 1:n3
                                  psnrVector(jj) = psnr3(A(:,:,jj)/255,B(:,:,jj)/255);
                             end
                              psnr_vector(ii) = mean(psnrVector);
            
                              ssimVector = zeros(1,n3);
                              for jj = 1:n3
                                   ssimVector(jj) = ssim3(A(:,:,jj),B(:,:,jj));
                              end
                               ssim_vector(ii) = mean(ssimVector);
                         end
        
                          psnr = mean(psnr_vector);
                          ssim = mean(ssim_vector);
        
        display(sprintf('psnr=%.2f,ssim=%.4f,beta=%.5f', psnr, ssim, opts.beta))
        fprintf('==================================\n')
        
        imname=[num2str(name{1}),'_SR_',num2str(SR),'_result_',num2str(methodname{j}),'_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_beta_',num2str(opts.beta),'.mat'];
        save(imname,'X','psnr_vector','ssim_vector','time');
    end
end

%% SiLRTC-TT
j = j+1;
if EN_SiLRTCTT 
    %%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);

    opts=[]; 
    opts.tol   = 1e-4; 
    opts.maxit = 200; 
    opts.X0    = X0;
    
    %%%%
      r = [0.01];
    for n = 1:length(r)     
        opts.r = r(n);
        t0=tic;
        [X, Out] = SiLRTC_TT( Xkn, Known, opts );
        
%         X = vdt2image256( X );
        X = min( 255, max( X, 0 ));  
        time=toc(t0);
                         psnr_vector = zeros(1,n4);
                         ssim_vector = zeros(1,n4);
     
                         for ii=1:n4
                             A = double(X0(:,:,:,ii)); B =double( X(:,:,:,ii));
                            
                             psnrVector = zeros(1,n3);
                             for jj = 1:n3
                                  psnrVector(jj) = psnr3(A(:,:,jj)/255,B(:,:,jj)/255);
                             end
                              psnr_vector(ii) = mean(psnrVector);
            
                              ssimVector = zeros(1,n3);
                              for jj = 1:n3
                                   ssimVector(jj) = ssim3(A(:,:,jj),B(:,:,jj));
                              end
                               ssim_vector(ii) = mean(ssimVector);
                         end
        
                          psnr = mean(psnr_vector);
                          ssim = mean(ssim_vector);
        

        display(sprintf('psnr=%.2f,ssim=%.4f,r=%.2f',psnr, ssim, r(n)))
        display(sprintf('=================================='))
        
        imname=[num2str(name{1}),'_SR_',num2str(SR),'_result_',num2str(methodname{j}),'_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_r_',num2str(r(n)),'.mat'];
        save(imname,'X','psnr_vector','ssim_vector','time');
    end
end

%% tSVD
j = j+1;
if EN_tSVD
     %%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);
       
    Mask  = zeros( nway );
    Mask( Known ) = 1;
%     Mask = vdt2image256(Mask);
    
    alpha = 1; 
    maxItr  = 200; 
    myNorm = 'tSVD_1'; 
    
    X   = zeros( nway );
    Mask0  = zeros( nway );
    Mask0( Known ) = 1;
    
%     beta1 = [1e-6, 5*1e-6, 1e-5, 5*1e-5];
     beta1 = [5*1e-6];
    for k = 1:length(beta1)
        beta = beta1(k);
         t0=tic;
         for p = 1:nway(3)       
            Mask = Mask0(:,:,p,:);
            XX0  = X0(:,:,p,:);
            A    = diag(sparse(double(Mask(:)))); 
            b    = A * XX0(:);
            XX   = tensor_cpl_admm( A, b, beta, alpha, size( XX0 ), maxItr, myNorm, 0 );
            XX   = reshape( XX, size( XX0 ) );
            X(:,:,p,:) = XX;         
         end
        time=toc(t0);
        X = min( 255, max( X, 0 ));
        
                         psnr_vector = zeros(1,n4);
                         ssim_vector = zeros(1,n4);
     
                         for ii=1:n4
                             A = double(X0(:,:,:,ii)); B =double( X(:,:,:,ii));
                            
                             psnrVector = zeros(1,n3);
                             for jj = 1:n3
                                  psnrVector(jj) = psnr3(A(:,:,jj)/255,B(:,:,jj)/255);
                             end
                              psnr_vector(ii) = mean(psnrVector);
            
                              ssimVector = zeros(1,n3);
                              for jj = 1:n3
                                   ssimVector(jj) = ssim3(A(:,:,jj),B(:,:,jj));
                              end
                               ssim_vector(ii) = mean(ssimVector);
                         end
        
                          psnr = mean(psnr_vector);
                          ssim = mean(ssim_vector);
        
        display(sprintf('psnr=%.2f,ssim=%.4f,beta=%.6f', psnr, ssim, beta))
        display(sprintf('=================================='))
        
        imname=[num2str(name{1}),'_SR_',num2str(SR),'_result_',num2str(methodname{j}),'_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_beta_',num2str(beta,'%.6f'),'.mat'];
        save(imname,'X','psnr_vector','ssim_vector','time');
    end
end

%% TMac-TT
j = j+1;
if EN_TMacTT
     %%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);    
    opts = [ ];
    opts.alpha = weightTC(nway);
    opts.tol   = 1e-4;
    opts.maxit = 200;
    opts.X0 = X0;
%     opts.Xtrue = Xtrue;
    th = [0.01];   %0.01 0.02
    for k = 1:length(th)
        opts.th = th(k);
        t0=tic;
        [X, Out] = TMac_TT( Xkn, Known, opts );
%         X = vdt2image256(X);
        X = min( 255, max( X, 0 ));
        time=toc(t0);
        
                         psnr_vector = zeros(1,n4);
                         ssim_vector = zeros(1,n4);
     
                         for ii=1:n4
                             A = double(X0(:,:,:,ii)); B =double( X(:,:,:,ii));
                            
                             psnrVector = zeros(1,n3);
                             for jj = 1:n3
                                  psnrVector(jj) = psnr3(A(:,:,jj)/255,B(:,:,jj)/255);
                             end
                              psnr_vector(ii) = mean(psnrVector);
            
                              ssimVector = zeros(1,n3);
                              for jj = 1:n3
                                   ssimVector(jj) = ssim3(A(:,:,jj),B(:,:,jj));
                              end
                               ssim_vector(ii) = mean(ssimVector);
                         end
        
                          psnr = mean(psnr_vector);
                          ssim = mean(ssim_vector);
                          
          display(sprintf('psnr=%.2f,ssim=%.4f,th=%.2f',psnr, ssim, opts.th))
          display(sprintf('=================================='))
                
          imname=[num2str(name{1}),'_SR_',num2str(SR),'_result_',num2str(methodname{j}),'_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_th_',num2str(opts.th),'.mat'];
          save(imname,'X','psnr_vector','ssim_vector','time');
    end
end
%% KBR
j = j+1;
if EN_KBR
    %%%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);
    
    Omega     = zeros( nway );
    Omega(Known) = 1;
    Omega     = (Omega > 0);
    
    opts = [];   
    opts.tol    = 1e-4;
    opts.maxit  = 1000;
    opts.rho    = 1.1;
    opts.lambda = 0.01; % 0.01 is even better
    opts.mu     = 5*1e-5;  
%       for lambda = [0.001 0.005 0.01 0.05]        
%       for mu = [1e-13, 1e-9, 1e-5, 5*1e-5]
  for lambda = [0.001]        
      for mu = [1e-13]
          
          opts.lambda = lambda;
           opts.mu = mu; 
    t0=tic;
    X = KBR_TC( X0.*Omega, Omega, opts );
    time=toc(t0);      
    X    = min( 255, max( X, 0 ));
        
                         psnr_vector = zeros(1,n4);
                         ssim_vector = zeros(1,n4);
     
                         for ii=1:n4
                             A = double(X0(:,:,:,ii)); B =double( X(:,:,:,ii));
                            
                             psnrVector = zeros(1,n3);
                             for jj = 1:n3
                                  psnrVector(jj) = psnr3(A(:,:,jj)/255,B(:,:,jj)/255);
                             end
                              psnr_vector(ii) = mean(psnrVector);
            
                              ssimVector = zeros(1,n3);
                              for jj = 1:n3
                                   ssimVector(jj) = ssim3(A(:,:,jj),B(:,:,jj));
                              end
                               ssim_vector(ii) = mean(ssimVector);
                         end
        
                          psnr = mean(psnr_vector);
                          ssim = mean(ssim_vector);
    display(sprintf('psnr=%.2f,ssim=%.4f,lambda=%.2f,mu=%.5f', psnr, ssim, opts.lambda, opts.mu))
    display(sprintf('=================================='))
        
    imname = [num2str(name{1}),'_SR_',num2str(SR),'_result_',num2str(methodname{j}),'_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_lambda_',num2str(opts.lambda),'_mu_',num2str(opts.mu),'.mat'];
    save(imname,'X','psnr_vector','ssim_vector','time');  
      end
      end
end
%% TT-TV
j = j+1;
if EN_TT_TV 
    %%%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);
    
    opts=[];
    opts.alpha  = weightTC(nway);
    opts.tol    = 1e-4;
    opts.maxit  = 200;
    opts.X0  = X0;
%     opts.th     = 0.01;
    opts.rho    = 10^(-3);
%     opts.lambda = 0.1; 
      opts.beta3 = 0.3; opts.beta1 = 5*10^(-3); opts.beta2  = 0.1;  
 
    for th = [0.01]
        for lambda = [0.1 0.3]
          opts.th = th;
          opts.lambda = lambda;
    t0=tic;
    [X, Out_TT_TV] = TT_TV( Xkn, Known, nway, opts );
     time=toc(t0);  
%     X = vdt2image256(X);
    X = min( 255, max( X, 0 ));
                        psnr_vector = zeros(1,n4);
                         ssim_vector = zeros(1,n4);
     
                         for ii=1:n4
%                              A = X0(:,:,:,ii); B = X(:,:,:,ii);
                            A = double(X0(:,:,:,ii)); B =double( X(:,:,:,ii))
                             psnrVector = zeros(1,3);
                             for jj = 1:3
                                  psnrVector(jj) = psnr3(A(:,:,jj)/255,B(:,:,jj)/255);
                             end
                              psnr_vector(ii) = mean(psnrVector);
            
                              ssimVector = zeros(1,3);
                              for jj = 1:3
                                   ssimVector(jj) = ssim3(A(:,:,jj),B(:,:,jj));
                              end
                               ssim_vector(ii) = mean(ssimVector);
                         end
        
                          psnr = mean(psnr_vector);
                          ssim = mean(ssim_vector);

        
                         display(sprintf('psnr=%.2f,ssim=%.4f,th=%.2f',psnr, ssim, opts.th))
                         display(sprintf('=================================='))
                            
                        imname=[num2str(name{1}),'_SR_',num2str(SR),'_result_',num2str(methodname{j}),'_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_th_',num2str(opts.th),'_lambda_',num2str(opts.lambda),'_beta3_',num2str(opts.beta3),'_beta1_',num2str(opts.beta1),'_beta2_',num2str(opts.beta2),'.mat'];
                        save(imname,'X','Out_TT_TV','psnr_vector','ssim_vector','time');
     end
    end
end
%% GLON
j = j+1;
if EN_GLON 
    %%%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);
    
    opts = [];
    opts.tol    = 2*1e-3;
    opts.maxit  = 100;
    opts.X0     = X0;
    opts.rho    = 10;
    
    for th = [0.01]
        for sigma1 = 0.05%[0.05 0.2 0.5 1 2]
            for sigma2 = 10%[1 10 40] 
                for beta1 = 30%[50 30 10 5 1]
                    for beta2 =30% [50 30 10 5 1]
                        opts.th     = th;
                        opts.sigma1  = sigma1;
                        opts.sigma2  = sigma2;
                        opts.beta1   = beta1;
                        opts.beta2   = beta2;      
                         t0=tic;   
                        [X, Out_TT_FFDnet] = TT_FFDnet_BM3D( Xkn, Known, nway, opts);
                       
                        X = min( 255, max( X, 0 ));
                        
                         time=toc(t0);  
                         psnr_vector = zeros(1,n4);
                         ssim_vector = zeros(1,n4);
     
                         for ii=1:n4
                             A = double(X0(:,:,:,ii)); B =double( X(:,:,:,ii))
                             psnrVector = zeros(1,3);
                             for jj = 1:3
                                  psnrVector(jj) = psnr3(A(:,:,jj)/255,B(:,:,jj)/255);
                             end
                              psnr_vector(ii) = mean(psnrVector);
            
                              ssimVector = zeros(1,3);
                              for jj = 1:3
                                   ssimVector(jj) = ssim3(A(:,:,jj),B(:,:,jj));
                              end
                               ssim_vector(ii) = mean(ssimVector);
                         end
        
                          psnr = mean(psnr_vector);
                          ssim = mean(ssim_vector);

        
                        display(sprintf('psnr=%.2f,ssim=%.4f,th=%.2f,sigma1=%.6f,sigma2=%.6f,beta1=%.8f,beta2=%.8f',psnr, ssim, opts.th, opts.sigma1, opts.sigma2, opts.beta1, opts.beta2))
                        display(sprintf('=================================='))
                            
                        imname=[num2str(name{1}),'_SR_',num2str(SR),'_result_',num2str(methodname{j}),'_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_th_',num2str(opts.th),'_sigma1_',num2str(opts.sigma1),'_sigma2_',num2str(opts.sigma2),'_beta1_',num2str(opts.beta1),'_beta2_',num2str(opts.beta2),'.mat'];
                        save(imname,'X','Out_TT_FFDnet','psnr_vector','ssim_vector','time');
                    end
                end
            end
        end
    end
end    
end