function [M] = FFDnet_BM3D_M(data,known,nway,N,X,Y,M0,opts)
global sigmas
alpha  = opts.alpha;
sigma1 = opts.sigma1;
sigma2 = opts.sigma2;
beta1  = opts.beta1; 
beta2  = opts.beta2; 
rho    = opts.rho;
tol    = opts.tol;

%% initialization
M = M0;
Mimg  = M0;
% Mimg  =  vdt2image256(M0); % 3 order
[w,h,c,d] = size(Mimg);

Eimg = Mimg; 
Pimg = Mimg; 
Fimg = zeros(size(Eimg));
Qimg = zeros(size(Pimg));

%% FFDnet parameter
useGPU      = 1;

%% FFDnet parameter
if c == 3
    load(fullfile('FFDNet_Clip_color.mat'));
else
    load(fullfile('FFDNet_Clip_gray.mat'));
end

net = vl_simplenn_tidy(net);
if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

for r = 1:10
    Mlast = M; 
      %% update E
      Mimg  = M;
%        Mimg  = vdt2image256(M);
       input0 = Mimg - Fimg/beta1;
       A = zeros(1,d);
       for i= 1:d
           Temp = input0(:,:,:,i);
           A(i) = max(Temp(:));
           input0(:,:,:,i) =  Temp /A(i);
       end
       
        [n1 n2 n3 n4] = size(input0);
        input1=zeros(n1*n4,n2,n3);      
        for ii=1:n1
            for jj=1:n4
                 input1((ii-1)*n4+jj,:,:) = input0(ii,:,:,jj);
            end
        end      
        
        
        
    if c==3
        input = input1;
    else
        input = unorigami(input0,[w h c]);
    end
     
    input = single(input); %
    if c==3
        if mod(w,2)==1
            input = cat(1,input, input(end,:,:)) ;
        end
        if mod(h,2)==1
            input = cat(2,input, input(:,end,:)) ;
        end
    else
        if mod(w,2)==1
            input = cat(1,input, input(end,:)) ;
        end
        if mod(h,2)==1
            input = cat(2,input, input(:,end)) ;
        end
    end
    
    if useGPU
        input = gpuArray(input);
    end
    max_in = max(input(:));min_in = min(input(:));
    input = (input-min_in)/(max_in-min_in);
    sigmas = sigma1/(max_in-min_in);
    
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
    output = res(end).x;
    
    output(output<0)=0;output(output>1)=1;
    output = output*(max_in-min_in)+min_in;
    
    if c==3
        if mod(w,2)==1
            output = output(1:end-1,:,:);
        end
        if mod(h,2)==1
            output = output(:,1:end-1,:);
        end
    else
        if mod(w,2)==1
            output = output(1:end-1,:);
        end
        if mod(h,2)==1
            output = output(:,1:end-1);
        end
    end
    
    if useGPU
        output = gather(output);
    end

    if c==3
        Eimg = double(output);   
    else
        Eimg = origami(double(output),[w h c]);
    end
      
        output1=zeros(n1,n2,n3,n4);      
        for ii=1:n1
            for jj=1:n4
                 output1(ii,:,:,jj) = Eimg((ii-1)*n4+jj,:,:);
            end
        end      
        
     for i= 1:d
        E(:,:,:,i) = A(i)*output1(:,:,:,i);
     end  
        Eimg = E;
      %% update P
      input2 = Mimg - Qimg/beta2;
       B = zeros(1,d);
       for i= 1:d
           Temp2 = input2(:,:,:,i);
           B(i) = max(Temp2(:));
           input2(:,:,:,i) =  Temp2/B(i);
       end
     
        input23=zeros(n1*n4,n2,n3);      
        for ii=1:n1
            for jj=1:n4
                 input23((ii-1)*n4+jj,:,:) = input2(ii,:,:,jj);
            end
        end      
        input2 = input23;
       
       
      max_in2 = max(input2(:));min_in2 = min(input2(:));
      input2 = (input2-min_in2)/(max_in2-min_in2);
      sigmas2 = sigma2/(max_in2-min_in2);
       
       [~, Pimg] = CBM3D(1, input2, sigmas2);  
       
       Pimg(Pimg<0)=0;Pimg(Pimg>1)=1;
       Pimg = Pimg*(max_in2-min_in2)+min_in2;  
       
        output2=zeros(n1,n2,n3,n4);      
        for ii=1:n1
            for jj=1:n4
                 output2(ii,:,:,jj) = Pimg((ii-1)*n4+jj,:,:);
            end
        end    
        
       for i= 1:d
            P(:,:,:,i) =   B(i)*output2(:,:,:,i);
       end
            Pimg = P;
   %% update M
     temp1 = Eimg+Fimg/beta1;
     temp2 = Pimg+Qimg/beta2;
     temp = beta1*temp1 + beta2*temp2+ rho*M0;
     for n = 1:N-1
        temp = temp+alpha(n)*reshape(X{n}*Y{n},nway);
     end
     M = temp/(1+rho+beta1+beta2);
     M(known) = data;
    M = max(0,min(M,255));
    %% update Lambda
    
     Fimg = Fimg + beta1*(Eimg-Mimg);      
     Qimg = Qimg + beta2*(Pimg-Mimg);    
    %% stopping criterion
     relerr(r) = abs(norm(M(:)-Mlast(:)) / norm(Mlast(:)));
     if relerr(r) < tol
        break
     end
end
end