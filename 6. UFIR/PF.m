classdef PF < handle
   properties
       % Particle
       particle;
       % Number of particle
       N;
       
       % Weight
       W;
       % Noise
       Q,R;
       % Noise sigma(표준편차)
       sigma_u,sigma_v;
       
       % Resampling strategy
       resampling_strategy;
   end
   
   methods
       function model = PF(particle,N,W,Q,R,sigma_u,sigma_v,resampling_strategy) 
            model.particle = particle;
            model.N = N;
            model.W = W;
            
            %model.P = P;
            model.Q = Q;
            model.R = R;
            
            model.sigma_u = sigma_u;
            model.sigma_v = sigma_v;
            
            model.resampling_strategy = resampling_strategy;
       end
       
       function xhk = estimator(model,k,z,u,dt,H)
           k = k + 1;
           xkm1 = model.particle(:,:,k-1); % extract particles from last iteration;
           wkm1 = model.W(:,k-1);
           
           xk   = zeros(size(xkm1));     % = zeros(nx,Ns);
           wk   = zeros(size(wkm1));     % = zeros(Ns,1);

           for i = 1:model.N
               xk(:,i) = model.fx(xkm1(:,i),u,dt) + diag(normrnd(0,model.Q));
               
               % yk = model.hx(xkm1(:,i)) + diag(normrnd(0,model.R));
               % yk = H*xkm1(:,i) + diag(normrnd(0,model.R)); % 대각선 요소만 뽑아옴 or 대각행렬으로 만듬
               yk = H*xk(:,i) + diag(normrnd(0,model.R));
               
               wk(i) = wkm1(i) * (normpdf((z - yk), 0, model.sigma_v))'*(normpdf((z - yk), 0, model.sigma_v));
           end
           wk = wk./sum(wk);
           Neff = 1/sum(wk.^2);
           percentage = 0.5;
           
           Nt = percentage * model.N;
           
           if(Neff < Nt)
               [xk, wk] = model.resample(xk, wk, model.resampling_strategy);
           end
           
           xhk = zeros(size(xk,1), 1);
           for i = 1 : model.N
                xhk = xhk + wk(i) * xk(:,i);
           end
           
           model.W(:,k) = wk;
           model.particle(:,:,k) = xk;
             
       end
       
       function [xk, wk, idx] =resample(model, xk, wk, resampling_strategy)
            Ns = length(wk);  % Ns = number of particles
            
            switch resampling_strategy
               case 'multinomial_resampling'
                  with_replacement = true;
                  idx = randsample(1:Ns, Ns, with_replacement, wk);
                  xk = xk(:,idx);                    % extract new particles
                  wk = repmat(1/Ns, 1, Ns);          % now all particles have the same weight
            %{
                  THIS IS EQUIVALENT TO:
                  edges = min([0 cumsum(wk)'],1); % protect against accumulated round-off
                  edges(end) = 1;                 % get the upper edge exact
                  % this works like the inverse of the empirical distribution and returns
                  % the interval where the sample is to be found
                  [~, idx] = histc(sort(rand(Ns,1)), edges);
            %}
               case 'stratified_resampling'
                    j = 1;
                    sumQ = wk(j);
                    for i=1:Ns
                       u = (rand()+i-1)/Ns;
                       while sumQ < u
                           j = j+1;
                           sumQ = sumQ + wk(j);
                       end
                       xk(:,i) = xk(:,j);              % extract new particles
                    end
                    wk = repmat(1/Ns, 1, Ns);          % now all particles have the same weight
                    
               case 'systematic_resampling'
                  % this is performing latin hypercube sampling on wk
                  edges = min([0 cumsum(wk)'],1); % protect against accumulated round-off
                  edges(end) = 1;                 % get the upper edge exact
                  u1 = rand/Ns;
                  % this works like the inverse of the empirical distribution and returns
                  % the interval where the sample is to be found
                  [~, idx] = histc(u1:1/Ns:1, edges);
                  xk = xk(:,idx);                    % extract new particles
                  wk = repmat(1/Ns, 1, Ns);          % now all particles have the same weight

                case 'Residual_resampling'
                    Nr=Ns; jj=0;
                    for i=1:Ns
                        j=floor(wk(i)*Ns);
                        for ii=1:j
                            jj=jj+1;
                            xk(:,jj)=xk(:,i);
                        end
                        wk(i)=wk(i)-j/Ns;
                        Nr=Nr-j;
                     end
                     if Nr>0
                         for i=1:Ns
                            wk(i)=wk(i)*Ns/Nr;
                         end
                        j=1; sumQ=wk(1);
                        for i=1:Nr
                            u=(rand()+i-1)/Nr;
                            while sumQ<u
                                j=j+1;
                                sumQ=sumQ+wk(j);
                            end
                            jj=jj+1;
                            xk(:,jj)=xk(:,j);
                        end
                     end
                     
                case 'Metropolis_resampling'
                    Nc = 10;
                    for i = 1:Ns
                        ii=i;
                       for j = 1:Nc
                          u = rand();
                          jj = floor(rand()*Ns) + 1;
                          if u<=wk(jj)/wk(ii)
                             ii=jj; 
                          end
                       end
                       xk(i)=xk(ii);
                    end
                    
                case 'Rejection_resampling'
                    maxW = max(wk);
                    for i=1:Ns
                       j=1;
                       u=rand();
                       while u>wk(j)/maxW
                          j=floor(rand()*Ns) + 1;
                          u=rand();
                       end
                       xk(i) = xk(j);
                    end
                case 'Partial_resampling'
                    
                    
               otherwise
                  error('Resampling strategy not implemented')                 
            end
            
                          
       end
      
       % system model f(x), h(x)
        function xp = fx(model,x,u,dt)
            A = eye(3);
            theta = x(3);
            B(1,1)=cos(theta);
            B(1,2)=-dt*sin(theta);

            B(2,1)=sin(theta);
            B(2,2)=dt*cos(theta);
            
            B(3,1)=0;
            B(3,2)=1;
            
            B=B*dt;
            
            xp = A*x + B*u;
        end
        
        function z = hx(model,x)
            z(1,1) = sqrt((x(1) - a1(1))^2+(x(2) - a1(2))^2);
            z(2,1) = sqrt((x(1) - a2(1))^2+(x(2) - a2(2))^2);
            z(3,1) = sqrt((x(1) - a3(1))^2+(x(2) - a3(2))^2);
            z(4,1) = sqrt((x(1) - a4(1))^2+(x(2) - a4(2))^2);
            z(5,1) = x(3);
        end
   end
end