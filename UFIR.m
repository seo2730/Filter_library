classdef UFIR < handle
   properties
       % System model & Noise size
       sizeA,sizeB,sizeC;
       sizeG;
       
       % Stack Data
       A_stack,B_stack,C_stack;
       
       % Horizon size
       N;
   end
   
   methods
       function model = UFIR(sizeA,sizeB,sizeC,sizeG,N)
            model.sizeA = sizeA;
            model.sizeB = sizeB;
            model.sizeC = sizeC;
            model.sizeG = sizeG;
           
            model.N = N; 
       end
       
       function xhat = batch_form(model,xm,Unm,Wnm,Vnm,Fnm,Enm,Hnm,Snm,Lnm)
           Ynm = Hnm*xm + Snm*Unm' + Lnm*Wnm' + Vnm';
           % Ynm = Hnm*xm + Snm*Unm';
           
           Knm = Fnm(1:model.sizeA(1),1:model.sizeA(2))*(Hnm'*Hnm)^-1*Hnm';
           xhat = Knm*Ynm + (Enm(1:model.sizeB(1),:)-Knm*Snm)*Unm';
       end
       
%        function xhat = iterative_estimator(model,Y,U)
%            [L,S] = model.MakeBigMatrices(model.F(:,1:60),model.E(:,1:48),model.H,model.N/2);
%            H_bar = S/(L'*L)*L';
%            model.G = (H_bar'*H_bar)^-1;
%            
%            xs = model.G*H_bar'*(Y-L*U)+S(31:45,:)*U;
%            
%            for k=(model.N/2):model.N
%                 if k==model.N/2
%                     xp = model.F(:,(15*k-14):15*k)*xs((15*k-14):15*k,:) + model.E(:,(15*k-14):15*k)*u;
%                     model.G = (model.H'*model.H + (model.F(:,(15*k-14):15*k)*model.G((15*k-14):15*k,(15*k-14):15*k)*model.F(:,(15*k-14):15*k)')^-1)^-1;
%                     K = model.G*model.H';
%                     xk = [zeros(9,1); delta_u_h] + K*(y-model.H*xp);            
%                 else
%                     xp = model.F(:,(15*k-14):15*k)*xp + model.E(:,(15*k-14):15*k)*u;
%                     model.G = (model.H'*model.H + (model.F(:,(15*k-14):15*k)*model.G((15*k-14):15*k,(15*k-14):15*k)*model.F(:,(15*k-14):15*k)')^-1)^-1;
%                     K = model.G*model.H';
%                     xk = [zeros(9,1); delta_u_h] + K*(y-model.H*xp);
%                 end
%            end
%            xhat = xk;
%        end
       
       function xhat = full_horizon_estimator(model,y,u)
            xp = model.F*model.x + model.E*u;
            model.G = (model.H'*model.H + (model.F*model.G*model.F')^-1)^-1;
            K = model.G*model.H';
            xhat = xp + K*(y-model.H*xp);
            model.x = xhat;
       end
       
       function xhat = extended_UFIR(model,y,u)
            xp = model.fx(model.x,u);
            model.F = model.jacobian_F(model.x,u);
            model.G = (model.H'*model.H + (model.F*model.G*model.F')^-1)^-1;
            K = model.G*model.H';
            xhat = xp + K*(y-model.H*xp);
            model.x = xhat;
       end
       
       function [F,E,H,L] = stack(model,A,B,C,G)
            persistent A_stack B_stack C_stack G_stack;
            if isempty(A_stack)
                A_stack = A;
            else
                A_stack = [A A_stack];
                if size(A_stack,2)>model.N*size(A,2)
                    A_stack(:,(model.N+1)*size(A,2)-size(A,2)+1:(model.N+1)*size(A,2)) = [];
                end
            end
            
            if isempty(B_stack)
                B_stack = B;   
            else
                B_stack = [B B_stack];
                if size(B_stack,2)>model.N*size(B,2)
                    B_stack(:,(model.N+1)*size(B,2)-size(B,2)+1:(model.N+1)*size(B,2)) = [];
                end
            end
            
            if isempty(C_stack)
                C_stack = C;   
            else
                C_stack = [C,C_stack];
                if size(C_stack,2)>model.N*size(C,2)
                    C_stack(:,(model.N+1)*size(C,2)-size(C,2)+1:(model.N+1)*size(C,2)) = [];
                end
            end
            
            if isempty(G_stack)
                G_stack = G;   
            else
                G_stack = [G,G_stack];
                if size(G_stack,2)>model.N*size(G,2)
                    G_stack(:,(model.N+1)*size(G,2)-size(G,2)+1:(model.N+1)*size(G,2)) = [];
                end
            end
            
            F = A_stack;
            E = B_stack;
            H = C_stack;
            L = G_stack;
       end
       
       function [Unm,Wnm,Vnm,Fnm,Enm,Hnm,Snm,Lnm] = MakeBigMatrices(model,u,w,v,F,E,H,L)
            % parameter setting
            N_system = model.sizeA(1); N_input = model.sizeB(1); N_output = model.sizeC(1); N_noise = model.sizeG(1);
            M_system = model.sizeA(2); M_input = model.sizeB(2); M_output = model.sizeC(2); M_noise = model.sizeG(2);
            
            % Input check
            IsInput = 1;
            
            % initialization
            if E == 0
                IsInput = 0;
                Enm = 0;
            end
            
            for i=1:model.N
                if i==1
                    F_i = F(:,M_system*model.N-M_system+1:M_system*model.N);
                    Latin_F = F_i;
                    Fnm = F_i';
                    
                    Unm = u(:,i)';
                    Wnm = w(:,i)';
                    Vnm = v(:,i)';
                else
                    F_i = F(1:N_system,M_system*(model.N-i+1)-M_system+1:M_system*(model.N-i+1));
                    Fnm = [(F_i*Latin_F(:,1:N_system))' Fnm];
                    Latin_F = [F_i*Latin_F(:,1:N_system) Latin_F];
                    
                    Unm = [u(:,i)' Unm];
                    Wnm = [w(:,i)' Wnm];
                    Vnm = [v(:,i)' Vnm];
                end
            end
            
            for i=1:model.N
                if i==1
                    if IsInput == 1
                        Enm = E(:,(M_input*(model.N-i+1)-M_input+1):M_input*(model.N-i+1));
                    end
                    H_bar = H(:,M_output*(model.N-i+1)-M_output+1:M_output*(model.N-i+1));
                    Gnm = L(:,(M_noise*(model.N-i+1)-M_noise+1):M_noise*(model.N-i+1));
                else
                    if IsInput == 1
                        Enm = [E(:,(M_input*(model.N-i+1)-M_input+1):M_input*(model.N-i+1)) F(:,M_system*(model.N-i+1)-M_system+1:M_system*(model.N-i+1))*Enm(1:N_input,:); zeros(size(Enm,1),M_input) Enm];
                    end
                     H_bar = blkdiag(H(:,M_output*(model.N-i+1)-N_system+1:M_output*(model.N-i+1)),H_bar);
                     Gnm = [L(:,(M_noise*(model.N-i+1)-M_noise+1):M_noise*(model.N-i+1)) F(:,M_system*(model.N-i+1)-M_system+1:M_system*(model.N-i+1))*Gnm(1:N_noise,:); zeros(size(Gnm,1),M_noise) Gnm];
                end
            end
            
            Hnm = H_bar*Fnm';
            Snm = H_bar*Enm;
            Lnm = H_bar*Gnm;
       end
       
       %%%% System model %%%%
       function xp = fx(model,x,u)
           xp(1,1) = x(1) + u(1) * cos(x(3) + 0.5*u(2));
           xp(2,1) = x(2) + u(1) * sin(x(3) + 0.5*u(2));
           xp(3,1) = x(3) + u(2);
       end
       
       function z = hx(model,x)
           z(1,1) = sqrt((x(1) - a1(1))^2+(x(2) - a1(2))^2);
           z(2,1) = sqrt((x(1) - a2(1))^2+(x(2) - a2(2))^2);
           z(3,1) = sqrt((x(1) - a3(1))^2+(x(2) - a3(2))^2);
           z(4,1) = sqrt((x(1) - a4(1))^2+(x(2) - a4(2))^2);
           z(5,1) = x(3);
       end
       
       function jf = jacobian_F(model,x,u)  
            jf(1,1) = 1;
            jf(1,2) = 0;
            jf(1,3) = -u(1)*sin(x(3) + u(2))/2;

            jf(2,1) = 0;
            jf(2,2) = 1;
            jf(2,3) = cos(x(3) + u(2)/2)*u(1);

            jf(3,1) = 0;
            jf(3,2) = 0;
            jf(3,3) = 1;
        end
   end
end