classdef UFIR < handle
   properties
       % System model
       F,E,H;
       
       % GNPG
       G;
       
       x;
       
       % Horizon size
       N;
   end
   
   methods
       function model = UFIR(F,E,H,N,x0) %(F,E,H,N)
            model.F = F;
            model.E = E;
            model.H = H;
            
            model.G = H;
            
            model.N = N; 
            model.x = x0;
       end
       
       function xhat = iterative_estimator(model,Y,U)
           [L,S] = model.MakeBigMatrices(model.F(:,1:60),model.E(:,1:48),model.H,model.N/2);
           H_bar = S/(L'*L)*L';
           model.G = (H_bar'*H_bar)^-1;
           
           xs = model.G*H_bar'*(Y-L*U)+S(31:45,:)*U;
           
           for k=(model.N/2):model.N
                if k==model.N/2
                    xp = model.F(:,(15*k-14):15*k)*xs((15*k-14):15*k,:) + model.E(:,(15*k-14):15*k)*u;
                    model.G = (model.H'*model.H + (model.F(:,(15*k-14):15*k)*model.G((15*k-14):15*k,(15*k-14):15*k)*model.F(:,(15*k-14):15*k)')^-1)^-1;
                    K = model.G*model.H';
                    xk = [zeros(9,1); delta_u_h] + K*(y-model.H*xp);            
                else
                    xp = model.F(:,(15*k-14):15*k)*xp + model.E(:,(15*k-14):15*k)*u;
                    model.G = (model.H'*model.H + (model.F(:,(15*k-14):15*k)*model.G((15*k-14):15*k,(15*k-14):15*k)*model.F(:,(15*k-14):15*k)')^-1)^-1;
                    K = model.G*model.H';
                    xk = [zeros(9,1); delta_u_h] + K*(y-model.H*xp);
                end
           end
           xhat = xk;
       end
       
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
       
       function [L,S] = MakeBigMatrices(model,A,B,C,N)
            % parameter setting
            N_system = size(A(:,1:15),1); N_input = size(B(:,1:12),2); N_output = size(C,1);
           
            % Error check
            IsInput = 1;
            
            % initialization
            if B == 0
                IsInput = 0;
                S = 0;
            else
                S = B;
            end
            
            for i=1:N
                if i==1
                    if IsInput == 1
                        S = B(:,(12*i-11):12*i);
                        S_i = [A(:,(15*(i+1)-14):15*(i+1))*S B(:,(12*(i+1)-11):12*(i+1))];
                    end
                    
                    %H = C;
                    C_bar = C;
                elseif i==2
                    if IsInput == 1
                        S = [S zeros(N_system,N_input); S_i];
                        S_i = [A(:,(15*(i+1)-14):15*(i+1))*S_i B(:,(12*(i+1)-11):12*(i+1))];
                    end
                    
                    %H = [C*(A(:,(15*i-14):15*i))^-1;H];
                    C_bar = blkdiag(C,C_bar);
                
                elseif i==N
                    if IsInput == 1
                        S = [S zeros(N_system*(i-1),N_input); S_i];
                        S_i = [A(:,(15*(i+1)-14):15*(i+1))*S_i B(:,(12*(i+1)-11):12*(i+1))];
                    end
                    C_bar = blkdiag(C,C_bar);
                else
                    if IsInput == 1
                        S = [S zeros(N_system*(i-1),N_input); S_i];
                        S_i = [A(:,(15*(i+1)-14):15*(i+1))*S_i B(:,(12*(i+1)-11):12*(i+1))];
                    end
                    
                     %H = [C*(A(:,(15*i-14):15*i))^-1;H];
                     C_bar = blkdiag(C,C_bar);
                end
            end
            
            L = C_bar*S;
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