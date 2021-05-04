classdef UKF < handle % handle class so properties persist
    properties
        % Noise
        Q,R;
        % Initial value
        x,P,kappa;
    end
    
    methods
        function model = UKF(Q,R,x0,P,kappa)      
            model.Q = Q;
            model.R = R;
            
            model.x = x0;
            model.P = P;
            
            model.kappa = kappa;
        end
        
        function xhat = estimator(model,m,z,u,H)
            [Xi, W, n] = model.SigmaPoints(model.x,model.P,model.kappa);
            
            fXi = zeros(n,2*n+1);
            for k=1:2*n+1
                fXi(:,k) = model.fx(Xi(:,k),u);
            end
            
            [xp, Pp] = model.UT(fXi,W,model.Q);
    
            hXi = zeros(m,2*n+1);
            for k=1:2*n+1
               hXi(:,k) = H*Xi(:,k);
            end   
            
            [zp, Pz] = model.UT(hXi,W,model.R);
    
            Pxz = zeros(n,m);
            for k=1:2*n+1
               Pxz = Pxz + W(k)*(fXi(:,k)-xp)*(hXi(:,k)-zp)'; 
            end

            K = Pxz*inv(Pz);

            xhat = xp + K*(z-zp);
            model.P = Pp - K*Pz*K';
            
            model.x = xhat;
        end
        
        function [Xi, W, n] = SigmaPoints(model,xm,P,kappa)
            n = numel(xm);
            Xi = zeros(n,2*n+1);            % sigma points = col of Xi
            W = zeros(n,1);

            Xi(:,1) = xm;
            W(1) = kappa / (n+kappa);
            U = chol((n+kappa)*P);          % U*U = (n+kappa)*P

            for k=1:n
               Xi(:,k+1) = xm + U(k,:)';    % row of U 
               W(k+1) = 1/(2*(n+kappa));
            end

            for k=1:n
               Xi(:,n+k+1) = xm - U(k,:)';
               W(n+k+1) = 1/(2*(n+kappa));
            end          
        end
        
        function [xm, xcov] = UT(model,Xi,W,noiseCov)
            [n,kmax] = size(Xi);
    
            xm = 0;
            for k=1:kmax
               xm = xm + W(k)*Xi(:,k); 
            end

            xcov = zeros(n,n);
            for k=1:kmax
                xcov = xcov + W(k)*(Xi(:,k) - xm)*(Xi(:,k) - xm)';
            end
            xcov = xcov + noiseCov;
        end
        
        % system model f(x), h(x)
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
    end
end