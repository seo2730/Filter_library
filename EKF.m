classdef EKF < handle % handle class so properties persist
    properties
        % system model
        F,H;
        Jacobian_F,Jacobian_H;
        Q,R;
        x_pre,P;
    end
    methods
        function model = EKF(Q,R,x0,P)      
            model.Q = Q;
            model.R = R;
            
            model.x_pre = x0;
            model.P = P;           
        end
        
        function xhat = estimator(model,z,vel,ang_vel,dt,a1,a2,a3,a4)
           xp = model.f(model.x_pre,vel,ang_vel,dt);
           jf = model.jacobian_F(model.x_pre,ang_vel,dt);
           Pp = jf*model.P*jf' + model.Q;
           
           jh = model.jacobian_H(model.x_pre,a1,a2,a3,a4);
           S = jh*Pp*jh' + model.R;
           K = Pp*jh'*S^-1;
           
           hx = model.h(model.x_pre,a1,a2,a3,a4);
           xhat = xp + K*(z-hx);
           model.P = Pp - K*jh*Pp;
           
           model.x_pre = xhat;
        end
        
        %%%% System model %%%%
        function xp = f(model,x,vel,ang_vel,dt)  
            xp(1,1) = x(1) + vel*dt * cosd(x(3) + 0.5*ang_vel*dt);
            xp(2,1) = x(2) + vel*dt * sind(x(3) + 0.5*ang_vel*dt);
            xp(3,1) = x(3) + ang_vel*dt;
        end
        
        function z = h(model,x,a1,a2,a3,a4)
            z(1,1) = sqrt((x(1) - a1(1))^2+(x(2) - a1(2))^2);
            z(2,1) = sqrt((x(1) - a2(1))^2+(x(2) - a2(2))^2);
            z(3,1) = sqrt((x(1) - a3(1))^2+(x(2) - a3(2))^2);
            z(4,1) = sqrt((x(1) - a4(1))^2+(x(2) - a4(2))^2);
            z(5,1) = x(3); 
        end
        
        function jf = jacobian_F(model,x,ang_vel,dt)  
            jf(1,1) = 1;
            jf(1,2) = 0;
            jf(1,3) = -(pi*sin((pi*(x(3) + 0.5*ang_vel*dt))/180))/3600;

            jf(2,1) = 0;
            jf(2,2) = 1;
            jf(2,3) = (pi*cos((pi*(x(3) + 0.5*ang_vel*dt))/180))/3600;

            jf(3,1) = 0;
            jf(3,2) = 0;
            jf(3,3) = 1;
        end
        
        function jh = jacobian_H(model,x,a1,a2,a3,a4)  
            jh(1,1) = 2*(x(1)-a1(1))/((x(1)-a1(1))^2 + (x(2) - a1(2))^2)^(1/2);
            jh(1,2) = 2*(x(2)-a1(2))/((x(1)-a1(1))^2 + (x(2) - a1(2))^2)^(1/2);
            jh(1,3) = 0;
            
            jh(2,1) = 2*(x(1)-a2(1))/((x(1)-a2(1))^2 + (x(2) - a2(2))^2)^(1/2);
            jh(2,2) = 2*(x(2)-a2(2))/((x(1)-a2(1))^2 + (x(2) - a2(2))^2)^(1/2);
            jh(2,3) = 0;
        
            jh(3,1) = 2*(x(1)-a3(1))/((x(1)-a3(1))^2 + (x(2) - a3(2))^2)^(1/2);
            jh(3,2) = 2*(x(2)-a3(2))/((x(1)-a3(1))^2 + (x(2) - a3(2))^2)^(1/2);
            jh(3,3) = 0;
            
            jh(4,1) = 2*(x(1)-a4(1))/((x(1)-a4(1))^2 + (x(2) - a4(2))^2)^(1/2);
            jh(4,2) = 2*(x(2)-a4(2))/((x(1)-a4(1))^2 + (x(2) - a4(2))^2)^(1/2);
            jh(4,3) = 0;
            
            jh(5,1) = 0;
            jh(5,2) = 0;
            jh(5,3) = 1;
        end
    end
end