classdef FIR < handle
    properties
        
    end
    methods
        function model = FIR()      
        
        end
        
        function [] = BigMatrices()
            
        end
        
        function xp = f(model,x,vel,ang_vel,dt)  
            xp(1,1) = x(1) + u(1) * cos(x(3) + 0.5*u(2));
            xp(2,1) = x(2) + u(1) * sin(x(3) + 0.5*u(2));
            xp(3,1) = x(3) + u(2);
        end
        
        function z = h(model,x,a1,a2,a3,a4)
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
            jf(2,3) = cos(theta + u(2)/2)*u(1);

            jf(3,1) = 0;
            jf(3,2) = 0;
            jf(3,3) = 1;
        end
    end
end