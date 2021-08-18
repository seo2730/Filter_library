classdef KalmanFilter < handle % handle class so properties persist
    properties
        % system model
        F,H;
        
        % Update estimate variable
        x,P;
        
        % Noise
        Q,R;
    end
    methods
        function model = KalmanFilter(F,H,x0,P0,Q,R)
            model.F = F;
            model.H = H;
            
            model.x = x0;
            model.P = P0;
            
            model.Q = Q;
            model.R = R;
        end
        
        function xhat = Kalman(model,y)
            xp = model.F * model.x;
            Pp = model.F*model.P*model.F' + model.Q;
           
            K = Pp*model.H'/(model.H*Pp*model.H' + model.R);
           
            xhat = xp + K*(y - model.H*xp);
            model.P = Pp - K*model.H*Pp;
                  
            model.x = xhat;
        end        
    end
end