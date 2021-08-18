classdef HinfFilter < handle % handle class so properties persist
    properties
        % System model
        F,H;
        
        % Update estimate variable
        x,P;
        
        % Parameter design
        L,S,Q,R,delta;
        
        S_bar;
    end
    methods
        function model = HinfFilter(F,H,x0,P0,Q,R,L,S,delta)
            model.F = F;
            model.H = H;
            
            model.x = x0;
            model.P = P0;
            
            model.L = L;
            model.S = S;
            model.Q = Q;
            model.R = R;
            model.delta = delta;
            
            model.S_bar = L'*S*L;
        end
        
        function [xhat,z] = H_inf(model,y)
                
            P_bar=eye(size(model.S_bar)) - model.delta.*model.S_bar*model.P + model.H' /model.R * model.H*model.P;
            
            K = model.P/P_bar*model.H'/model.R;
            xhat = model.F*model.x + model.F*K*(y-model.H*model.x);
            model.P = model.F*model.P/P_bar*model.F' * model.Q;               
         
            z = model.L*xhat;
            model.x=xhat;
        end
        
    end
end