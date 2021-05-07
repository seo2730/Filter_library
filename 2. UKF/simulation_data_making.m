close all
clear all
%% initialization
s = [0, 0, 0]';
z = [0, 0, 0]';
s = [0, 0, 0]';
z = [0, 0, 0]';
u = [0, 0]';
s_size = size(s,1);
z_size = size(z,1);
u_size = size(u,1);
simulation_step = 50;
%% Nonlinaer State equation, measurement equation (mobile robot, kinematic model)
syms x y theta      % state variables
syms u1 u2          % input variables, u1: 이동 거리, u2: 각도 변화

f1(x,y,theta,u1,u2) = x + u1 * cos(theta + 1/2 * u2);
f2(x,y,theta,u1,u2) = y + u1 * sin(theta + 1/2 * u2);
f3(x,y,theta,u1,u2) = theta + u2;
f = [f1 f2 f3]';
clear f1 f2 f3
Jacobian_F = jacobian(f,[x y theta]);
 
%% Nonlinear measurement
% h1(x,y,theta,u1,u2) = sqrt((x-x1)^2 + (y-y1)^2) - sqrt((x-x2)^2 + (y-y2)^2);
% h2(x,y,theta,u1,u2) = sqrt((x-x1)^2 + (y-y1)^2) - sqrt((x-x3)^2 + (y-y3)^2);
% h3(x,y,theta,u1,u2) = sqrt((x-x1)^2 + (y-y1)^2) - sqrt((x-x4)^2 + (y-y4)^2);
% h4(x,y,theta,u1,u2) = theta;
% h = [h1 h2 h3 h4]';
% clear h1 h2 h3 h4
% Jacobian_H = jacobian(h,[x y theta]);
% Linear measurement
%% linear measurement(I)
 H = eye(s_size); 

%% save data
saved_s = zeros(s_size,simulation_step);
saved_z = zeros(z_size,simulation_step);
saved_u = zeros(u_size,simulation_step);

%% simulation

for k = 1:simulation_step
    w = 0.01*randn(s_size,1);
    v = 0.5*randn(s_size,1);
    
    arguments = num2cell([s',u']);
    s= f(arguments{:});
    s = double(s);
    z = H*s+ v;
    saved_s(:,k) = s;
    saved_z(:,k) = z;
    saved_u(:,k) = u;
    u = [5, 0.32]';

end
saved_z1 = saved_z(1,:);
saved_z2 = saved_z(2,:);

%% measurement position
figure(1)
plot(saved_s(1,:), saved_s(2,:),  '*-', 'color', [0.3 0.3 0.3], 'Displayname', 'state'); hold on; grid on;
plot(saved_z1, saved_z2,  '*-', 'color', [0.9 0.3 0.3],'LineWidth',1.5,'Displayname', 'measurement');  
title('measurement position')
legend('Location','northeast')
save('simulation_data','saved_s','saved_z','saved_u')