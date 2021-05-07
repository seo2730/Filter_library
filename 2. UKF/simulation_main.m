close all
clear all
%% initialization
s = [0, 0, 0]';
z = [0, 0, 0]';
u = [0, 0]';
s_size = size(s,1);
z_size = size(z,1);
u_size = size(u,1);

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

%% simulation
load('simulation_data.mat')
simulation_step = size(saved_s,2);
saved_s_hat_first = zeros(s_size,simulation_step);
gen_x0 = @(x) normrnd(0, sqrt(10));
N = 50;
W = zeros(N,simulation_step);
W(:,1) = repmat(1/N,N,1);
particle = zeros(3,N,simulation_step);
sigma_u = sqrt(1);
sigma_v = sqrt(2);

% Noise
Q = 0.1*eye(3);%normrnd(0, sigma_u);
% R = 10*eye(5,5);
R = 0.5*eye(3);%normrnd(0, sigma_v);
P0 = eye(3,3);

for i = 1:N
   particle(:,i,1) = saved_s(:,1); %[10 10 10]';
end

kappa = 5; m = 3;
Filtering_UKF = UKF(Q,R,s,P0,kappa);

for k = 1:simulation_step
    
    s = saved_s(:,k);
    z = saved_z(:,k);
    u = saved_u(:,k);
    %% filter
    % UKF
    est_UKF(:,k) = Filtering_UKF.estimator(m,z,u,H);
end

% ERROR FIRST
Error_s_first_data = zeros(s_size,simulation_step);
for i = 1:simulation_step  
    Error_s_first_data(:,i) = saved_s(:,i) - est_UKF(:,i);
end
%% Calculate RMSE FIRST
RMSE_interval = 20:simulation_step;

RMSE_x1_first = sqrt(mean(Error_s_first_data(1,RMSE_interval).^2));
RMSE_x2_first = sqrt(mean(Error_s_first_data(2,RMSE_interval).^2));
RMSE_x3_first = sqrt(mean(Error_s_first_data(3,RMSE_interval).^2));
%% ERROR SECOND
% Error_s_second_data = zeros(s_size,simulation_step);
% for i = 1:simulation_step
%    Error_s_second_data(:,i) = saved_s(:,i) - saved_s_hat_second(:,i);
% end
% %% Calculate RMSE SECOND
% RMSE_interval = 20:simulation_step;
% RMSE_x1_second = sqrt(mean(Error_s_second_data(1,RMSE_interval).^2));
% RMSE_x2_second = sqrt(mean(Error_s_second_data(2,RMSE_interval).^2));
% RMSE_x3_second = sqrt(mean(Error_s_second_data(3,RMSE_interval).^2));
%% PLOT
figure(1)
plot(saved_s(1,:), saved_s(2,:),  'r*-','Displayname', 'state'); hold on; grid on;
plot(saved_z(1,:), saved_z(2,:),  'b*-','Displayname', 'measurement'); 
plot(est_UKF(1,:), est_UKF(2,:),  'm*-','LineWidth',1.5, 'Displayname', 'UKF');
title('measurement position')
legend('Location','northeast')
%% state ERROR
time_interval = 1:simulation_step;
figure(2)
subplot(3,1,1)
plot(time_interval, Error_s_first_data(1,:), 'm*-','Displayname','UKF');
title('state error x')
legend('Location','southeast')
subplot(3,1,2)
plot(time_interval, Error_s_first_data(2,:), 'm*-','Displayname','UKF');
title('state error y')
legend('Location','southeast')
subplot(3,1,3)
plot(time_interval, Error_s_first_data(3,:), 'm*-','Displayname','UKF');
title('state error theta')
legend('Location','southeast')