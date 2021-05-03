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
N = 500;
W = zeros(N,simulation_step);
W(:,1) = repmat(1/N,N,1);
particle = zeros(3,N,simulation_step);
sigma_u = sqrt(1);
sigma_v = sqrt(2);

% Noise
Q = 0.01*eye(3);%normrnd(0, sigma_u);
% R = 10*eye(5,5);
R = 0.5*eye(3);%normrnd(0, sigma_v);

for i = 1:N
   particle(:,i,1) = saved_s(:,1); %[10 10 10]';
end

Filtering_PF1 = PF(particle,N,W,Q,R,sigma_u,sigma_v,'multinomial_resampling');
Filtering_PF2 = PF(particle,N,W,Q,R,sigma_u,sigma_v,'systematic_resampling');
Filtering_PF3 = PF(particle,N,W,Q,R,sigma_u,sigma_v,'stratified_resampling');
Filtering_PF4 = PF(particle,N,W,Q,R,sigma_u,sigma_v,'Residual_resampling');
Filtering_PF5 = PF(particle,N,W,Q,R,sigma_u,sigma_v,'Metropolis_resampling');
Filtering_PF6 = PF(particle,N,W,Q,R,sigma_u,sigma_v,'Rejection_resampling');

Filtering_UFIR = UFIR(0,0,H,2,saved_s(:,1));

for k = 1:simulation_step
    
    s = saved_s(:,k);
    z = saved_z(:,k);
    u = saved_u(:,k);
    %% filter
    % PF 
    est_PF1(:,k) = Filtering_PF1.estimator(k,z,u,H); 
    est_PF2(:,k) = Filtering_PF2.estimator(k,z,u,H); 
    est_PF3(:,k) = Filtering_PF3.estimator(k,z,u,H); 
    est_PF4(:,k) = Filtering_PF4.estimator(k,z,u,H); 
    est_PF5(:,k) = Filtering_PF5.estimator(k,z,u,H); 
    est_PF6(:,k) = Filtering_PF6.estimator(k,z,u,H); 
    
    % UFIR
    %est_UFIR(:,k) = Filtering_UFIR.extended_UFIR(z,u);
end

% ERROR FIRST
Error_s_first_data = zeros(s_size,simulation_step);
for i = 1:simulation_step
    Error_s_first_data1(:,i) = saved_s(:,i) - est_PF1(:,i);
    Error_s_first_data2(:,i) = saved_s(:,i) - est_PF2(:,i);
    Error_s_first_data3(:,i) = saved_s(:,i) - est_PF3(:,i);
    Error_s_first_data4(:,i) = saved_s(:,i) - est_PF4(:,i);
    Error_s_first_data5(:,i) = saved_s(:,i) - est_PF5(:,i);
    Error_s_first_data6(:,i) = saved_s(:,i) - est_PF6(:,i);
    %Error_s_first_data(:,i) = saved_s(:,i) - est_UFIR(:,i);
end
%% Calculate RMSE FIRST
RMSE_interval = 20:simulation_step;
RMSE_x1_first1 = sqrt(mean(Error_s_first_data1(1,RMSE_interval).^2));
RMSE_x2_first1 = sqrt(mean(Error_s_first_data1(2,RMSE_interval).^2));
RMSE_x3_first1 = sqrt(mean(Error_s_first_data1(3,RMSE_interval).^2));

RMSE_x1_first2 = sqrt(mean(Error_s_first_data2(1,RMSE_interval).^2));
RMSE_x2_first2 = sqrt(mean(Error_s_first_data2(2,RMSE_interval).^2));
RMSE_x3_first2 = sqrt(mean(Error_s_first_data2(3,RMSE_interval).^2));

RMSE_x1_first3 = sqrt(mean(Error_s_first_data3(1,RMSE_interval).^2));
RMSE_x2_first3 = sqrt(mean(Error_s_first_data3(2,RMSE_interval).^2));
RMSE_x3_first3 = sqrt(mean(Error_s_first_data3(3,RMSE_interval).^2));

RMSE_x1_first4 = sqrt(mean(Error_s_first_data4(1,RMSE_interval).^2));
RMSE_x2_first4 = sqrt(mean(Error_s_first_data4(2,RMSE_interval).^2));
RMSE_x3_first4 = sqrt(mean(Error_s_first_data4(3,RMSE_interval).^2));

RMSE_x1_first5 = sqrt(mean(Error_s_first_data5(1,RMSE_interval).^2));
RMSE_x2_first5 = sqrt(mean(Error_s_first_data5(2,RMSE_interval).^2));
RMSE_x3_first5 = sqrt(mean(Error_s_first_data5(3,RMSE_interval).^2));

RMSE_x1_first6 = sqrt(mean(Error_s_first_data6(1,RMSE_interval).^2));
RMSE_x2_first6 = sqrt(mean(Error_s_first_data6(2,RMSE_interval).^2));
RMSE_x3_first6 = sqrt(mean(Error_s_first_data6(3,RMSE_interval).^2));
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
% plot(saved_z(1,:), saved_z(2,:),  'k*-','Displayname', 'measurement'); 
plot(est_PF1(1,:), est_PF1(2,:),  'k*-','LineWidth',1.5, 'Displayname', 'multinomial resampling');
plot(est_PF2(1,:), est_PF2(2,:),  'g*-','LineWidth',1.5, 'Displayname', 'systematic resampling');
plot(est_PF3(1,:), est_PF3(2,:),  'c*-','LineWidth',1.5, 'Displayname', 'stratified resampling');
plot(est_PF4(1,:), est_PF4(2,:),  'b*-','LineWidth',1.5, 'Displayname', 'Residual resampling');
plot(est_PF5(1,:), est_PF5(2,:),  'y*-','LineWidth',1.5, 'Displayname', 'Metropolis resampling');
plot(est_PF6(1,:), est_PF6(2,:),  'm*-','LineWidth',1.5, 'Displayname', 'Rejection resampling');
% plot(est_UFIR(1,:), est_UFIR(2,:),  '*-', 'color', [0.3 0.3 0.9],'LineWidth',1.5, 'Displayname', 'estimate');
title('measurement position')
legend('Location','northeast')
% save('simulation_data','saved_s','saved_z','saved_u')
%% state ERROR
time_interval = 1:simulation_step;
figure(2)
subplot(3,1,1)
plot(time_interval, Error_s_first_data1(1,:), 'r*-','Displayname','multinomial resampling');
hold on
plot(time_interval, Error_s_first_data2(1,:), 'k*-','Displayname','systematic resampling');
plot(time_interval, Error_s_first_data3(1,:), 'g*-','Displayname','stratified resampling');
plot(time_interval, Error_s_first_data4(1,:), 'c*-','Displayname','Residual resampling');
plot(time_interval, Error_s_first_data5(1,:), 'b*-','Displayname','Metropolis resampling');
plot(time_interval, Error_s_first_data6(1,:), 'm*-','Displayname','Rejection resampling');
hold off
title('state error x')
legend('Location','southeast')
subplot(3,1,2)
plot(time_interval, Error_s_first_data1(2,:), 'r*-','Displayname','multinomial resampling');
hold on
plot(time_interval, Error_s_first_data2(2,:), 'k*-','Displayname','systematic resampling');
plot(time_interval, Error_s_first_data3(2,:), 'g*-','Displayname','stratified resampling');
plot(time_interval, Error_s_first_data4(2,:), 'c*-','Displayname','Residual resampling');
plot(time_interval, Error_s_first_data5(2,:), 'b*-','Displayname','Metropolis resampling');
plot(time_interval, Error_s_first_data6(2,:), 'm*-','Displayname','Rejection resampling');
hold off
title('state error y')
legend('Location','southeast')
subplot(3,1,3)
plot(time_interval, Error_s_first_data1(3,:), 'r*-','Displayname','multinomial resampling');
hold on
plot(time_interval, Error_s_first_data2(3,:), 'k*-','Displayname','systematic resampling');
plot(time_interval, Error_s_first_data3(3,:), 'g*-','Displayname','stratified resampling');
plot(time_interval, Error_s_first_data4(3,:), 'c*-','Displayname','Residual resampling');
plot(time_interval, Error_s_first_data5(3,:), 'b*-','Displayname','Metropolis resampling');
plot(time_interval, Error_s_first_data6(3,:), 'm*-','Displayname','Rejection resampling');
hold off
title('state error theta')
legend('Location','southeast')