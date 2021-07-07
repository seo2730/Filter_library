close all
clear all
clc
%% initial state/ measurement/ input define
x = [1 1 pi/6]';
x_real = [1 1 pi/6]';
y = [1 1 pi/6]';

u = [0 0]';
dt = 0.03;
x_size = size(x,1);
x_real_size = size(x,1);
y_size = size(y,1);
u_size = size(u,1);
%% system model
global A B C G A_real B_real
A = zeros(x_size);
B = zeros(x_size, u_size);
A_real = zeros(x_size);
B_real = zeros(x_size, u_size);
C = eye(x_size);
G = eye(x_size);
%% simulation
load('simulation_data.mat')
simulation_step = 100;

% KF setting
Q = 0.1*eye(3);
R = 0.9*eye(3);
P0 = eye(3);

% PF setting
N = 100;
W = zeros(N,simulation_step);
W(:,1) = repmat(1/N,N,1);
for i = 1:N
   particle(:,i,1) = saved_x(:,1); %[10 10 10]';
end
sigma_u = sqrt(1);
sigma_v = sqrt(2);
PF_estimator = PF(particle,N,W,Q,R,sigma_u,sigma_v,'systematic_resampling');

% UFIR setting
horizon_size = 10;
for k = 1:simulation_step
    %% Obtain A,B Matrix
    [A, B] = ABjacob(saved_x(:,k), saved_u(:,k), dt);
    % KF
    KF_estimator = KalmanFilter(A,B,C,saved_x(:,k),P0,Q,R);
    xhat_KF = KF_estimator.Kalman(saved_y(:,k),saved_u(:,k));
    saved_xhat_KF(:,k) = xhat_KF;
    
    % PF
    saved_xhat_PF(:,k) = PF_estimator.estimator(k,saved_y(:,k),saved_u(:,k),dt,C); 
    
    % UFIR
    UFIR_estimator = UFIR(size(A),size(B),size(C),size(G),horizon_size);
    [F,E,H,L] = UFIR_estimator.stack(A,B,C,G);
    if k>horizon_size
        [Ynm,Unm,Fnm,Enm,Hnm,Snm,Lnm] = UFIR_estimator.MakeBigMatrices(saved_y(:,k-horizon_size+1:k),saved_u(:,k-horizon_size+1:k),F,E,H,L);
        xhat_UFIR = UFIR_estimator.batch_form(Ynm,Unm,Fnm,Enm,Hnm,Snm);
        saved_xhat_UFIR(:,k) = xhat_UFIR;
    else
        saved_xhat_UFIR(:,k) = saved_x(:,k);
    end
  
%     u = [2  2*sin(0.5*k)]';  
    u = [2  2]';
end

%% Error
for k = 1:simulation_step
    Error_s_KF(:,k) = saved_y(:,k)-saved_xhat_KF(:,k);
    Error_s_PF(:,k) = saved_y(:,k)-saved_xhat_PF(:,k);
    Error_s_UFIR(:,k) = saved_y(:,k)-saved_xhat_UFIR(:,k);
end

%% Calculate RMSE FIRST
RMSE_interval = horizon_size:simulation_step;
RMSE_x1_KF = sqrt(mean(Error_s_KF(1,RMSE_interval).^2));
RMSE_x2_KF = sqrt(mean(Error_s_KF(2,RMSE_interval).^2));
RMSE_x3_KF = sqrt(mean(Error_s_KF(3,RMSE_interval).^2));

RMSE_x1_PF = sqrt(mean(Error_s_PF(1,RMSE_interval).^2));
RMSE_x2_PF = sqrt(mean(Error_s_PF(2,RMSE_interval).^2));
RMSE_x3_PF = sqrt(mean(Error_s_PF(3,RMSE_interval).^2));

RMSE_x1_UFIR = sqrt(mean(Error_s_UFIR(1,RMSE_interval).^2));
RMSE_x2_UFIR = sqrt(mean(Error_s_UFIR(2,RMSE_interval).^2));
RMSE_x3_UFIR = sqrt(mean(Error_s_UFIR(3,RMSE_interval).^2));

%% measurement position
t = 0:dt:simulation_step*dt-dt;

figure(1)
plot(saved_y(1,:), saved_y(2,:), 'k*');
hold on; grid on;
plot(saved_xhat_KF(1,:), saved_xhat_KF(2,:), 'bo');
plot(saved_xhat_PF(1,:),saved_xhat_PF(2,:), 'mo');
plot(saved_xhat_UFIR(1,:), saved_xhat_UFIR(2,:), 'ro');
legend('Measurement state','KF estimate','PF estimate','UFIR estimate')
title('state position')

figure(2)
tiledlayout(3,1)
nexttile
plot(t,saved_y(1,:)-saved_xhat_UFIR(1,:),'r')
hold on
plot(t,saved_y(1,:)-saved_xhat_PF(1,:),'m')
plot(t,saved_y(1,:)-saved_xhat_KF(1,:),'b')
hold off
legend('UFIR error','PF error','KF error')
nexttile
plot(t,saved_y(2,:)-saved_xhat_UFIR(2,:),'r')
hold on
plot(t,saved_y(2,:)-saved_xhat_PF(2,:),'m')
plot(t,saved_y(2,:)-saved_xhat_KF(2,:),'b')
hold off
legend('UFIR error','PF error','KF error')
nexttile
plot(t,saved_y(3,:)-saved_xhat_UFIR(3,:),'r')
hold on
plot(t,saved_y(3,:)-saved_xhat_PF(3,:),'m')
plot(t,saved_y(3,:)-saved_xhat_KF(3,:),'b')
hold off
legend('UFIR error','PF error','KF error')
