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
%% save data
simulation_step = 100;
saved_x = zeros(x_size,simulation_step);
saved_x_real = zeros(x_size,simulation_step);
saved_y = zeros(y_size,simulation_step);
saved_u = zeros(u_size,simulation_step);

%% simulation 
horizon_size = 10;
for k = 1:simulation_step

%     w = sqrt(0.0)*ones(x_size,1);
    w = 0.001*randn(x_size,1);
%     v = 0.1*randn(x_size,1);
    v = 0.0*randn(x_size,1);

    %% Obtain A,B Matrix
    [A, B] = ABjacob(x, u, dt);
    [A_real, B_real] = ABjacob_real(x_real, u, dt);
    %% system update
    x = A*x + B*u + G*w;
    x_real = A_real*x_real + B_real*u;
    y = C*x + v;
    
    saved_x(:,k) = x;
    saved_x_real(:,k) = x_real;
    saved_y(:,k) = y;
    saved_u(:,k) = u;
    saved_w(:,k) = w;
    saved_v(:,k) = v;
    
  
%     u = [2  2*sin(0.5*k)]';  
    u = [2  2]';
end

saved_y1 = saved_y(1,:);
saved_y2 = saved_y(2,:);
saved_u1 = saved_u(1,:);
saved_u2 = saved_u(2,:);

save('simulation_data','saved_x','saved_y','saved_u')