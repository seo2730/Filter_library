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
global A B H G A_real B_real
A = zeros(x_size);
B = zeros(x_size, u_size);
A_real = zeros(x_size);
B_real = zeros(x_size, u_size);
H = eye(x_size);
G = eye(x_size);
%% save data
simulation_step = 100;
saved_x = zeros(x_size,simulation_step);
saved_x_real = zeros(x_size,simulation_step);
saved_y = zeros(y_size,simulation_step);
saved_u = zeros(u_size,simulation_step);

%% simulation 
for k = 1:simulation_step

%     w = sqrt(0.0)*ones(x_size,1);
    w = 0.01*randn(x_size,1);
%     v = 0.1*randn(x_size,1);
    v = 0.0*randn(x_size,1);

    %% Obtain A,B Matrix
    [A, B] = ABjacob(x, u, dt);
    [A_real, B_real] = ABjacob_real(x_real, u, dt);
    %% system update
    x = A*x + B*u + G*w;
    x_real = A_real*x_real + B_real*u;
    y = H*x + v;
    test = UFIR(5,x);
    [F,E,H_bar] = test.stack(A,B,H);
    if k>5
        disp(E)
    end
    saved_x(:,k) = x;
    saved_x_real(:,k) = x_real;
    saved_y(:,k) = y;
    saved_u(:,k) = u;

%     u = [2  2*sin(0.5*k)]';  
    u = [2  2]';
end

saved_y1 = saved_y(1,:);
saved_y2 = saved_y(2,:);
saved_u1 = saved_u(1,:);
saved_u2 = saved_u(2,:);


%% measurement position
figure(1)
plot(saved_y1, saved_y2,  '*-', 'color', [0.9 0.3 0.3], 'Displayname', 'measurement');  hold on; grid on;
title('measurement position')
%% measurement position
figure(2)
plot(saved_x(1,:), saved_x(2,:),  '*-', 'color', [0.3 0.3 0.3], 'Displayname', 'measurement');  hold on; grid on;
plot(saved_x_real(1,:), saved_x_real(2,:),  '*-', 'color', [0.9 0.3 0.3], 'Displayname', 'measurement'); 
title('state position')






