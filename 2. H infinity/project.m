clc
clear all

dt = 0.1;
Nsamples = 1000;

x_original = zeros(4,Nsamples);
y_original = zeros(2,Nsamples);
est_K = zeros(4,Nsamples);
est_H = zeros(4,Nsamples);

A = [1 0 dt 0;
     0 1 0  dt;
     0 0 1  0;
     0 0 0  1];
 
A1 = [1 0 0.5 0;
      0 1 0  0.5;
      0 0 1  0;
      0 0 0  1];
 
C = [1 0 0 0;
     0 1 0 0];

for k=1:Nsamples
    if k==1
        x = [0 0 1 0.5]';
        
    elseif k>=200 && k<300
        x = A1 * x;       
    else
        x = A * x;
    end
    
    y = C*x;
    
    x_original(:,k) = x;
    y_original(:,k) = y;
end

% time step
t = 0:dt:Nsamples*dt-dt;

% Noise(measurement 위치에서만 노이즈(w))
Pos = awgn(y_original,30,'measured');

% 공분산 행렬 크기 조심!!!
kal = KalmanFilter(A,C,x_original(:,1),zeros(4,4),0.000001*eye(4,4), 10*eye(2,2));
% Design parameter 만들 때 P^-1 - delta * S_bar + H'R^-1*H > 0 인 조건 생각해야함
Hinf = HinfFilter(A,C,x_original(:,1),eye(4,4),0.01*eye(4,4),10*eye(2,2),eye(4,4), eye(4,4),0.5);

% k에 따라 Pos 대입해보자.
for k = 1:Nsamples
    est_K(:,k) = kal.Kalman(Pos(:,k));
    est_H(:,k) = Hinf.H_inf(Pos(:,k));
end

for i = 1:Nsamples
    figure(1)
    plot(y_original(1,i), y_original(2,i),'bo')
    hold on
    plot(Pos(1,i),Pos(2,i),'ro')
    plot(est_K(1,i),est_K(2,i),'ko')
    plot(est_H(1,i),est_H(2,i),'go')
    axis([-10 200 -10 200])
    drawnow
    hold off
end
 
figure(2)
plot(y_original(1,:), y_original(2,:),'bo')
hold on
plot(Pos(1,:),Pos(2,:),'ro')
plot(est_K(1,:),est_K(2,:),'ko')
plot(est_H(1,:),est_H(2,:),'go')
hold off
legend('real robot ','real + noise robot','Kalman','H infinity')

figure(3)
plot(t,y_original(1,:),'b')
hold on
plot(t,Pos(1,:),'r--')
plot(t,est_K(1,:),'k--')
plot(t,est_H(1,:),'g--')
hold off
legend('real robot ','real + noise robot','Kalman','H infinity')

figure(4)
plot(t,y_original(2,:),'b')
hold on
plot(t,Pos(2,:),'r--')
plot(t,est_K(2,:),'k--')
plot(t,est_H(2,:),'g--')
hold off
legend('real robot ','real + noise robot','Kalman','H infinity')

figure(5)
plot(t,y_original(1,:)-est_K(1,:),'b')
hold on
plot(t,y_original(1,:)-est_H(1,:),'r--')
hold off
legend('error X position by Kalman ','error X position by H infinity')

figure(6)
plot(t,y_original(2,:)-est_K(2,:),'b')
hold on
plot(t,y_original(2,:)-est_H(2,:),'r--')
hold off
legend('error Y position by Kalman ','error Y position by H infinity')
