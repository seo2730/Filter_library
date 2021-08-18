function [A,B]=ABjacob_real(xhat,u,dt)
%
%
A=zeros(3,3);
d=0.3;

x=xhat(1);
y=xhat(2);
theta=xhat(3);

v=u(1);
w=u(2);

A(1,1)=1;
A(1,2)=0;
A(1,3)=-sin(theta)*v-dt*w*cos(theta);

A(2,1)=0;
A(2,2)=1;
A(2,3)=cos(theta)*v-dt*w*sin(theta);

A(3,1)=0;
A(3,2)=0;
A(3,3)=1;

A=eye(3);

B(1,1)=cos(theta);
B(1,2)=-d*sin(theta);

B(2,1)=sin(theta);
B(2,2)=d*cos(theta);

B(3,1)=0;
B(3,2)=1;

B=B*dt;
end