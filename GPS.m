%% declare variables %%

%initialize vectors to hold error metrics

%Steepest descent variables
e_b_SD = [];    %vector to hold error in b
e_S_SD = [];    %vector to hold error in S
loss_SD = [];   %vector to hold loss
k_SD = [];     %vector to hold iteration index

%Gauss-Newton variables
e_b_GN = [];    %vector to hold error in b
e_S_GN = [];    %vector to hold error in S
loss_GN = [];   %vector to hold loss
k_GN = [];     %vector to hold iteration index

%earth radius in meters
ER = 6370E3;

%true S and b
S0 = [1 0 0]';
b = 2.354788068E-3;

%function for calculating error in b
eb = @(X) abs(b - X(4));

%satellite coordinates
global S1; global S2; global S3; global S4;
S1 = [3.5852 2.07 0]';
S2 = [2.9274 2.9274 0]';
S3 = [2.6612 0 3.1712]';
S4 = [1.4159 0 3.8904]';

%initial estimates
curS = [0.93310 0.25 0.258819]';
curB = 0;
X0 = [curS;curB];

%calculate y vector
global y;
y = [R(S0,S1)+b;R(S0,S2)+b;R(S0,S3)+b;R(S0,S4)+b];


%% find S0 through steepest descent %%

a = 0.1;   %learning rate
curX = X0;
oldX = curX;    %first guess
curX = curX + a*(jacob(curX))'*(y-h(curX)); %second guess

%init iteration index and find initial error metrics
k = 1;          
k_SD(k) = k;     
e_b_SD(k) = eb(curX);    
e_S_SD(k) = R(S0,curX(1:3));    
loss_SD(k) = loss(curX);

%iterate until termination condition met
while ( (curX-oldX)'*(curX-oldX) )^(0.5) > 1E-11 && (k < 1E6)

    %store old estimate and calculate new estimate
    oldX = curX;
    curX = curX + a*(jacob(curX))'*(y-h(curX));
    
    %increment iteration index and update error metrics
    k = k + 1;          
    k_SD(k) = k;     
    e_b_SD(k) = eb(curX);    
    e_S_SD(k) = R(S0,curX(1:3));    
    loss_SD(k) = loss(curX);   
end

%store final estimate of X from Steepest Descent method
X_SD = curX;

%% find S0 through gauss-newton %%

a = 1;   %learning rate
curX = X0;
oldX = curX;    %first guess
H = jacob(curX);
curX = curX + a*inv(H' * H) * H' *(y-h(curX));  %second guess

%init iteration index and find initial error metrics
k = 1;          
k_GN(k) = k;    
e_b_GN(k) = eb(curX);  
e_S_GN(k) = R(S0,curX(1:3));  
loss_GN(k) = loss(curX);   

%iterate until termination condition met
while ( (curX-oldX)'*(curX-oldX) )^(0.5) > 1E-11 && (k < 1E6)

    H = jacob(curX);    %calc Jacobian

    %store old estimate and calculate new estimate
    oldX = curX;
    curX = curX + a*inv(H' * H) * H' *(y-h(curX));
    
    %increment iteration index and update error metrics
    k = k+1;
    k_GN(k) = k;
    e_b_GN(k) = eb(curX);
    e_S_GN(k) = R(S0,curX(1:3));
    loss_GN(k) = loss(curX);
end 

%store final estimate of X from Gauss-Newton method
X_GN = curX;

%% begin plots %%

%convert from earth radii to meters
e_b_SD = e_b_SD * ER;   
e_S_SD = e_S_SD * ER;
loss_SD = loss_SD*ER;

e_b_GN = e_b_GN * ER;   
e_S_GN = e_S_GN * ER;
loss_GN = loss_GN*ER;

%plot error metrics for Steepest Descent

%plot position and clock error
figure;
hold on;
plot(k_SD,e_b_SD);
plot(k_SD,e_S_SD,'--');
%axis([1 7E8 0 2.3E6])
title('Error metrics for Steepest Descent')
xlabel('iteration')
ylabel('error metric (m)')
legend('clock bias estimate error','receiver position estimate error')

%plot loss
figure;
plot(k_SD,loss_SD);
axis([1 2E3 0 5E4])
title('Loss for Steepest Descent')
xlabel('iteration')
ylabel('loss (m)')

%plot error metrics for Gauss-Newton

%plot position and clock error

figure;
hold on;
plot(k_GN,e_b_GN);
plot(k_GN,e_S_GN,'--');
%axis([1 5 0 9E4])
title('Error metrics for Gauss-Newton')
xlabel('iteration')
ylabel('error metric (m)')
legend('clock bias estimate error','receiver position estimate error')

%plot loss
figure;
plot(k_GN,loss_GN);
title('Loss for Gauss-Newton')
xlabel('iteration')
ylabel('loss (m)')

