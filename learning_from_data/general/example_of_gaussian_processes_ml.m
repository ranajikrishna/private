

clear all;
close all;

% Choose a kernel (covariance function).
kernel = 6;
switch kernel
    case 1; k = @(x,y) 1*x'*y;   % Linear.
    case 2; k = @(x,y) 1*min(x,y);   % Brownian motion.
    case 3; k = @(x,y) exp(-100*(x-y)'*(x-y));  % Squared exponential.
    case 4; k = @(x,y) exp(-1*sqrt((x-y)'*(x-y)));  % Ornstein-Uhlenbeck.
    case 5; k = @(x,y) exp(-1*sin(5*pi*(x-y))^2);   % Periodic.
    case 6; k = @(x,y) exp(-100*min(abs(x-y), abs(x+y)).^2); % Symmetric.
end

% Choose points at which to sample.
x = (0:.005:1);     % Change to (-1:.005:1) for kernel = 6.
n = length(x)

% Construct the covariance matrix.
C = zeros (n,n);
for i = 1:n
    for j = 1:n
        C(i,j) = k(x(i),x(j));
    end
end

% Sample from the Gaussian process at the points.
u = randn(n,1);         % Sample u ~ N(0,I).
[A,S,B] = svd(C);       % factor C = ASB'
z = A*sqrt(S)*u;        % z = A.S^.5 u ~ N(0,C)

% Plot
figure(2); hold on; clf
plot(x,z,'.-')
axis([0,1,-2,2])        % Comment out for kernel = 6.