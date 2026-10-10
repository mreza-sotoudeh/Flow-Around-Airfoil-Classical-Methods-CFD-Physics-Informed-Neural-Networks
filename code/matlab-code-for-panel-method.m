clear; close all; clc 
 
%% ================= PARAMETERS ================= 
Uinf = 1;            % Free-stream velocity 
alpha = 0;           % Angle of attack [deg] 
c = 1;               % Chord length 
 
thickness_list = [0.06 0.12 0.18 0.24 0.30];  % t/c 
Npanel_list    = [40 80 160 320];            % Panels 
for independence study 
 
%% ================= NACA symmetric thickness function 
================= 
thickness_fun = @(x,t) 5*t*c*( ... 
    0.2969*sqrt(x/c) - 0.1260*(x/c) ... 
  - 0.3516*(x/c).^2 + 0.2843*(x/c).^3 ... 
  - 0.1015*(x/c).^4 ); 
 
%% ================= Effect of Thickness on Pressure 
Drag ================= 
Cd_thickness = zeros(size(thickness_list)); 
 
for it = 1:length(thickness_list) 
    t = thickness_list(it); 
    N = 160;  % Fixed number of panels for thickness 
study 
 
    % Geometry (cosine spacing) 
    beta = linspace(0,pi,N+1); 
    x = c/2*(1-cos(beta)); 
    yt = thickness_fun(x,t); 
 
    % Upper and lower surface 
    xu = x;   yu = yt; 
    xl = fliplr(x); yl = -fliplr(yt); 
 
 
    % Combine panels 
    X = [xu xl(2:end-1)]; 
    Y = [yu yl(2:end-1)]; 
    np = length(X)-1; 
 
    % Panel geometry 
    xmid = zeros(1,np); ymid = zeros(1,np); 
    S = zeros(1,np); theta = zeros(1,np); 
    nx = zeros(1,np); ny = zeros(1,np); 
 
    for i = 1:np 
        xmid(i) = 0.5*(X(i)+X(i+1)); 
        ymid(i) = 0.5*(Y(i)+Y(i+1)); 
        dx = X(i+1)-X(i); 
        dy = Y(i+1)-Y(i); 
        S(i) = sqrt(dx^2+dy^2); 
        theta(i) = atan2(dy,dx); 
        nx(i) = sin(theta(i)); 
        ny(i) = -cos(theta(i)); 
    end 
 
    % Influence matrix 
    A = zeros(np); 
    RHS = zeros(np,1); 
 
    for i = 1:np 
        for j = 1:np 
            if i == j 
                A(i,j) = 0.5; 
            else 
                [u,v] = source_panel_velocity(xmid(i), 
ymid(i), X(j), Y(j), X(j+1), Y(j+1)); 
                A(i,j) = u*nx(i) + v*ny(i); 
            end 
        end 
        RHS(i) = -Uinf*nx(i); 
    end 
 
 
    % Solve source strengths 
    sigma = A\RHS; 
 
    % Pressure drag 
    Fx = 0; 
    for i = 1:np 
        Fx = Fx - sigma(i)^2 * nx(i)*S(i); 
    end 
    Cd_thickness(it) = Fx/c; 
end 
 
%% ================= Panel Independence Study 
================= 
Cd_panel = zeros(length(Npanel_list),1); 
t = 0.12;  % Representative thickness 
 
for k = 1:length(Npanel_list) 
    N = Npanel_list(k); 
 
    beta = linspace(0,pi,N+1); 
    x = c/2*(1-cos(beta)); 
    yt = thickness_fun(x,t); 
 
    xu = x; yu = yt; 
    xl = fliplr(x); yl = -fliplr(yt); 
 
    X = [xu xl(2:end-1)]; 
    Y = [yu yl(2:end-1)]; 
    np = length(X)-1; 
 
    xmid = zeros(1,np); ymid = zeros(1,np); 
    S = zeros(1,np); theta = zeros(1,np); 
    nx = zeros(1,np); ny = zeros(1,np); 
 
    for i = 1:np 
        xmid(i) = 0.5*(X(i)+X(i+1)); 
        ymid(i) = 0.5*(Y(i)+Y(i+1)); 
        dx = X(i+1)-X(i); 
        dy = Y(i+1)-Y(i); 

        S(i) = sqrt(dx^2+dy^2); 
        theta(i) = atan2(dy,dx); 
        nx(i) = sin(theta(i)); 
        ny(i) = -cos(theta(i)); 
    end 
 
    A = zeros(np); 
    RHS = zeros(np,1); 
 
    for i = 1:np 
        for j = 1:np 
            if i == j 
                A(i,j) = 0.5; 
            else 
                [u,v] = source_panel_velocity(xmid(i), 
ymid(i), X(j), Y(j), X(j+1), Y(j+1)); 
                A(i,j) = u*nx(i) + v*ny(i); 
            end 
        end 
        RHS(i) = -Uinf*nx(i); 
    end 
 
    sigma = A\RHS; 
 
    Fx = 0; 
    for i = 1:np 
        Fx = Fx - sigma(i)^2 * nx(i)*S(i); 
    end 
    Cd_panel(k) = Fx/c; 
end 
 
%% ================= Flow Field for Maximum Thickness 
(Smooth ψ) ================= 
t = max(thickness_list);  %  ترین ایرفویل ضخیم 
N = 160; 
 
beta = linspace(0,pi,N+1); 
x = c/2*(1-cos(beta)); 
yt = thickness_fun(x,t); 

 
xu = x; yu = yt; 
xl = fliplr(x); yl = -fliplr(yt); 
 
X = [xu xl(2:end-1)]; 
Y = [yu yl(2:end-1)]; 
np = length(X)-1; 
 
xmid = 0.5*(X(1:end-1)+X(2:end)); 
ymid = 0.5*(Y(1:end-1)+Y(2:end)); 
S = sqrt(diff(X).^2 + diff(Y).^2); 
theta = atan2(diff(Y),diff(X)); 
nx = sin(theta); ny = -cos(theta); 
 
A = zeros(np); 
RHS = zeros(np,1); 
for i=1:np 
    for j=1:np 
        if i==j 
            A(i,j)=0.5; 
        else 
            [u,v] = source_panel_velocity(xmid(i), 
ymid(i), X(j), Y(j), X(j+1), Y(j+1)); 
            A(i,j) = u*nx(i)+v*ny(i); 
        end 
    end 
    RHS(i) = -Uinf*nx(i); 
end 
sigma = A\RHS; 
  شبکه جریان با تراکم بیشتر  
%
nxg = 500; nyg = 500;  % افزایش resolution 
[xg, yg] = meshgrid(linspace(-0.5,1.5,nxg), linspace(
0.6,0.6,nyg)); 
 
phi = Uinf*xg;  
psi = Uinf*yg; 
 
eps = 1e-5; %  فاصله کوچک برای جلوگیری از برخورد grid  با سطح 

for j = 1:np 
    [phi_s, psi_s] = source_panel_potential(xg, yg, 
X(j), Y(j), X(j+1), Y(j+1)); 
    phi = phi + sigma(j)*phi_s; 
    psi = psi + sigma(j)*psi_s; 
end 
 
in = inpolygon(xg+eps, yg+eps, X, Y); 
phi(in) = NaN; 
psi(in) = NaN; 
psi = psi - nanmean(psi(:)); 
 
%% ================= Plots ================= 
figure 
plot(thickness_list,abs(Cd_thickness),'o
','LineWidth',2) 
xlabel('t/c'); ylabel('C_d') 
title('Effect of Thickness on Pressure Drag') 
 
figure 
plot(Npanel_list,abs(Cd_panel),'s-','LineWidth',2) 
xlabel('Number of panels'); ylabel('C_d') 
title('Panel Independence Study') 
 
figure 
contour(xg, yg, psi, 60) 
hold on 
plot(X,Y,'k','LineWidth',2) 
axis equal 
xlabel('x'); ylabel('y') 
title('Streamlines \psi (Smooth)') 
 
figure 
contour(xg, yg, phi, 60) 
hold on 
plot(X,Y,'k','LineWidth',2) 
axis equal 
xlabel('x'); ylabel('y') 
title('Equipotential Lines \phi (Smooth)') 

 
%% ================= Functions ================= 
function [u,v] = 
source_panel_velocity(xp,yp,x1,y1,x2,y2) 
dx = x2-x1; dy = y2-y1; 
L = sqrt(dx^2+dy^2); 
phi = atan2(dy,dx); 
 
xt = (xp-x1)*cos(phi)+(yp-y1)*sin(phi); 
yt = -(xp-x1)*sin(phi)+(yp-y1)*cos(phi); 
 
u = (1/(2*pi))*log(((xt-L).^2+yt.^2)./(xt.^2+yt.^2)); 
v = (1/pi)*(atan2(yt,xt-L)-atan2(yt,xt)); 
 
u_rot = u*cos(phi) - v*sin(phi); 
v_rot = u*sin(phi) + v*cos(phi); 
 
u = u_rot; v = v_rot; 
end 
 
function [phi,psi] = 
source_panel_potential(xp,yp,x1,y1,x2,y2) 
dx = x2-x1; dy = y2-y1; 
L = sqrt(dx^2+dy^2); 
phi_p = atan2(dy,dx); 
 
xt = (xp-x1)*cos(phi_p)+(yp-y1)*sin(phi_p); 
yt = -(xp-x1)*sin(phi_p)+(yp-y1)*cos(phi_p); 
 
phi = (1/(4*pi))*log(((xt-L).^2+yt.^2)./(xt.^2+yt.^2)); 
psi = (1/(2*pi))*(atan2(yt,xt-L)-atan2(yt,xt)); 
end 
%% ================= Cp Distribution ================= 
Cp = zeros(1,np); 
 
for i = 1:np 
    ut = Uinf; vt = 0; 
    for j = 1:np 
 
        [u,v] = source_panel_velocity(xmid(i), ymid(i), 
X(j), Y(j), X(j+1), Y(j+1)); 
        ut = ut + sigma(j)*u; 
        vt = vt + sigma(j)*v; 
    end 
    Vt = ut*cos(theta(i)) + vt*sin(theta(i)); 
    Cp(i) = 1 - (Vt/Uinf)^2; 
end 
  نقاط برای رسم  
%
X_surface = X(1:end-1);  %  مراکز پنل 
figure 
plot(X_surface, Cp, 'bo-', 'LineWidth', 1.5) 
set(gca,'YDir','reverse')  % Cp convention:  پایین تر فشار مثبت 
xlabel('x/c') 
ylabel('C_p') 
title('Pressure Coefficient Distribution on NACA 
Airfoil') 
grid on 
