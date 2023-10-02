
addpath('../tensorlab/');

r = 5; % rank
l_r = 40; % multilinear rank-(Lr,Lr,1)
max_iter = 30;
display = true;
tol_fun = 10e-6;
tol_x = 10e-6;
T = rand(50, 51, 12);
[result, output] = myBTD2(r, l_r, max_iter, tol_fun, tol_x, display, T);