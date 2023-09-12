
addpath('../tensorlab/');

r = 15; % rank
multirank = [40 40 1]; % multilinear rank-(Lr,Lr,1)
max_iter = 30;
display = true;
tol_fun = 10e-5;
tol_x = 10e-5;
T = rand(50, 51, 12);
[result, output] = myBTD(r, multirank, max_iter, tol_fun, tol_x, display, T);