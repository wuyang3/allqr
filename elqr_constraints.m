clear;clc;

syms l0 l1 l2 l3 l4 l5 l6 l7 l8 l9
syms m0 m1 m2 m3 m4 m5 m6 m7 m8 m9
syms k1 k2
syms g0 g1 g2 g3 g4 g5 g6 g7 g8 g9
syms x0 x1 x2 x3 x4 x5 u0 u1


lambda = [l0;l1;l2;l3;l4;l5;l6;l7;];%l8;l9
Iu = diag([m0 m1 m2 m3 m4 m5 m6 m7]);% m8 m9
g = [g0;g1;g2;g3;g4;g5;g6;g7;];%g8;g9
s = [x0;x1;x2;x3;x4;x5;u0;u1];

delta_g = [0 0 0  0   1   0  0  0;
           0 0 0  0   -1  0  0  0;
           0 0 0  0   0   1  0  0;
           0 0 0  0   0  -1  0  0;
           0 0 0  0   0   0  1  0;
           0 0 0  0   0   0 -1  0;
           0 0 0  0   0   0  0  1;
           0 0 0  0   0   0  0 -1;];
           %0 0 0  k1  k2  0  0  0;
           %0 0 0  -k1 -k2 0  0  0];


%transpose(delta_g) * Iu;


transpose(delta_g) *  Iu * delta_g
transpose(delta_g) *  Iu * delta_g * s;
transpose(delta_g) * lambda;
transpose(delta_g) * Iu * g;

transpose(delta_g) * lambda + transpose(delta_g) * Iu * g - transpose(delta_g) *  Iu * delta_g * s

transpose(delta_g())

%% only state constraints.
lambda_s = lambda(1:4, :);
Iu_s = Iu(1:4, 1:4);
g_s = g(1:4, :);
s_s = s(1:4);
delta_g_s = delta_g(1:4, :);


transpose(delta_g_s) *  Iu_s * delta_g_s
transpose(delta_g_s) * lambda_s + transpose(delta_g_s) * Iu_s * g_s - transpose(delta_g_s) *  Iu_s * delta_g_s * s