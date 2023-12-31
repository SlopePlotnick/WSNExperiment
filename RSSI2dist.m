function [y1] = RSSI2dist(x1)
%MYNEURALNETWORKFUNCTION neural network simulation function.
%h
% Auto-generated by MATLAB, 29-Nov-2023 21:09:25.
%
% [y1] = myNeuralNetworkFunction(x1) takes these arguments:
%   x = Qx1 matrix, input #1
% and returns:
%   y = Qx1 matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = -90.2;
x1_step1.gain = 0.0302114803625378;
x1_step1.ymin = -1;

% Layer 1
b1 = [14.120089187446925649;10.888639287089542762;-7.8480360968994169468;-3.5520573372738595275;1.3403290314889924062;2.1262202384352120177;-4.3299203373172234777;7.4871025695126105504;0.92333784950638431166;13.475130758494321626];
IW1_1 = [-13.879912820095459125;-13.999376392164089467;13.967362676253268106;14.273149352404807644;-11.621265450024949928;11.123848128520954504;-15.938375012095491456;13.468816079926817508;8.4217173587576024119;16.969734033506966853];

% Layer 2
b2 = -0.35192744948977472408;
LW2_1 = [0.15140074023794458657 -0.051495753842966958402 -0.016648175542362253498 -0.058707158145138209349 0.034150112297480245127 0.03056634774217824313 0.07902649493344313103 -0.15099411911436261269 -0.13699261215004299164 -0.28910070677291904717];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = 0.101155177200631;
y1_step1.xoffset = 0.13;

% ===== SIMULATION ========

% Dimensions
Q = size(x1,1); % samples

% Input 1
x1 = x1';
xp1 = mapminmax_apply(x1,x1_step1);

% Layer 1
a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*xp1);

% Layer 2
a2 = repmat(b2,1,Q) + LW2_1*a1;

% Output 1
y1 = mapminmax_reverse(a2,y1_step1);
y1 = y1';
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
y = bsxfun(@minus,x,settings.xoffset);
y = bsxfun(@times,y,settings.gain);
y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
x = bsxfun(@minus,y,settings.ymin);
x = bsxfun(@rdivide,x,settings.gain);
x = bsxfun(@plus,x,settings.xoffset);
end
