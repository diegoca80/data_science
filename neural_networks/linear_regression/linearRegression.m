% ======== linearRegression ========
% This script finds a line equation that best aproximates based on an example dataset, and plots the
% resulting function.
% 
% There are three main steps to the logistic regression process:
%   1. Consider the equation of MSE (mean square error)
%   2. Substitute the values of dataset on line equation
%   3. The MSE equation has partial derivates related to "a" and "b" equals 0 
% in order to find minimize sum. So, values of "a" and "b" can be found by simple equation system
%
% Once the linear regression has been finished this script performs the following:
%   1. Generates a plot showing the dataset
%   2. Draws a line equation that describes the dataset
%   3. Evaluates the MSE of the dataset.

% Author: Diego Alves     Date: 2016/09/18 22:00:00 

clear
% Dataset
data = [0 0;1 2;2 3;5 8;7 7;9 10];
% prepare data before system equation solve
X = data(:,1)
Y = data(:,2)
N = length(X)
somaX=sum(X)
somaY=sum(Y)
Xquad = X.^2
somaXquad = sum(Xquad)
somaXY= sum(X.*Y)
% system equation to find "a" and "b"
A = [N somaX;somaX somaXquad]
B = [somaY;somaXY]
Z = linsolve(A,B)
b = Z(1)
a = Z(2)
% Calculating MSE
yi = (a*X+b);
errors = bsxfun(@minus, Y, yi);
mse = mean(errors .^ 2);
% Plot points (x,y)
plot(X,Y, 'rx', 'Markersize', 10);
xlabel('X');
ylabel('Y');
hold on;
% Plot line equation found
x = -1:10;
y = a*x + b;
plot(x,y)