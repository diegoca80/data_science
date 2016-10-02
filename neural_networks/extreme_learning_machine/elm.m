% ======== Extreme Learning Machine ========
% This script train the dataset to evaluate test data using ELM and cross-validation 
% plotting the results.
% 
% There are three main steps to ELM process:
%   1. Consider random input parameters of weights and bias
%   2. Use tanh function to adjust for each hidden neurons
%   3. Calculates best output weights based on MSE of each training loop always
%   updating the parameters using cross-validation
%
% Once the ELM has been finished this script performs the following:
%   1. Generates a plot showing the training accuracy for all points
%   2. Generates a plot for testing proccess
%   3. Keep track of MSE for each iteration of cross-validation

% Author: Diego Alves     Date: 2016/09/18 22:00:00 
clear
function [w_input,w_output,bias,p,eqm,yi] = train(neurons,p,X,Y)
    % random input weights between input layer and hidden layer
    % 0 < rand() < 1
    % 0 < 2* rand() < 2
    % -1 < 2* rand() - 1 < 1 , so with different signal weights to balance
    w_input = rand(neurons,size(X,2))*2-1;
    % temporary matrix H with input weights
    tempH = w_input * X';
    % random of hidden bias layer column
    bias = rand(neurons,1);
    %increment to adjust matrix bias
    inc = ones(1,size(X,1));
    biasMatrix = bias * inc;
    % wint*X + bint
    tempH += biasMatrix;
    %apply tanh function with p to adjust curve
    H = (exp(p*tempH) - exp(-p*tempH)) ./ (exp(p*tempH) + exp(-p*tempH));
    % uncomment to use sigmoid function
    %H = 1 ./ (1 + exp(-tempH));
    w_output = pinv(H') * Y;
    yi = (H' * w_output);
    % Calculating EQM
    errors = bsxfun(@minus, Y, yi);
    eqm = mean(errors .^ 2);
end

function plotTrain(X,Y,yi)
    figure;
    hold on;
    RGB = hsv(2);
    scatter3(X(:,1),X(:,2),Y,[],RGB(1,:));
    scatter3(X(:,1),X(:,2),yi,[],RGB(2,:));
end

function [eqm_test] = test(X_test,Y_test,w_input,w_output,bias,p)
    tempH_test = w_input * X_test';
    inc = ones(1,size(X_test,1)); 
    biasMatrix = bias * inc;
    tempH_test+= biasMatrix
    H_test = (exp(p*tempH_test) - exp(-p*tempH_test)) ./ (exp(p*tempH_test) + exp(-p*tempH_test));
    yi_test = (H_test' * w_output);
    % Calculating EQM
    errors = bsxfun(@minus, Y_test, yi_test);
    eqm_test = mean(errors .^ 2);
    plotTest(X_test,Y_test,yi_test)
end

function plotTest(X_test,Y_test,yi_test)
    figure(2);
    hold on;
    RGB = hsv(2);
    scatter3(X_test(:,1),X_test(:,2),Y_test,[],RGB(1,:));
    scatter3(X_test(:,1),X_test(:,2),yi_test,[],RGB(2,:));
end

function [best_eqm,best_winput,best_woutput,best_bias,best_p,best_neurons,best_yi] = cross_validation(X,Y)
    best_eqm = 1
    for i = 1:20
      for j = 1:10
            % neurons
            neurons = i
            % p of curve tanh
            p = j
            % training
            [w_input,w_output,bias,p,eqm,yi] = train(neurons,p,X,Y)
            if(eqm < best_eqm)
                best_eqm = eqm
                best_winput = w_input
                best_woutput = w_output
                best_bias = bias
                best_p = p
                best_neurons = neurons
                best_yi = yi
            end
      end
    end
end

% input samples
load("dados2.mat")
% training data
X = ponto(:,1:2);
% labels
Y = ponto(:,3);
[best_eqm,best_winput,best_woutput,best_bias,best_p,best_neurons,best_yi] = cross_validation(X,Y)
% plotTrain
plotTrain(X,Y,best_yi)

% test samples
load("dados3.mat")
% training data
X_test = ponto(:,1:2);
% labels
Y_test = ponto(:,3);
% test
[eqm_test] = test(X_test,Y_test,best_winput,best_woutput,best_bias,best_p)

