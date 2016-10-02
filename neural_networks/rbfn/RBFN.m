% ======== Radial Basis Function Function Aproximation ========
% This script trains the dataset in order to aproximate his behavior considering
% the minimum mean square error. The use of k-means to find the initial best centroids
% of the dataset is required since we need that in RBF code. As result a plot of the 
% original dataset compared to the best plane found is showed.
% 
% There are three main steps to RBFN process:
%   1. Calculate initial centroids using k-means clustering. This step is provided with such differents
%   ways to start but here is considered the random centers and Euclidean distance
%   2. Assign number centroids found by k-means as number of neurons and train in order to find the
%   output weights
%   3. Consider cross validation to find the best parameters related to k-means and RBFN 
%
% Once the RBFN has been finished this script performs the following:
%   1. Generates a plot showing the clusters based on k-means output
%   2. Plot the function that best aproximates the training data
%   3. Show the best MSE found based on this training and evaluation

% Author: Diego Alves     Date: 2016/09/18 22:00:00 
clear
% input samples
load("dados_map.mat")

function clusters = findClosestCentroids(X,centroids, k)
    % In k-means clustering, data points are assigned to a cluster based on the
    % Euclidean distance between the data point and the cluster centroids.

    % Set 'm' to the number of data points.
    m = size(X, 1);
    % 'clusters' will have the cluster numbers for each training data.
    clusters = zeros(m, 1);

    % Create a matrix to hold the distances between each data point and each cluster center.
    distances = zeros(m, k);
    % For each cluster get Euclidean distance
    for i = 1 : k
        % Subtract centroid i from all data points.
        diffs = bsxfun(@minus, X, centroids(i, :));
        % Square the differences.
        sqrdDiffs = diffs .^ 2;
        % Take the sum of the squared differences.
        distances(:, i) = sum(sqrdDiffs, 2);
    end
    % Find the minimum distance value, also set the index of cluster
    [minVals clusters] = min(distances, [], 2);
end


function centroids = computeCentroids(X, prev_centroids, clusters, k)
    % X contains 'm' samples with 'n' dimensions each.
    [m n] = size(X);
    centroids = zeros(k, n);
    % For each centroid...
    for (i = 1 : k)
        % If no points are assigned to the centroid, don't move it.
        if (~any(clusters == i))
            centroids(i, :) = prev_centroids(i, :);
        % Otherwise, compute the cluster's new centroid.
        else
            % Select the data points assigned to centroid k.
            points = X((clusters == i), :);

            % Compute the new centroid as the mean of the data points.
            centroids(i, :) = mean(points);    
        end
    end
end

function [betas,w,centroids,clusters,H] = train(k,max_iters,X,Y,sigma)
    % ----------------------------------------k-means-----------------------------------------------

    % initialize with zeros
    initial_centroids = zeros(k, size(X,2));
    % Creating random indices of training data
    randidx = randperm(size(X));
    % Take the first k examples as centroids
    initial_centroids = X(randidx(1:k), :);

    % assign to final centroids
    centroids = initial_centroids;
    % store the previous for compare on each epoch
    previous_centroids = centroids;

    % Run K-Means
    for (i = 1 : max_iters)
        % Output progress
        %fprintf('K-Means iteration %d / %d...\n', i, max_iters);
        %fflush(stdout);
        
        % For each example in X, assign it to the closest centroid
        clusters = findClosestCentroids(X, centroids, k);
            
        % Given the clusters, compute new centroids
        centroids = computeCentroids(X, centroids, clusters, k);
        
        % Check for convergence. If the centroids haven't changed since
        % last iteration, we've converged.
        if (previous_centroids == centroids)
            %fprintf("Stopping after %d iterations.\n", i);
            %fflush(stdout);
            break;
        end
        % Update the 'previous' centroids.
        previous_centroids = centroids;
    end

    % ----------------------------------------RBFN-----------------------------------------------
    % Calculate the beta value to use for all neurons.
    beta = 1 ./ (2 .* sigma.^2);
    % Used same beta coefficient for all neurons. num neurons = num centroids
    betas = ones(size(centroids, 1), 1) * beta;
    neurons = k;
    % The H matrix stores the RBF neuron activation values for each training 
    % and each neuron
    H = zeros(size(X,1), neurons);

    % Calculate matrix activation H
    for i = 1 : size(X,1)
       % Get the activation for all RBF neurons for this input.
       % Subtract the input from all of the centers.
       diffs = bsxfun(@minus, centroids, X(i,:));
       % Calculate the sum of squared distances for each diff(row) (x-c)^2
       % sum(matrix,2) calculates sum in each row
       sqrdDists = sum(diffs .^ 2, 2);
       % Apply the beta coefficient and take the negative exponent.
       fx = exp(-betas .* sqrdDists);
       % Store the activation values for training example 'i'.
       H(i, :) = fx';
    end
    % Normalize H matrix based on each row that is the sum of neurons activators
    H = bsxfun(@rdivide, H, sum(H, 2));
    % Add a column of 1s for the bias term.
    H = [ones(size(X,1), 1), H];
    % Calculate all of the output weights
    w = pinv(H' * H) * H' * Y;   


    % Calculating mse
    %yi = zeros(size(X,1),1);
    %for i = 1:size(X,1)
    %   yi(i) =  sum(w'.* H(i,:));
    %end
    %errors = bsxfun(@minus, Y, yi);
    %mse = mean(errors .^ 2);
    % end function
end

function [mse , yi] = validation(k, centroids, betas, w, X_validation,Y_validation)
    H = zeros(size(X_validation,1), k);

    % Calculate matrix activation H
    for i = 1 : size(X_validation,1)
       % Get the activation for all RBF neurons for this input.
       % Subtract the input from all of the centers.
       diffs = bsxfun(@minus, centroids, X_validation(i,:));
       % Calculate the sum of squared distances for each diff(row) (x-c)^2
       % sum(matrix,2) calculates sum in each row
       sqrdDists = sum(diffs .^ 2, 2);
       % Apply the beta coefficient and take the negative exponent.
       fx = exp(-betas .* sqrdDists);
       % Store the activation values for training example 'i'.
       H(i, :) = fx';
    end
    % Normalize H matrix based on each row that is the sum of neurons activators
    H = bsxfun(@rdivide, H, sum(H, 2));
    % Add a column of 1s for the bias term.
    H = [ones(size(X_validation,1), 1), H];
    
    % Calculating mse
    yi = zeros(size(X_validation,1),1);
    for i = 1:size(X_validation,1)
       yi(i) =  sum(w'.* H(i,:));
    end
    errors = bsxfun(@minus, Y_validation, yi);
    mse = mean(errors .^ 2);
end

function plotClusters(k,X,Y,clusters)
    % Generate array of colors
    RGB = hsv(k);
    figure
    hold on
    % For each cluster analyse training data finding it by clusters array and plot them
    for i = 1:k
      points = X((clusters == i), :);
      labels = Y((clusters == i), :);
      scatter3(points(:,1),points(:,2),labels,[],RGB(i,:));
    end
end

function plotFunction(X,Y,yi)
    %Plot output function
    figure(2);
    hold on;
    RGB = hsv(2);
    scatter3(X(:,1),X(:,2),Y,[],RGB(1,:));
    scatter3(X(:,1),X(:,2),yi,[],RGB(2,:));
end


function [best_mse,best_k,best_sigma] = crossValidation(X,Y)
    % Fixed max iterations
    max_iters = 100;
    best_mse = 1;
    best_k = 0;
    best_sigma = 0;
    for i = 1:100
      for j = 1:15
        k = i;
        sigma = j;
        X_train = X(1:round((size(X,1))*0.9),:);
        Y_train = Y(1:round((size(Y,1)*0.9)),:);
        X_validation = X(1:round((size(X,1)*0.1)),:);
        Y_validation = Y(1:round((size(Y,1)*0.1)),:);
        [betas,w,centroids,clusters,H] = train(k,max_iters,X_train,Y_train,sigma);
        [mse , yi] = validation(k,centroids, betas, w, X_validation,Y_validation);
        if mse < best_mse
            fprintf('mse = %d / k = %d...\n', mse, k);
            fflush(stdout);
            best_mse = mse;
            best_k = k;
            best_sigma = sigma;
        end
      end
    end
end

% training data
X = dados_rbf(:,1:2);
% labels
Y = dados_rbf(:,3);
% number of centroids desired
k = 6;
% k-means max iterations before stop
max_iters = 100;
% Set the sigma for gaussian curve
sigma = 10;
%[betas,w,centroids,clusters] = train(k,max_iters,X,Y,sigma);

[best_mse,best_k,best_sigma] = crossValidation(X,Y)
[betas,w,centroids,clusters,H] = train(best_k,max_iters,X,Y,best_sigma);

% Calculating final mse
yi = zeros(size(X,1),1);
for i = 1:size(X,1)
   yi(i) =  sum(w'.* H(i,:));
end
errors = bsxfun(@minus, Y, yi);
mse = mean(errors .^ 2);

plotClusters(best_k,X,Y,clusters);
plotFunction(X,Y,yi);

