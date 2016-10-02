% ======== Radial Basis Function Function Aproximation no Cross Validation========
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
%
% Once the RBFN has been finished this script performs the following:
%   1. Generates a plot showing the clusters based on k-means output
%   2. Plot the function that best aproximates the training data
%   3. Show the best MSE found based on this training

% Author: Diego Alves     Date: 2016/09/18 22:00:00 
clear
% input samples
load("dados_map.mat")
%training data
X = dados_rbf(:,1:2)
%labels
Y = dados_rbf(:,3)
% number of centroids desired
k = 6
%max iterations before stop
max_iters = 100

%initialize with zeros
initial_centroids = zeros(k, size(X,2));
% Creating random indices of training data
randidx = randperm(size(X));
% Take the first k examples as centroids
initial_centroids = X(randidx(1:k), :);

%assign to final centroids
centroids = initial_centroids;
%store the previous for compare on each epoch
previous_centroids = centroids;


function clusters = findClosestCentroids(X,centroids, k)
% In k-means clustering, data points are assigned to a cluster based on the
% Euclidean distance between the data point and the cluster centroids.

% Returns a column vector containing the index of the closest centroid (a value
% between 1 - k) for each corresponding data point in X.

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



% Run K-Means
for (i = 1 : max_iters)
    % Output progress
    fprintf('K-Means iteration %d / %d...\n', i, max_iters);
    fflush(stdout);
    
    % For each example in X, assign it to the closest centroid
    clusters = findClosestCentroids(X, centroids, k);
        
    % Given the clusters, compute new centroids
    centroids = computeCentroids(X, centroids, clusters, k);
    
    % Check for convergence. If the centroids haven't changed since
    % last iteration, we've converged.
    if (previous_centroids == centroids)
        fprintf("Stopping after %d iterations.\n", i);
        fflush(stdout);
        break;
    end
    %fprintf("Previous %d \n", previous_centroids);
    %fprintf("Actual %d \n", centroids);
    %fflush(stdout);
    % Update the 'previous' centroids.
    previous_centroids = centroids;
end

% Generate array of colors
RGB = hsv(k);
%figure
%hold on
% For each cluster analyse training data finding it by clusters array and plot them
%for i = 1:k
%  points = X((clusters == i), :);
%  labels = Y((clusters == i), :);
%  scatter3(points(:,1),points(:,2),labels,[],RGB(i,:));
%end


% Calculate the beta value to use for all neurons.
% Set the sigmas
sigma = 10;
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

%Plot output function
yi = zeros(size(X,1),1);
figure(2)
hold on
% Generate colors
RGB = hsv(2);
scatter3(X(:,1),X(:,2),Y,[],RGB(1,:));
for i = 1:size(X,1)
   yi(i) =  sum(w'.* H(i,:));
   scatter3(X(i,1),X(i,2),yi(i),[],RGB(2,:));
end


% Calculating mse
errors = bsxfun(@minus, Y, yi);
mse = mean(errors .^ 2);
