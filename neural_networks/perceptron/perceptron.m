% ======== perceptron ========
% This script finds the best plane equation that separates the example dataset, and plots the
% resulting plane to visualize.
% 
% There are three main steps to perceptron process:
%   1. Consider step function to separates labels
%   2. Assign initial values for weights and alfa
%   3. Loop through all epoch until error equals 0 updating
%   errors list to track number of error of each epoch and    
%   weights vector to improve performance
%
% Once the perceptron has been finished this script performs the following:
%   1. Generates a plot showing the dataset
%   2. Draws a plane equation that separates the labels
%   3. Keep track of number of errors per epoch until errors = 0

% Author: Diego Alves     Date: 2016/09/18 22:00:00 
clear
% input samples
load("dados1.mat")

% init weigth vector
w=[0.02 0.02 0.02];
alfa = 0.5;
count = 0;
errors_per_epoch = 0

function result = step(x)
  if x<0
    result = -1
  else
    result = 1
  end
end

figure(2)
hold on
loop = true
while loop
  errors_per_epoch = 0
  count += 1
  for idx = 1:numel(desejado)
    #scalar product dot
    result = dot(w,x(idx,:))
    error = desejado(idx) - step(result)
    if (error!=0)
        errors_per_epoch += 1
    end
    errors(idx) = error
    w+= alfa * error * x(idx,:)
  end
  
  bar(count,errors_per_epoch,0.3,'b');
  xlabel("Epoch");
  ylabel("Errors");
  #check if errors = 0
  if all(errors==0)
      break
  end
end

[a, b] = meshgrid(-5:0.5:5);
z =((-a.*w(1) - (b.*w(2)))/w(3));

#colors in rgb but divided by 255 since Matlab pattern of between 0 and 1
blue = [0,0,255] / 255;
red = [255,0,0] / 255;
figure
hold on;
for idx = 1:numel(desejado)
  if desejado(idx) ==  -1
    #4 parameter is the size and 5 parameter is the color RGB
    scatter3(x(idx,1),x(idx,2),x(idx,3),[],red);
  else
    #4 parameter is the size and 5 parameter is the color RGB
    scatter3(x(idx,1),x(idx,2),x(idx,3),[],blue);
  end
end
surf(a,b,z)
xlabel("X");
ylabel("Y");
zlabel("Z");
