function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1:

% getting the result using the current theta
% add 1 to X
X = [ones(m, 1), X];
temp = X * Theta1.';
temp = sigmoid(temp);
% add 1 to temp
temp = [ones(m, 1), temp];
result = temp * Theta2.';
result = sigmoid(result);

% find the cost

for i = 1:m
    tempVector = result(i,:);
    % create temporary vector for representing the real result
    temp = zeros(num_labels, 1);
    temp(y(i)) = 1;
    for j = 1:num_labels
        J = J + (-temp(j)*log(tempVector(j))-(1-temp(j))*log(1-tempVector(j)));
    end;
end;

J = J / m;
    


% regularized

% make copy
tempTheta1 = Theta1;
tempTheta2 = Theta2;

% delete all theta00(which is located at first column)
tempTheta1(:,1) = [];
tempTheta2(:,1) = [];

tempJ = 0;

for i = 1:(input_layer_size * hidden_layer_size)
    tempJ = tempJ + tempTheta1(i)^2;
end

for i = 1:(hidden_layer_size * num_labels)
    tempJ = tempJ + tempTheta2(i)^2;
end

tempJ = tempJ * lambda / (2*m);

J = J + tempJ;



% Part 2 backpropagation algorithm:

z_2 = X * Theta1.';
a_2 = sigmoid(z_2);
a_2 = [ones(m,1), a_2];

z_3 = a_2 * Theta2.';
a_3 = sigmoid(z_3);


for i = 1:m
    label = zeros(num_labels, 1);
    label(y(i)) = 1;
    % getting ith result
    result = a_3(i,:);
    result = result.';
    delta_3 = result - label;
    z2 = z_2(i,:);
    z2 = z2.';
    delta_2 = (Theta2.' * delta_3) .* sigmoidGradient([1;z2]);
    % remove the delta_2_0;
    delta_2 = delta_2(2:end);
    % accumlate the gradient
    a2 = a_2(i,:);
    a1 = X(i,:);
    Theta1_grad = Theta1_grad + delta_2 * a1;
    Theta2_grad = Theta2_grad + delta_3 * a2;
    
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% regularized
Theta1_no_first_column = Theta1;
[row col] = size(Theta1);
% set the first column to zeros
Theta1_no_first_column(:,1) = [zeros(row, 1)];
Theta1_grad = Theta1_grad + (lambda / m) * Theta1_no_first_column; 

Theta2_no_first_column = Theta2;
[row col] = size(Theta2);
Theta2_no_first_column(:,1) = [zeros(row, 1)];
Theta2_grad = Theta2_grad + (lambda / m) * Theta2_no_first_column;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
