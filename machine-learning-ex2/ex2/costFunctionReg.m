function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

for i = 1:m
    tempx = X(i,:);
    tempg = tempx * theta;
    temph = sigmoid(tempg);
    J = J + (-y(i) * log(temph) - (1-y(i))*log(1-temph));
    for j = 1:length(grad)
        grad(j) = grad(j) + (temph - y(i)) * X(i, j);
    end
end

J = J / m;
sum = 0;
for i = 2:length(grad)
    sum = sum + theta(i)^2;
end
J = J + sum * lambda/(2*m);

for i = 2:length(grad)
    grad(i) = grad(i) / m + (lambda / m) * theta(i);
end
grad(1) = grad(1) / m;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end