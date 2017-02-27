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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


argumento = theta' * X';
[h] = sigmoid( argumento );

suma = 0.0;
for i = 1:m
	suma = suma + ( -y(i)*log(h(i)) - (1-y(i))*log(1-h(i)) ) ;
endfor

% starting in two because theta_0 is not consider for regularization 
suma2 = 0.0;
for i = 2:length(theta)
	suma2 = suma2 + theta(i)*theta(i);
endfor

J = (1.0/m)*suma + lambda/(2*m)*suma2;

aux = (h' .- y) .* X;
grad(1) = (1.0/m) * sum(aux)(1);

for i=2:length(theta)
	grad(i) = ((1.0/m) * sum(aux)(i)) + (lambda/m)*theta(i);
endfor


% =============================================================

end
