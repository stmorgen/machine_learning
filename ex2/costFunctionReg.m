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


% calculate costs first

sum = 0;

for i = 1:m

	Xi = X(i, :)';
	%temp = (theta' * Xi - y(i))^2;
	
	hXi = sigmoid(theta' * Xi);
	
	temp = -y(i) * log(hXi) - (1-y(i)) * log(1-hXi);
	
	sum = sum + temp;
end

J = sum / (m);
% J = sum / (2*m);

offsetSum = 0;

for j = 2:length(theta);
	offsetSum = offsetSum + theta(j)^2;
end

offset = (lambda / (2 * m)) * offsetSum;

J = J + offset;


% calculate gradient

n = length(theta); % number of attributes
	
factor = 1 / m;
%factor = alpha / m;

for j = 1:n
	tempSum = 0;
	for i = 1:m
	
		Xi = X(i,:)';
	
		hXi = sigmoid(theta' * Xi);
		%hi = theta' * Xi;
		
		Xij = X(i,j);
		
		offset = 0;
		if (j > 1) 
			offset = lambda / m * theta(j);
		end
		
		tempSum = tempSum + (hXi - y(i)) * Xij + offset;
	end
	
	delta = factor * tempSum;
	
	grad(j,1) = delta;
	% tempTheta(j,1) = theta(j,1) - delta;
end



% =============================================================

end
