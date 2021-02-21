function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	n = length(theta); % number of attributes
	
	factor = alpha / m;
	
	tempTheta = theta;
	
	
	
	for j = 1:n
		tempSum = 0;
		for i = 1:m
		
			Xi = X(i,:)';
		
			hi = theta' * Xi;
			
			Xij = X(i,j);
			
			tempSum = tempSum + (hi - y(i)) * Xij;
		end
		
		delta = factor * tempSum;
		
		tempTheta(j,1) = theta(j,1) - delta;
	end
	
	theta = tempTheta;
	
	


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
