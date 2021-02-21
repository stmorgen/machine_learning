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

% feed forward, result in p
a1 = [ones(m,1), X];
z2 = Theta1 * a1';
t2 = sigmoid(z2);
a2 = [ones(m,1), t2'];
z3 = Theta2 * a2';
a3 = sigmoid(z3);

for i=1:m

	tempY = zeros(num_labels, 1);
	tempY(y(i), 1) = 1;
	
	hxi = a3(: , i);

	tempCost = -tempY' * log(hxi) - (1-tempY)' * log (1-hxi);
	J += sum(tempCost);
end

% add regularization for costs
tempTheta1 = Theta1;
tempTheta1(:, 1) = 0;
tempTheta2 = Theta2;
tempTheta2(:, 1) = 0;

jOffset = (sum(sum(tempTheta1.^2)) + sum(sum(tempTheta2.^2))) * lambda / ( 2 * m);

J = J / m + jOffset;

 

for i=1:m

% step 1
	% extract sample
	xi = X(i, :)';
	% calculate layer 1
	a1 = [1; xi];
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	% calculate layer 2
	a2 = [1; a2];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);
	% convert y value for sample into vector
	tempY = zeros(num_labels, 1);
	tempY(y(i), 1) = 1;
	

% step 2	
	delta3 = a3 - tempY;
	
% step 3

	t2 = Theta2(:, 2:end); % without this adjustment there is a dimension error in line 120

	delta2 = (t2' * delta3) .* sigmoidGradient(z2); 
	%delta2 = delta2(2:end); % after prior adjustement this has to be skipped to avoid dimension error in line 124
	
% step 4
	Theta1_grad += delta2 * a1';
	Theta2_grad += delta3 * a2';
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% add regularization for gradients
offset1 = Theta1 * lambda / m;
offset1(:, 1) = 0; % set values for j=0 to zero
offset2 = Theta2 * lambda / m;
offset2(:, 1) = 0; % set values for j=0 to zero

Theta1_grad += offset1;
Theta2_grad += offset2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
