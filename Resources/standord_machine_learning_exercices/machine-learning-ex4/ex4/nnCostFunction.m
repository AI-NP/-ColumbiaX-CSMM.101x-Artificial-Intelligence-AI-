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
K = size(Theta2,1);
         
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
X = [ones(m,1) X];
z2 = Theta1*X';
a2 = sigmoid(z2);
a2 = [ones(1,columns(a2)); a2];
z3 = Theta2*a2;
a3 = sigmoid(z3)';

delta3 = zeros(K,1);
DELTA1 = 0;
DELTA2 = 0;
d3 = 0;
for t=1:m,
  	Y = zeros(1,K);
	Y(1,y(t)) = 1;
	for k = 1:K,
		cost = (-Y(k)*log(a3(t,k)))-((1-Y(k))*log(1-a3(t,k))); 
		J+=cost;
	end;
	delta3 = a3(t,:) - Y;%El valor que la neurona calcula, menos el valor eral
	delta2 = Theta2'(2:end,:)*delta3'.*sigmoidGradient(z2(:,t));
	DELTA1 = DELTA1 + delta2*X(t,:);
	DELTA2 = DELTA2 + delta3'*a2(:,t)';
end;

J= J/m;

Theta1Squared = Theta1(1:end,2:end).*Theta1(1:end,2:end);
Theta2Squared = Theta2(1:end,2:end).*Theta2(1:end,2:end);

J = J+ (lambda/(2*m))*(sum(Theta1Squared(:))+sum(Theta2Squared(:)));

Theta1_grad  = (1/m)*DELTA1 + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
Theta2_grad  = (1/m)*DELTA2 + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];
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
% Note to me: implemente la parte 2 dentro del codigo de la parte 1 ya que hacen calculos compartidos y asi es mas optimo

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
