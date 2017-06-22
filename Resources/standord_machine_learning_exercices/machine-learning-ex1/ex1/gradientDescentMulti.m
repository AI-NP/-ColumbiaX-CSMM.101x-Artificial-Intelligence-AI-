function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters

    evaluated_matrix = X*theta;

    number_of_features = columns(X);
    temp = zeros(number_of_features,1);
    for j = 1:number_of_features,
        temp(j) = theta(j) - alpha*(1/m)* sum((evaluated_matrix(:,1)-y(:,1)).*X(:,j) ); 
    end;
    
    theta = temp;


    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    disp(J_history(iter));	
end

end
