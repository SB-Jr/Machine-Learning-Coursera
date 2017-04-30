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

  sumJ = 0;

  % === calculating the loop ===
  temp = (X*theta - y);
  %display(temp);
  temp = [temp,temp].*X;
  %display(temp);
  %fprintf('temp sum 1 and 2');
  %display(sum([temp(:,1)]));
  temp2 = [sum(temp(:,1));sum(temp(:,2))];
  sumJ = temp2;

  % === calculating theta ===
  theta = theta - (alpha*sumJ)/m;





    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end
