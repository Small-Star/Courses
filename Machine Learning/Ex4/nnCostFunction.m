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

%P1
a1 = [ones(size(X,1),1), X];
%size(a1)
a2 = sigmoid(a1*Theta1');
a2 = [ones(size(a2, 1), 1) a2];	%Prepend column
%size(a2)
h = sigmoid(a2*Theta2');
%size(h)

y_n = zeros(size(X,1),num_labels);
%size(y_n)

for l = 1:size(y_n)
	y_n(l,y(l)) = 1;
end

J = 0;
for k = 1:num_labels
	J += sum((1/m)*(-y_n(:,k)'*log(h(:,k)) - (1 - y_n(:,k)')*log(1 - h(:,k))));
end

J_reg = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J += J_reg;

%P2
acc = 0;
Theta1_grad = 0;
Theta2_grad = 0;
Del_2 = 0;
Del_1 = 0;
for t = 1:m
	%S1
	%Feedforward calculated previously

	%S2
	del_3 = h(t,:) - y_n(t,:);

	%S3
	del_2 = (Theta2'*del_3').*sigmoidGradient([1; Theta1*(a1(t,:)')]);

	%S4
	del_2 = del_2(2:end)';
	Del_2 += del_3'*(a2(t,:));
	Del_1 += del_2'*(a1(t,:));

end
	%S5
	Theta1_grad = Del_1/m;
	Theta2_grad = Del_2/m;

	Theta1_grad += (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; 
	Theta2_grad += (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:,2:end)]; 









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
