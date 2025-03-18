function [Z] = ClasificadorBayesiano(m, S, P, X)
%% BayesianClassifier: This function implements a Bayesian classifier to classify data samples in matrix X into two classes (c=2). 
% The classifier uses the estimated mean vectors m, covariance matrices S, and prior probabilities P for the two classes. 
% For each observation in X, it calculates the posterior probabilities for both classes, and assigns the class with the higher probability as the output label Z.
c = 2; % Number of classes (assumed to be 2 in this case)
    %N = length(X); % Number of observations in X
    [~,N]=size(X);  % Number of observations in X
    Z = zeros(1, N); % Initialize the output array for class labels
    
    for i = 1:N
        T = zeros(1, c); % Initialize the array for storing the posterior probabilities for each class
        
        % Calculate the posterior probability for each class
        for j = 1:c
            T(j) = P(j) * PDFGaussiana(m(:, j), S(:, :, j), X(:, i));
        end
        
        % Assign the class label based on the maximum posterior probability
        [~, Z(i)] = max(T);
    end
end

function [VAL] = PDFGaussiana(m, S, x)
%% PDFGaussian: This function computes the probability density function (PDF) of a multivariate Gaussian distribution with mean vector m, covariance matrix S, and input data vector x. 
% The PDF value is calculated based on the formula for a multivariate Gaussian distribution. Note that the function assumes m and x are column vectors, and S is a positive definite covariance matrix. 
    [p, ~] = size(m); % p is the number of dimensions (features) in the data
    
    % Calculate the probability density function of the multivariate Gaussian distribution
    VAL = (1 / ((2 * pi)^(p/2) * det(S)^0.5)) * exp(-0.5 * (x - m)' * inv(S) * (x - m));
end
