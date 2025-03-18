%% A function to calculate a maximum likelihood estimator to obtain the parameters of a Gaussian distribution from a data set in matrix form.
% mu_hat = maximum likelihood estimator for the mean
% S_hat = maximum likelihood estimator for the covariance matrix
function [mu_hat, S_hat] = GaussML_estimador(X)
    [r, N] = size(X); % r is the number of rows (dimension of variables), N is the number of observations
    
    mu_hat = (1/N) * sum(X')'; % Calculation of the maximum likelihood estimator for the mean
    
    S_hat = zeros(r); % Initialize the estimated covariance matrix with zeros
    
    for k = 1:N
        S_hat = S_hat + (X(:, k) - mu_hat) * (X(:, k) - mu_hat)'; % Calculation of the sum of products (x - mu) * (x - mu)'
    end
    
    S_hat = (1/N) * S_hat; % Calculation of the maximum likelihood estimator for the covariance matrix
end


