%% Validation of characteristics optimized by simulated annealing using bostrapping aggregation
% This program is used to test the predictive validity of the set of features optimized by simulated annealing using the boostrap aggregation technique (number of samples = 150).
% Input:
% options = Selection of the Top combinations to test. It is possible to test the top 5 feature combinations 
% with the highest Area Under the Curve (AUC).

% Output:
% Results: A results table containing Sensitivity (SE), Specificity (SP), and AUC for the train, test, 
% and validation subsets, along with the global mean and standard deviation for each metric.

close all
clear all
clc

%% Loading of stabilimetric database
% Load train, test and validation data sets 
X_train=csvread('data/draf_0/X_trainEst.csv',1,1); X_test=csvread('data/draf_0/X_testEst.csv',1,1); X_val=csvread('data/draf_0/X_valEst.csv',1,1);
Y_train=csvread('data/draf_0/Y_trainEst.csv',1,1); Y_test=csvread('data/draf_0/Y_testEst.csv',1,1); Y_val=csvread('data/draf_0/Y_valEst.csv',1,1);
Y_train(Y_train==1)=2;Y_train(Y_train==0)=1; Y_test(Y_test==1)=2;Y_test(Y_test==0)=1; Y_val(Y_val==1)=2;Y_val(Y_val==0)=1;
resul="";

% Characteristics optimized by simulated annealing
% Define the options for optimized feature sets
options = {
    [1   3   4   5   6   7   9  13  14  15  17  18  19  20  21  22  23  24  25  26  31  32  33  34  35  36  37  38  40  45  46]', ...  % Top 1
    [1   3   4   5   6   7   9  13  14  15  16  20  23  24  25  26  27  30  33  35  36  37  38  39  40  41  42  43  45]', ...  % Top 2
    [1   2   4   5   6   7  12  14  15  16  20  21  22  23  28  31  35  36  37  38  42  43  44  46]', ...  % Top 3
    [1   3   4   8  16  20  26  30  36  38  40  42  44  45  46]', ...  % Top 4
    [1   3   5   8  12  14  15  17  18  19  20  21  22  23  24  25  26  27  28  31  32  35  36  37  38  41  42  43  46  47]'  % Top 5
};
choice = menu('Select the set of optimized features:', ...
    'Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5');

% Assign the selected features to the variable
Indicadores_Utilizados = options{choice};
dimensiones = length(Indicadores_Utilizados);


%% Bayesian model parameters
[N,num_caract]=size(X_train);
P=[.5 1-.5]'; 
X_train_fallers=X_train(find(Y_train==2),:);
X_train_no_fallers=X_train(find(Y_train==1),:);
m1=median(X_train_fallers)'; m2=median(X_train_no_fallers)'; 
Xtrain=X_train'; Xtest=X_test'; Xval=X_val';
Ytrain=Y_train'; Ytest=Y_test'; Yval=Y_val';


%% Bootstrap aggregation  method
% Number of classifiers
num_classifiers = 150;
% Initialization of matrices to store results
Ybayes_train_all = zeros(num_classifiers, length(Ytrain));
Ybayes_test_all = zeros(num_classifiers, length(Ytest));
Ybayes_val_all = zeros(num_classifiers, length(Yval));


for i = 1:num_classifiers
    % Generate bootstrap samples
    idx = randsample(1:N, N, true);
    X_train_bootstrap = X_train(idx, :);
    Y_train_bootstrap = Y_train(idx);
    
    % Train Bayesian classifier with bootstrap samples
    [m1r, m2r, Xr] = Means_XtrainReduce(Indicadores_Utilizados', m1, m2, X_train_bootstrap);
    NoFaller = Xr(:, Y_train_bootstrap == 1);
    [mu1_circumflex, S1_circumflex] = GaussML_estimador(NoFaller);
    Faller = Xr(:, Y_train_bootstrap == 2);
    [mu2_circumflex, S2_circumflex] = GaussML_estimador(Faller);
    
    S(:,:,1) = S1_circumflex;
    S(:,:,2) = S2_circumflex;
    m = [m1r, m2r];
    
    % Classification with the trained model
    [Xtrain_Reduced] = XReduce(Indicadores_Utilizados', Xtrain);
    Ybayes_train_all(i, :) = ClasificadorBayesiano(m, S, P, Xtrain_Reduced);
    
    [Xtest_Reduced] = XReduce(Indicadores_Utilizados', Xtest);
    Ybayes_test_all(i, :) = ClasificadorBayesiano(m, S, P, Xtest_Reduced);

    [Xval_Reduced] = XReduce(Indicadores_Utilizados', Xval);
    Ybayes_val_all(i, :) = ClasificadorBayesiano(m, S, P, Xval_Reduced);
end

% Aggregate results using majority vote
Ybayes_train_final = mode(Ybayes_train_all, 1);
Ybayes_test_final = mode(Ybayes_test_all, 1);
Ybayes_val_final = mode(Ybayes_val_all, 1);

% Evaluate the final model on the training set
[Se_bayesTrain, Es_bayesTrain] = evaluation(Ytrain, Ybayes_train_final);
[Xbayes, Ybayes, Tbayes, AUCbayes_train] = perfcurve(Ytrain', Ybayes_train_final, 2);


% Evaluate the final model on the test set
[Se_bayesTest, Es_bayesTest] = evaluation(Ytest, Ybayes_test_final);
[Xbayes_test, Ybayes_test, Tbayes_test, AUCbayes_test] = perfcurve(Ytest', Ybayes_test_final, 2);


% Evaluate the final model on the validation set
[Se_bayesVal, Es_bayesVal] = evaluation(Ytest, Ybayes_test_final);
[Xbayes_test, Ybayes_test, Tbayes_test, AUCbayes_val] = perfcurve(Yval', Ybayes_val_final, 2);

% Descriptive AUC metric by mean and standard deviation
mean_AUC = mean([AUCbayes_train,AUCbayes_test, AUCbayes_val]);
std_AUC = std([AUCbayes_train,AUCbayes_test, AUCbayes_val]);

% Results
Results = table(...
    dimensiones, ...
    {Indicadores_Utilizados'}, ...
    Se_bayesTrain, Es_bayesTrain, AUCbayes_train, ...
    Se_bayesTest, Es_bayesTest, AUCbayes_test, ...
    Se_bayesVal, Es_bayesVal, AUCbayes_val, ...
    mean_AUC, std_AUC, ...
    'VariableNames', {'Dimension', 'Indices', 'SE Train', 'SP Train', 'AUC Train', ...
    'SE Test', 'SP Test', 'AUC Test', 'SE Val', 'SP Val', 'AUC Val', 'mean AUC','std AUC'});

disp('Results of Bayesian Classifier  by Bootstrap Aggregation:');
disp(Results);
