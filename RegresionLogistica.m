% Clean the environment
close all;
clear all;
clc;

% Load training, test, and validation datasets from CSV files
X_train = csvread('data/X_trainEst.csv', 1, 1);
X_test = csvread('data/X_testEst.csv', 1, 1);
X_val = csvread('data/X_valEst.csv', 1, 1);
Y_train = csvread('data/Y_trainEst.csv', 1, 1);
Y_test = csvread('data/Y_testEst.csv', 1, 1);
Y_val = csvread('data/Y_valEst.csv', 1, 1);

% Create a matrix to store evaluation results (42 features x 9 metrics)
Resul = zeros(47, 9);

% Loop over each of the 42 features for classification
for i = 1:47
    
    % Train a logistic regression model using the current feature from X_train and corresponding labels Y_train
    Mdl = fitclinear(X_train(:, i), Y_train, 'Learner', 'logistic');

    % Make predictions on the training set using the trained model
    y_train = predict(Mdl, X_train(:, i));

    % Calculate ROC curve metrics and AUC for the training set
    [XRL, YRL, TRL, AUCtrain] = perfcurve(Y_train, y_train, 1);

    % Evaluate logistic regression results for the training set
    [Se_Zbayes, Es_Zbayes] = evaluationRL(Y_train, y_train);

    % Store the results for the training set in the Resul matrix
    Resul(i, 1) = Se_Zbayes; % Sensitivity (True Positive Rate)
    Resul(i, 2) = Es_Zbayes; % Specificity (True Negative Rate)
    Resul(i, 3) = AUCtrain;  % Area Under the ROC Curve

    % Repeat the process for the test set
    y_test = predict(Mdl, X_test(:, i));
    [XRL, YRL, TRL, AUCtest] = perfcurve(Y_test, y_test, 1);
    [Se_Zbayes, Es_Zbayes] = evaluationRL(Y_test, y_test);
    Resul(i, 4) = Se_Zbayes;
    Resul(i, 5) = Es_Zbayes;
    Resul(i, 6) = AUCtest;

    % Repeat the process for the validation set
    y_val = predict(Mdl, X_val(:, i));
    [XRL, YRL, TRL, AUCval] = perfcurve(Y_val, y_val, 1);
    [Se_Zbayes, Es_Zbayes] = evaluationRL(Y_val, y_val);
    Resul(i, 7) = Se_Zbayes;
    Resul(i, 8) = Es_Zbayes;
    Resul(i, 9) = AUCval;
    
end
