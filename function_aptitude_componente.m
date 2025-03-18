close all
clear all
clc

% Cambiar el directorio de trabajo al principal (donde están las carpetas)
cd('C:\Users\Enrique\OneDrive - Universidad Autónoma del Estado de México\Documents\Fall-risk-Bayesian-classifiers-main\Proyecto genetico\Public repository GitHub content\Public repository GitHub content');

% Agregar acceso a subcarpetas
addpath('data', 'Lib', 'parameters');



%% Optimal Bayesian classifiers:
% Este programa ahora lee los índices desde un archivo de texto y genera 
% un archivo de texto con el resultado de mean_AUC - std_AUC.

%% Cargar los datos estabilimétricos
% Cargar conjuntos de datos de entrenamiento, prueba y validación 
X_train = csvread('data/draf_0/X_trainEst.csv', 1, 1); 
X_test = csvread('data/draf_0/X_testEst.csv', 1, 1); 
X_val = csvread('data/draf_0/X_valEst.csv', 1, 1);
Y_train = csvread('data/draf_0/Y_trainEst.csv', 1, 1); 
Y_test = csvread('data/draf_0/Y_testEst.csv', 1, 1); 
Y_val = csvread('data/draf_0/Y_valEst.csv', 1, 1);
Y_train(Y_train==1) = 2; Y_train(Y_train==0) = 1; 
Y_test(Y_test==1) = 2; Y_test(Y_test==0) = 1; 
Y_val(Y_val==1) = 2; Y_val(Y_val==0) = 1;

%% Parámetros del modelo bayesiano
[N, num_caract] = size(X_train);
P = [.5 1-.5]'; % Probabilidad de caídas
X_train_fallers = X_train(find(Y_train==2), :);
X_train_no_fallers = X_train(find(Y_train==1), :);
m1 = median(X_train_fallers)'; 
m2 = median(X_train_no_fallers)'; 
Xtrain = X_train'; 
Xtest = X_test'; 
Xval = X_val';
Ytrain = Y_train'; 
Ytest = Y_test'; 
Yval = Y_val';

%% Leer los indicadores desde un archivo de texto
fileID = fopen('Indicadores.txt', 'r');
Indicadores_Utilizados = fscanf(fileID, '%d');
fclose(fileID);

% Clasificador bayesiano entrenado con las características optimizadas
[m1r, m2r, Xr] = Means_XtrainReduce(Indicadores_Utilizados', m1, m2, X_train);
P = [.5 1-.5]'; 
NoFaller = Xr(:, find(Ytrain == 1));
[mu1_circumflex, S1_circumflex] = GaussML_estimador(NoFaller);
Faller = Xr(:, find(Ytrain == 2));
[mu2_circumflex, S2_circumflex] = GaussML_estimador(Faller);
S(:, :, 1) = S1_circumflex;
S(:, :, 2) = S2_circumflex;
m = [m1r, m2r];

% Evaluación en entrenamiento
[Xtrain_Reduced] = XReduce(Indicadores_Utilizados', Xtrain);
Ybayes_train = ClasificadorBayesiano(m, S, P, Xtrain_Reduced);
[Se_bayesTrain, Es_bayesTrain] = evaluation(Ytrain, Ybayes_train);
[Xbayes, Ybayes, Tbayes, AUCbayes_train] = perfcurve(Ytrain', Ybayes_train, 2);

% Evaluación en prueba
[Xtest_Reduced] = XReduce(Indicadores_Utilizados', Xtest);
Ybayes_test = ClasificadorBayesiano(m, S, P, Xtest_Reduced);
[Se_bayesTest, Es_bayesTest] = evaluation(Ytest, Ybayes_test);
[Xbayes, Ybayes, Tbayes, AUCbayes_test] = perfcurve(Ytest', Ybayes_test, 2);

% Evaluación en validación
[Xval_Reduced] = XReduce(Indicadores_Utilizados', Xval);
Ybayes_val = ClasificadorBayesiano(m, S, P, Xval_Reduced);
[Se_bayesVal, Es_bayesVal] = evaluation(Yval, Ybayes_val);
[Xbayes, Ybayes, Tbayes, AUCbayes_val] = perfcurve(Yval', Ybayes_val, 2);

% Métrica descriptiva AUC por media y desviación estándar
mean_AUC = mean([AUCbayes_train, AUCbayes_test, AUCbayes_val]);
std_AUC = std([AUCbayes_train, AUCbayes_test, AUCbayes_val]);
result = mean_AUC - std_AUC;

% Guardar resultados en un archivo de texto
fileID = fopen('Resultados.txt', 'w');
fprintf(fileID, 'Resultado (mean_AUC - std_AUC): %.4f\n', result);
fclose(fileID);

disp('Resultados guardados en "Resultados.txt".');
