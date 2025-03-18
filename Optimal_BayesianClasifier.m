close all
clear all
clc

%% Optimal Bayesian classifiers:
% This program allows the user to easily check which characteristics were 
% optimized by the simulated annealing. Please just run the program, 
% and it will then display the results of the performance of the Top 4 
% model for fall risk identification.


%% Loading of stabilimetric database
% Load train, test and validation data sets 
X_train=csvread('data/draf_0/X_trainEst.csv',1,1); X_test=csvread('data/draf_0/X_testEst.csv',1,1); X_val=csvread('data/draf_0/X_valEst.csv',1,1);
Y_train=csvread('data/draf_0/Y_trainEst.csv',1,1); Y_test=csvread('data/draf_0/Y_testEst.csv',1,1); Y_val=csvread('data/draf_0/Y_valEst.csv',1,1);
Y_train(Y_train==1)=2;Y_train(Y_train==0)=1; Y_test(Y_test==1)=2;Y_test(Y_test==0)=1; Y_val(Y_val==1)=2;Y_val(Y_val==0)=1;

%% Bayesian model parameters
[N,num_caract]=size(X_train);
P=[.5 1-.5]'; %Probabilidad de caídas
X_train_fallers=X_train(find(Y_train==2),:);
X_train_no_fallers=X_train(find(Y_train==1),:);
m1=median(X_train_fallers)'; m2=median(X_train_no_fallers)'; 
Xtrain=X_train'; Xtest=X_test'; Xval=X_val';
Ytrain=Y_train'; Ytest=Y_test'; Yval=Y_val';

% Dimension
dimensiones = 15;

% Characteristics optimized by simulated annealing
Indicadores_Utilizados= [1   3   4   8  16  20  26  30  36  38  40  42  44  45  46]';

% Bayesian classifier trained from characteristics optimized by simulated annealing
[m1r,m2r,Xr]= Means_XtrainReduce(Indicadores_Utilizados',m1,m2,X_train);
P=[.5 1-.5]'; 
NoFaller=Xr(:,find(Ytrain==1));
[mu1_circumflex,S1_circumflex] = GaussML_estimador(NoFaller);
Faller=Xr(:,find(Ytrain==2));
[mu2_circumflex,S2_circumflex] = GaussML_estimador(Faller);
S(:,:,1)= S1_circumflex;
S(:,:,2)=S2_circumflex;
m=[m1r,m2r];

% train evaluation
[Xtrain_Reduced]=XReduce(Indicadores_Utilizados',Xtrain);
Ybayes_train= ClasificadorBayesiano(m,S,P,Xtrain_Reduced);
[Se_bayesTrain,Es_bayesTrain] = evaluation(Ytrain,Ybayes_train);
[Xbayes,Ybayes,Tbayes,AUCbayes_train]=perfcurve(Ytrain',Ybayes_train,2);

% test evaluation
[Xtest_Reduced]=XReduce(Indicadores_Utilizados',Xtest);
Ybayes_test = ClasificadorBayesiano(m,S,P,Xtest_Reduced);
[Se_bayesTest,Es_bayesTest] = evaluation(Ytest,Ybayes_test);
[Xbayes,Ybayes,Tbayes,AUCbayes_test]=perfcurve(Ytest',Ybayes_test,2);

% validation evaluation
[Xval_Reduced]=XReduce(Indicadores_Utilizados',Xval);
Ybayes_val= ClasificadorBayesiano(m,S,P,Xval_Reduced);
[Se_bayesVal,Es_bayesVal] = evaluation(Yval,Ybayes_val);
[Xbayes,Ybayes,Tbayes,AUCbayes_val]=perfcurve(Yval',Ybayes_val,2);

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
