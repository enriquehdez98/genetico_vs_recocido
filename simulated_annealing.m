warning('off')
close all;
clear all;
clc;

%% Optimizing the diagnostic power of fall risk in young to older adults using Bayesian classifiers and simulated annealing.
% This program serves to optimize the selection of features that best describe the fall risk using Bayesian classification technique and simulated annealing.

% Input:
% dimensiones = Fimension number of Bayesian classifiers. 

% Output:
% Results: 
% A results table containing Optimal feature combination (Indices) and its dimension, Sensitivity (SE), Specificity (ES), and AUC for the train, test, 
% and validation subsets, along with the global mean and standard deviation for each metric.


%% Loading of stabilimetric database
% Load train, test and validation data sets 
X_train=csvread('data/draf_0/X_trainEst.csv',1,1); X_test=csvread('data/draf_0/X_testEst.csv',1,1); X_val=csvread('data/draf_0/X_valEst.csv',1,1);
Y_train=csvread('data/draf_0/Y_trainEst.csv',1,1); Y_test=csvread('data/draf_0/Y_testEst.csv',1,1); Y_val=csvread('data/draf_0/Y_valEst.csv',1,1);
Y_train(Y_train==1)=2;Y_train(Y_train==0)=1; Y_test(Y_test==1)=2;Y_test(Y_test==0)=1; Y_val(Y_val==1)=2;Y_val(Y_val==0)=1;

%% Bayesian model parameters
[N,num_caract]=size(X_train);
P=[.5 1-.5]'; 
X_train_fallers=X_train(find(Y_train==2),:); X_train_no_fallers=X_train(find(Y_train==1),:);
m1=median(X_train_fallers)'; m2=median(X_train_no_fallers)'; 
Xtrain=X_train'; Xtest=X_test'; Xval=X_val'; Ytrain=Y_train'; Ytest=Y_test'; Yval=Y_val';

%% Simulated annealing algorithm
%%% initial parameters 
T=0.5979; 
Tmin=0.0232; 
Lk=1; 
i=1;C0=0;k=1;
resul="";
[~,total_features]= size(X_train);

% Ask user to enter the number of the dimension
dimensiones = input('Enter dimension number (11-36): ');
if dimensiones < 11 || dimensiones > 36
    error('The dimension must be between 11 and 36.');
end

% Define the nums vector based on the selected dimension and previous experimental results.
switch dimensiones
    case 11
        nums = [1 4 5 7 11 15 16 17 19 23 25];
    case 12
        nums = [1 2 15 19 20 21 36 37 43 44 45];
    case 13
        nums = [1 2 4 10 16 17 23 30 34 37 42 47];
    case 14
        nums = [1 4 5 16 17 18 22 29 30 37 42 44 47];
    case 15
        nums = [1 3 4 8 12 16 18 22 35 36 37 42 44 47];
    case 16
        nums = [1 3 4 8 16 20 26 30 36 38 40 42 44 45 46];
    case 17
        nums = [1 2 4 6 7 12 13 16 17 20 22 27 32 38 42 43];
    case 18
        nums = [1 2 4 5 7 9 13 17 22 27 30 35 37 38 40 43 47];
    case 19
        nums = [1 3 4 6 7 8 11 19 20 24 27 28 30 34 36 38 40 42];
    case 20
        nums = [2 4 5 6 7 9 10 11 13 14 16 20 32 33 34 36 40 42 47];
    case 21
        nums = [1 3 4 6 7 10 11 14 15 18 20 22 24 31 34 35 37 42 44 47];
    case 22
        nums = [1 2 7 9 10 13 18 19 21 22 24 25 26 30 32 34 36 38 39 40 46];
    case 23
        nums = [1 2 3 5 12 16 20 21 22 25 26 27 28 30 31 34 38 40 44 45 46 47];
    case 24
        nums = [1 2 6 9 10 11 13 14 15 16 18 23 26 29 32 34 38 39 40 41 42 45 47];
    case 25
        nums = [1 2 4 5 6 7 12 14 15 16 20 21 22 23 28 31 35 36 37 38 42 43 44 46];
    case 26
        nums = [1 3 5 10 12 16 17 18 19 20 21 22 23 24 25 26 28 32 35 36 38 41 42 44 45];
    case 27
        nums = [1 2 3 5 7 9 10 12 15 18 20 21 22 24 25 26 27 31 32 34 37 38 40 44 46 47];
    case 28
        nums = [2 3 5 6 12 13 15 16 17 20 21 22 23 24 25 26 27 30 33 35 36 37 38 40 42 43 47];
    case 29
        nums = [1 2 4 5 6 7 10 12 13 14 15 16 17 22 25 26 27 28 31 32 33 35 37 38 40 41 42 43];
    case 30
        nums = [1 3 4 5 6 7 9 13 14 15 16 20 23 24 25 26 27 30 33 35 36 37 38 39 40 41 42 43 45];
    case 31
        nums = [1 3 5 8 12 14 15 17 18 19 20 21 22 23 24 25 26 27 28 31 32 35 36 37 38 41 42 43 46 47];
    case 32
        nums = [1 3 4 5 6 7 9 13 14 15 17 18 19 20 21 22 23 24 25 26 31 32 33 34 35 36 37 38 40 45 46];
    case 33
        nums = [2 4 5 6 7 8 11 13 15 16 17 18 19 21 22 23 24 25 26 28 32 36 37 38 40 41 42 43 44 45 46 47];
    case 34
        nums = [1 3 4 5 6 7 10 12 14 15 16 17 18 19 21 22 23 24 25 26 27 30 34 35 36 37 38 39 40 41 44 45 47];
    case 35
        nums = [1 4 5 6 10 11 12 14 15 16 17 19 20 21 22 23 24 25 26 28 30 31 32 34 36 37 38 39 40 42 43 44 45 47];
    case 36
        nums = [2 4 5 7 13 14 15 18 19 20 21 22 25 26 30 33 35 37 38 39 40 42 43 44 49 51 57 62 64 65 69 70 71 72 73];
end

% Create vector of selected features and a new random value extra 
vect_opt = zeros(1, total_features);
vect_opt(nums) = 1;
while length(nums) < dimensiones
    index_next = randperm(length(find(vect_opt == 0)), 1);
    index_past = randperm(length(find(vect_opt == 1)), 1);
    index_past = max(find(vect_opt == 1, index_past));
    index_next = max(find(vect_opt == 0, index_next));
    vect_opt(index_next) = 1;
    nums = find(vect_opt == 1);
end

disp('Optimizing feature selection, this process can take several minutes ...')

while (T>Tmin)
    for l=1:Lk
        %% Permutation and dimensionality reduction
        index_next=randperm(length(find(vect_opt==0)),1);
        index_past=randperm(length(find(vect_opt==1)),1);
        index_past=max(find(vect_opt == 1, index_past));
        index_next=max(find(vect_opt == 0, index_next));
        vect_opt(index_past)=0;
        vect_opt(index_next)=1;
        nums = find(vect_opt==1);
        [m1r,m2r,Xr]= Means_XtrainReduce(nums,m1,m2,X_train);
        
        %% Bayessinao classifier 
        % Bayesian model parameters for reduced sets
        NoFaller=Xr(:,find(Ytrain==1));
        [mu1_circumflex,S1_circumflex] = GaussML_estimador(NoFaller);
        Faller=Xr(:,find(Ytrain==2));
        [mu2_circumflex,S2_circumflex] = GaussML_estimador(Faller);
        S(:,:,1)= S1_circumflex;
        S(:,:,2)=S2_circumflex;
        m=[m1r,m2r];
        
        % Reduced sets
        [Xtrain_Reduced]=XReduce(nums,Xtrain);
        [Xtest_Reduced]=XReduce(nums,Xtest);
        [Xval_Reduced]=XReduce(nums,Xval);
        
        % Bayessinao classifier prediction
        Ybayes_train = ClasificadorBayesiano(m,S,P,Xtrain_Reduced);  
        Ybayes_test = ClasificadorBayesiano(m,S,P,Xtest_Reduced);
        Ybayes_val = ClasificadorBayesiano(m,S,P,Xval_Reduced);
               
        %% Bayessinao classifier evaluation
        % AUCs
        [Xbayes,Ybayes,Tbayes,AUCbayes_train]=perfcurve(Ytrain',Ybayes_train,2);
        [Xbayes,Ybayes,Tbayes,AUCbayes_test]=perfcurve(Ytest',Ybayes_test,2);
        [Xbayes,Ybayes,Tbayes,AUCbayes_val]=perfcurve(Yval',Ybayes_val,2);
        % Sensitivity and Specificity
        [Se_bayesTrain,Es_bayesTrain] = evaluation(Ytrain,Ybayes_train);
        [Se_bayesTest,Es_bayesTest] = evaluation(Ytest,Ybayes_test);
        [Se_bayesVal,Es_bayesVal] = evaluation(Yval,Ybayes_val);
        
        %% Bayessinao classifier penalty
        if (Se_bayesTest<0.6)|(Es_bayesTest<0.6)|(Se_bayesVal<0.6)|(Es_bayesVal<0.6)|(Se_bayesTrain<0.6)|(Es_bayesTrain<0.6)
            AUCbayes_test=0; AUCbayes_val=0; AUCbayes_train=0;
        end   
        
        %% Simulated annealed algorithm cost function
        error=[AUCbayes_test;AUCbayes_val;AUCbayes_train];
        er(:,:,i)=sum(error);    
        Cp=mean(error)-std(error); 
        Vec(i)= mean(error)-std(error);

        %% Continuity of simulated annealing algorithm
        DeltaE= Cp-C0; 
        if DeltaE>=0
            C0=Cp; 
        elseif exp(DeltaE/(T))>rand(1,1) 
            C0=Cp; 
        end 
            numeros(:,:,i)= nums';
            i=i+1; 
    end
    k=k+1; 
    T=T*0.83; 
    Lk=Lk+Lk*(1-exp(-(1))) ; 
end


%% Results
% Characteristics optimized by simulated annealing
minimos=find(Vec==max(Vec));
Indicadores_Utilizados=numeros(:,:,minimos(1));

% Bayesian classifier trained from characteristics optimized by simulated annealing
[m1r,m2r,Xr]= Means_XtrainReduce(numeros(:,:,minimos(1))',m1,m2,X_train);
P=[.5 1-.5]'; 
NoFaller=Xr(:,find(Ytrain==1));
[mu1_circumflex,S1_circumflex] = GaussML_estimador(NoFaller);
Faller=Xr(:,find(Ytrain==2));
[mu2_circumflex,S2_circumflex] = GaussML_estimador(Faller);
S(:,:,1)= S1_circumflex;
S(:,:,2)=S2_circumflex;
m=[m1r,m2r];

% train evaluation
[Xtrain_Reduced]=XReduce(numeros(:,:,minimos(1))',Xtrain);
Ybayes_train= ClasificadorBayesiano(m,S,P,Xtrain_Reduced);
[Se_bayesTrain,Es_bayesTrain] = evaluation(Ytrain,Ybayes_train);
[Xbayes,Ybayes,Tbayes,AUCbayes_train]=perfcurve(Ytrain',Ybayes_train,2);

% test evaluation
[Xtest_Reduced]=XReduce(numeros(:,:,minimos(1))',Xtest);
Ybayes_test = ClasificadorBayesiano(m,S,P,Xtest_Reduced);
[Se_bayesTest,Es_bayesTest] = evaluation(Ytest,Ybayes_test);
[Xbayes,Ybayes,Tbayes,AUCbayes_test]=perfcurve(Ytest',Ybayes_test,2);

% validation evaluation
[Xval_Reduced]=XReduce(numeros(:,:,minimos(1))',Xval);
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





