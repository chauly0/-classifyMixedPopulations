% April 22, 2023
% Written by Chau Ly
% Department of Bioengineering 
% Laboratory of Dr. Amy Rowat, University of California, Los Angeles 

% The general design of the code is further detailed in the publication
% referenced below: 
% Ly C, Ogana H, Kim HN, Hurwitz S, Deeds EJ, Kim YM, Rowat AC."Altered
% physical phenotypes of leukemia cells that survive chemotherapy
% treatment." Integrative Biology (2023). 
% 
% This code is divided into two parts:
% (1) Randomize a single-cell physical phenotype (transit time, TT1, only) using experimental
% qc-DC data into computationally generated mixed populations containing
% varying proportions of "resistant" cells 
% (2) Train machine learning models - kNN, SVM, and ensemble classifier of
% decision trees - and complete Bayesian optimization using hold out of trained 
% mixed populations of varying proportions of "resistant" cells. 
% Test the trained models and assess classification accuracy of known mixed populations.
% Repeat pipeline for 10 Monte Carlo iterations. 
clc
clear


%% Initialize 

% Choose how many cells will be in a sample from each synthetic patient
numCells = 100;
numPatients = 500;

% Variables for machine learning 
numModelsMonteCarl = 10;
fractionHoldOut = 0.2; 
bayesCycle = 10; 

% Proportions of VDL-treated cells in the "Resistant" class
% Will hold machine learning accuracy for each unique model from a Monte
% Carlo iteration
accuracyKNN = zeros(numModelsMonteCarl+1,7);
accuracySVM = zeros(numModelsMonteCarl+1,7);
accuracyEns = zeros(numModelsMonteCarl+1,7);

resistantPer = [1 2 10 25 50 75 99];

accuracyKNN(1,:) = resistantPer;
accuracySVM(1,:) = resistantPer;
accuracyEns(1,:) = resistantPer;

% Will hold hyperparameters for each Monte Carlo iteration
optimalHyperp = cell(numModelsMonteCarl,4,3);

%% Load physical phenotypes from qc-DC experimental data
LAX53 = readtable('LAX53.xlsx');
LAX53VDL = readtable('LAX53VDL.xlsx');
LAX7R = readtable('LAX7R.xlsx');
LAX7RVDL = readtable('LAX7RVDL.xlsx');

% Edit tables for only relevant features
LAX53 = LAX53(:,2);
LAX53VDL = LAX53VDL(:,2);
LAX7R = LAX7R(:,2);
LAX7RVDL = LAX7RVDL(:,2);

poolSens = vertcat(LAX53,LAX7R);
poolRes = vertcat(LAX53VDL,LAX7RVDL);

% Convert from table to cell array due to indexing working better in cell array
poolSens = table2cell(poolSens);
poolRes = table2cell(poolRes);

for kMdls = 1:numModelsMonteCarl
%% Generate computational sets, or "patients", with a random sample of cells

% Combine the pure and mixed populations. Partition for machine learning
% Note, can't combine at the beginning. Need to be able to know which
% "resistant" patients are from which proportion of mixed populations
% Combine pure populations and mixed populations before partitioning

patientsPure = cell(1,1,7);
patientsMixed = cell(1,1,7);

for kPatients = 1:size(resistantPer,2)
patientsPure{1,1,kPatients} = randomizeTT1_Pure(poolSens, numCells, numPatients);
patientsMixed{1,1,kPatients} = randomizeTT1_Mixed(poolSens, poolRes, numCells, numPatients, ...
    round(1-resistantPer(1,kPatients)*0.01,2), resistantPer(1,kPatients)*0.01);
end 


patientsComb = cell(2,1,7);
for kPatientsComb = 1:size(resistantPer,2)
    patientsComb{1,1,kPatientsComb} = cell2table(vertcat(patientsPure{1,1,kPatientsComb}', patientsMixed{1,1,kPatientsComb}'));
    patientsComb{2,1,kPatientsComb} = resistantPer(1,kPatientsComb);

end 



%% Partition into 80%-20% holdout 
patientsPartition = cvpartition(height(patientsComb{1,1,1}),'Holdout',0.2); 

% Training data indexing 
iTrain = training(patientsPartition);

patientsTrain = vertcat(patientsComb{1,1,1}(iTrain,:), patientsComb{1,1,2}(iTrain,:), patientsComb{1,1,3}(iTrain,:), ...
    patientsComb{1,1,4}(iTrain,:), patientsComb{1,1,5}(iTrain,:), patientsComb{1,1,6}(iTrain,:), patientsComb{1,1,7}(iTrain,:));

% Test data indexing 
iTest = test(patientsPartition);

% Partition test set of each known mixed population
patientsTest = cell(2,1,size(resistantPer,2));
patientsTest(2,1,:) = patientsComb(2,1,:); %Label mixed populations

for kPatientsTest = 1:size(resistantPer,2)
   patientsTest{1,1,kPatientsTest} = patientsComb{1,1,kPatientsTest}(iTest,:); 

end

%% Train model
% Hyperparameter optimization - Bayesian

clear mdlKNN
clear mdlSVM
clear mdlEns

tic

%% KNN
mdlKNN = fitcknn(patientsTrain(:,1:end-1),patientsTrain(:,end), 'Standardize', true,'OptimizeHyperparameters',{'Distance', 'NumNeighbors', 'DistanceWeight'}, ...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus', 'ShowPlots', 0, 'MaxObjectiveEvaluations', bayesCycle));

%Save
optimalHyperp{kMdls, 1, 1} = kMdls;
optimalHyperp{kMdls, 2, 1}  = convertCharsToStrings(mdlKNN.ModelParameters.Distance);
optimalHyperp{kMdls, 3, 1} = mdlKNN.ModelParameters.NumNeighbors;
optimalHyperp{kMdls, 4, 1} = convertCharsToStrings(mdlKNN.ModelParameters.DistanceWeight);



%% SVM
mdlSVM = fitcsvm(patientsTrain(:,1:end-1),patientsTrain(:,end), 'Standardize', true, 'OptimizeHyperparameters','auto', ...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus', 'ShowPlots', 0, 'MaxObjectiveEvaluations', bayesCycle));


%Save
optimalHyperp{kMdls, 1, 2} = kMdls;
optimalHyperp{kMdls, 2, 2}  = mdlSVM.ModelParameters.KernelScale;
optimalHyperp{kMdls, 3, 2} = mdlSVM.ModelParameters.BoxConstraint;


%% Ensemble 
mdlEns = fitcensemble(patientsTrain(:,1:end-1),patientsTrain(:,end), 'OptimizeHyperparameters',{'Method','NumLearningCycles','LearnRate'}, ...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus', 'ShowPlots', 0, 'MaxObjectiveEvaluations', bayesCycle));

% Save
optimalHyperp{kMdls, 1, 3} = kMdls;
optimalHyperp{kMdls, 2, 3}  = convertCharsToStrings(mdlEns.ModelParameters.Method);
optimalHyperp{kMdls, 3, 3} = mdlEns.ModelParameters.NLearn;
optimalHyperp{kMdls, 4, 3} = mdlEns.ModelParameters.LearnRate;


%% Test KNN model 
toc

    prediLabelsKNN = cell(size(patientsTest{1,1},1),size(resistantPer,2));
    
    % For each of the mixed populuations, make predictions using the KNN model 
    for kPredi = 1:size(resistantPer,2)
    
    % Use KNN model to make predictions of classification of patient
    prediLabelsKNN(:,kPredi) = predict(mdlKNN, patientsTest{1,1,kPredi}(:,1:end-1));
    
    %Initialize class performance with true classification labels
    KNNcp = classperf(table2cell(patientsTest{1,1,kPredi}(:,end)));
    
    %Compare predictions with true classification labels 
    classperf(KNNcp,prediLabelsKNN(:,kPredi));
    accuracyKNN(kMdls+1,kPredi) = KNNcp.CorrectRate;
    
    clear KNNcp
    end


%% Test SVM model 

    prediLabelsSVM = cell(size(patientsTest{1,1},1),size(resistantPer,2));
    
    % For each of the mixed populuations, make predictions using the SVM model 
    for kPredi = 1:size(resistantPer,2)
    
    % Use SVM model to make predictions of classification of patient
    prediLabelsSVM(:,kPredi) = predict(mdlSVM, patientsTest{1,1,kPredi}(:,1:end-1));
    
    %Initialize class performance with true classification labels
    SVMcp = classperf(table2cell(patientsTest{1,1,kPredi}(:,end)));
    
    %Compare predictions with true classification labels 
    classperf(SVMcp,prediLabelsSVM(:,kPredi));
    accuracySVM(kMdls+1,kPredi) = SVMcp.CorrectRate;
    
    clear SVMcp
    end

%% Test Ensemble model 

    prediLabelsEns = cell(size(patientsTest{1,1},1),size(resistantPer,2));
    
    % For each of the mixed populuations, make predictions using the Ens model 
    for kPredi = 1:size(resistantPer,2)
    
    % Use Ens model to make predictions of classification of patient
    prediLabelsEns(:,kPredi) = predict(mdlEns, patientsTest{1,1,kPredi}(:,1:end-1));
    
    %Initialize class performance with true classification labels
    Enscp = classperf(table2cell(patientsTest{1,1,kPredi}(:,end)));
    
    %Compare predictions with true classification labels 
    classperf(Enscp,prediLabelsEns(:,kPredi));
    accuracyEns(kMdls+1,kPredi) = Enscp.CorrectRate;
    
    clear Enscp
    end

 toc 
end

save("accuracyKNN.mat","accuracyKNN")
save("accuracySVM.mat","accuracySVM")
save("accuracyEns.mat","accuracyEns")

save("optimalHyperp.mat","optimalHyperp")

beep
pause(2)
beep
pause(2)
beep
