# classifyMixedPopulations

April 22, 2023.
Written by Chau Ly.
The general design of the code is further detailed in the publication
referenced below: 
Ly C, Ogana H, Kim HN, Hurwitz S, Deeds EJ, Kim YM, Rowat AC."Altered
physical phenotypes of leukemia cells that survive chemotherapy
treatment." Integrative Biology (2023). 

This code is divided into two parts:
(1) Randomize single-cell physical phenotypes (all features) using experimental
qc-DC data into computationally generated mixed populations containing
varying proportions of "resistant" cells 
(2) Train machine learning models - kNN, SVM, and ensemble classifier of
decision trees - and complete Bayesian optimization using hold out of trained 
mixed populations of varying proportions of "resistant" cells. 
Test the trained models and assess classification accuracy of known mixed populations.
Repeat pipeline for 10 Monte Carlo iterations. 
