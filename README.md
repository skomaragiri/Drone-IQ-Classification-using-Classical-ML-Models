# IQ Classification Classical Models
Building classical machine learning models for IQ and UAV classification.

IQ samples: An IQ sample is a pair of digital values (In-phase and Quadrature) that represent a radio signal's amplitude and phase at a specific moment, capturing its complete information (amplitude, phase, frequency) in two orthogonal components. 

These models are able to classify different drones using their emitted radio frequncy signals—in the form of IQ samples. 

## The models 
Each of the models has there own file in the root directory (this directory). Each model uses similar functions stored in the ./helper/preprocessing.py file.
1. Gaussian Naive Bayes 
2. K-Nearest Neighbors 
3. Random Forest
4. XG Boosting Model 