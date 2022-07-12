# Comparing Quantum and Classical ML algorithm to determine the advantage in Games

## Introduction
Here demonstrated is the nearest centroid classification via implementations in 1) a fully classical model and 2) a model taking advantage of quantum distance estimation. 
Nearest centroid clustering was used to identify game state advantage in generated game boards, 
with comparable applications beyond the given application, from rapid medical classification to 
sentiment analysis. The quantum model is compared with the classical model to evaluate the 
efficacy of the quantum model in comparison to that of the classical model. 
Additionally, this comparison is gameified, pitting human, classical, and quantum 
classifications against each other in an interactive game.


## Basic Principle
Nearest centroid classification is a supervised learning task which takes labelled training data and computes a best centroid of the data in its feature space. This centroid is mapped to our training data via Euclidean distance in both the classical and quantum models.
In the quantum model, this inner product is calculated using a quantum algorithm that provides an exponential speedup from O(n) to O(log(n)).
