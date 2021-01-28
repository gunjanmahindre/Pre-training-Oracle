# A Pre-training Oracle for Predicting Distances in Social Networks

This research project was documented for Knowledge Discovery from Databases (KDD) 2021.


## Problem statement:
We try to predict missing distances in real-world social networks when only a fraction of distances are known.
Neural network is pre-trained with artificial data and the right parameters for this pre-training data needs to be identified.


## Approach:
* Neural networks need a lot of data to train on.
* So we use artificially generated data to pre-train on.
* For accurate prediction, this pre-training data must faithful to the target data.. but we do not know anything abou tthe target data.
* Building an Oracle to predict suitable Pre-training parameters.
* We demonstrate the Oracle for predicting distances in real-world social networks like Facebook, Emails, and train bomboing network. 


## Code:
The model has three stages. Folders "STAGE 0, I, and II" provide all the python files used to run the model. 
"trivial tests" gives codes for trivial cases to calculate base error such as:
* trivial 0 - when all missing values are replaced by 0
* trivial 1 - when all missing values are replaced by 1
* trained on observed - when the autoencoder is trained only on observed distances from the target network

## Data used:
Artificially generated Power law networks are used to pre-train the  autoencoder.
We simulate the Powerlaw networks using following function in python's networkx module (https://networkx.github.io/):
* G = nx.powerlaw_cluster_graph(N, m, p, seed=None)

We test the prediction of our model on following three real-world social networks:
* Facebook network
* Virgili Emails network
* Train Bombing network
These networks are available freely on KONECT (http://konect.cc/networks/). The distance matrices of these networks can be found in the folder "social network data used".
