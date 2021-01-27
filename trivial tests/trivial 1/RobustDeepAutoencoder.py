"""
Supervided Pretrained Autoencoders for Inference in Networks

@author: Gunjan Mahindre, Rasika Karkare, Chong Zhou, Randy Paffenroth
Version: 1.0
Changes from unsupervised: 
    Train function takes the actual values of missing entries into account while training.
    Functions have been created.
    Structure of the script has been changed.
Date last modified: 07/24/2020

Description: 
    Trains the supervised Autoencoder using several networks which are sparsely sampled.
    Tests on a network it has not been trained on.
    Calculates Mean error, Absolute Hop Distance Error (AHDE), std. dev. for mean error and std. dev. for AHDE.

"""

# IMPORT MODULES REQUIRED 

import os
import random
import numpy as np
import pandas as pd
import DeepAE as DAE
import networkx as nx
from math import sqrt
import l1shrink as SHR 
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy.linalg as nplin
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import mean_absolute_error

np.random.seed(123)

# FUNCTIONS AND CLASS DEFINITIONS

class RDAE(object):
    """
    @author: Chong Zhou
    2.0 version.
    complete: 10/17/2016
    version changes: move implementation from theano to tensorflow.
    3.0
    complete: 2/12/2018
    changes: delete unused parameter, move shrink function to other file
    update: 03/15/2019
        update to python3 
    Des:
        X = L + S
        L is a non-linearly low rank matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_1
        Use Alternating projection to train model
    """
    def __init__(self, sess, layers_sizes, lambda_=1.0, error = 1.0e-7):
        """
        sess: a Tensorflow tf.Session object
        layers_sizes: a list that contains the deep ae layer sizes, including the input layer
        lambda_: tuning the weight of l1 penalty of S
        error: converge criterior for jump out training iteration
        """
        self.lambda_ = lambda_
        self.layers_sizes = layers_sizes
        self.error = error
        self.errors=[]
        self.AE = DAE.Deep_Autoencoder( sess = sess, input_dim_list = self.layers_sizes)

    def fit(self, X, Y, sess, learning_rate=0.15, inner_iteration = 50,
            iteration=20, batch_size=50, verbose=False):
        ## The first layer must be the input layer, so they should have same sizes.
        assert X.shape[1] == self.layers_sizes[0]

        ## initialize L, S, mu(shrinkage operator)
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        self.hadamard_train = np.array(hadamard_train)
        self.cost = list()

        mu = (X.size) / (4.0 * nplin.norm(X,1))
        print ("shrink parameter:", self.lambda_ / mu)
        LS0 = self.L + self.S

        XFnorm = nplin.norm(X,'fro')
        if verbose:
            print ("X shape: ", X.shape)
            print ("L shape: ", self.L.shape)
            print ("S shape: ", self.S.shape)
            print ("mu: ", mu)
            print ("XFnorm: ", XFnorm)

        for it in range(iteration):
            if verbose:
                print ("Out iteration: " , it)
            ## alternating project, first project to L
            #self.L = X - self.S
            ## Using L to train the auto-encoder
            self.cost.append(self.AE.fit(X = X, Y = Y, sess = sess, S =self.S, h = self.hadamard_train,
                                    iteration = inner_iteration,
                                    learning_rate = learning_rate,
                                    batch_size = batch_size,
                                    verbose = verbose))
            ## get optmized L
            self.L = self.AE.getRecon(X = X, sess = sess)
            ## alternating project, now project to S
            self.S = SHR.shrink(self.lambda_/np.min([mu,np.sqrt(mu)]), (X - self.L).reshape(X.size)).reshape(X.shape)

            ## break criterion 1: the L and S are close enough to X
            c1 = nplin.norm(X - self.L - self.S, 'fro') / XFnorm
            ## break criterion 2: there is no changes for L and S 
            c2 = np.min([mu,np.sqrt(mu)]) * nplin.norm(LS0 - self.L - self.S) / XFnorm

            if verbose:
                print ("c1: ", c1)
                print ("c2: ", c2)

            if c1 < self.error and c2 < self.error :
                print ("early break")
                break
            ## save L + S for c2 check in the next iteration
            LS0 = self.L + self.S
            
        return self.L , self.S, self.cost
    
    def transform(self, X, sess):
        #L = X - self.S
        return self.AE.transform(X = X, sess = sess)
    
    def getRecon(self, X, sess):
        return self.AE.getRecon(X, sess = sess)


# define a function to sample from the given hop distance matrix----

def delete_percentage(fraction, data_matrix):

    '''
    INPUT:
    fraction: percentage to be deleted
    data_matrix: hop distance matrix from which we need to delete the given percentage of entries

    OUTPUT:
    data_matrix: the hop distance matrix with only (100-fraction)% of sampled entries. The missing entries are replaced by 0
    '''

    [Rn, Cn] = data_matrix.shape
    uT = []
    for i in range(Cn):
        for j in range(i + 1, Cn):
            uT.append(data_matrix[i, j])
    # calculate entries to be deleted
    rem_num = ((len(uT)) * fraction / 100)  # total number of entries to be removed
    # has to be an integer value
    rem_num = int(rem_num)
    # select random elements from the upper triangle:
    ind = np.random.choice(len(uT), rem_num, replace=False)
    # make these indices -1
    for i in ind:
        uT[i] = 0  # now place these values back in the upper triangle:

    # -------delete the symmetric entry for undirected networks. Comment it for directed networks ------
    p = 0
    for i in range(Cn):
        for j in range(i + 1, Cn):
            data_matrix[i, j] = uT[p]
            # delete symmetric entris - applicable for undirected graphs
            if data_matrix[i, j] == 0:
                data_matrix[j, i] = 0
            p += 1
    #---------------------------------------------------------------------------------------------------


    return data_matrix

# MAIN FUNCTION
def main_code(fraction, w):
# if __name__ == "__main__":

    print ("inside main_code")




    # load data for testing---REAL WORLD NETWORK
    # path = 'C:/Users/gunjan/Google Drive/PhD work/data/undirected networks/virgili emails/'
    # for VIRGILI NETWORK-----------------------------------------------
    path = '/content/drive/MyDrive/PhD work/data/undirected networks/facebook/'
    data_test = np.loadtxt(path + 'dHp.txt')
     # for VIRGILI NETWORK-----------------------------------------------


    # # for FACEBOOK NETWORK--------------------------------------------
    # G = nx.read_edgelist('/content/drive/MyDrive/PhD work/data/undirected networks/facebook/edges.txt', create_using = nx.Graph(), nodetype = int)
    # # from scipy.sparse.csgraph import dijkstra
    # A = nx.adjacency_matrix(G)
    # data_test = np.array(dijkstra(A))
    # # for FACEBOOK NETWORK--------------------------------------------

    data_test_original = data_test.copy()



    # --------------define hyperparameters-------------------------
    # [R,C] = [1133,1133]
    [R,C] = [4039,4039]
    [R,C] = data_test.shape
    hidden_layer_size = 50
    learning_rate_alpha = 0.001
    batch_size = 1
    inner_iteration = 20
    #-------------------------------------------------------------


    # # for TESTING performance on training data variation:
    # par4 = w[3]
    # print ("testing parameter: ", par4)
    # # for TESTING
    # GT = nx.powerlaw_cluster_graph(R, par4, 0.4, seed=None)
    # # for TESTING
    # A = nx.adjacency_matrix(GT)
    # data_test = np.array(dijkstra(A))
    # data_test_original = data_test.copy()


    

    par1 = w[0]
    par2 = w[1]
    par3 = w[2]




    print ("training parameters: ", par1, par2, par3)

    # 2. create networks

    #------- BARABASI -------------
    # G1 = nx.barabasi_albert_graph(R, par1, seed=None)
    # G2 = nx.barabasi_albert_graph(R, par2, seed=None)
    # G3 = nx.barabasi_albert_graph(R, par3, seed=None)

    # A = nx.adjacency_matrix(G1)
    # data1 = np.array(dijkstra(A))
    # data1_ori = data1.copy()

    # A = nx.adjacency_matrix(G2)
    # data2 = np.array(dijkstra(A))
    # data2_ori = data2.copy()

    # A = nx.adjacency_matrix(G3)
    # data3 = np.array(dijkstra(A))
    # data3_ori = data3.copy()

    # #------- POWER LAW -------------
    # G4 = nx.powerlaw_cluster_graph(R, par1, 0.1, seed=None)
    # G5 = nx.powerlaw_cluster_graph(R, par2, 0.5, seed=None)
    # G6 = nx.powerlaw_cluster_graph(R, par3, 0.9, seed=None)

    
    # A = nx.adjacency_matrix(G4)
    # data4 = np.array(dijkstra(A))
    # data4_ori = data4.copy()

    # A = nx.adjacency_matrix(G5)
    # data5 = np.array(dijkstra(A))
    # data5_ori = data5.copy()

    # A = nx.adjacency_matrix(G6)
    # data6 = np.array(dijkstra(A))
    # data6_ori = data6.copy()


    print (data_test)
    print (data_test_original)

    print (data_test_original.shape)

    # delete the given percentage:
    # fraction = 60

    # tf.Session() initiates a TensorFlow Graph object in which tensors are processed through operations.
    # The "with" block terminates the session as soon as the operations are completed. 
    with tf.compat.v1.Session() as sess:

        print ("inside session")
    
        # create object rae
        rae = RDAE(sess = sess, lambda_= 500000, layers_sizes=[R,hidden_layer_size])

        global hadamard_train 

        # # Process data1---------------------------------------------------------------
        # data1 = delete_percentage(fraction, data1)

        # # Hadamard part
        
        # hadamard_train = np.ones(data1.shape)
        # hadamard_train = np.where(data1 == 0, 0 , hadamard_train)
        # hadamard_train = pd.DataFrame(hadamard_train)

        # print ("before fit function")

        # # train the autoencoder with data1
        # L, S, cost = rae.fit(data1, data1_ori ,sess = sess, learning_rate=learning_rate_alpha, batch_size =batch_size,inner_iteration =inner_iteration,iteration=1, verbose=True)

        # print ("after fit function")

        # # Process data2---------------------------------------------------------------
        # data2 = delete_percentage(fraction, data2)

        # hadamard_train = np.ones(data2.shape)
        # hadamard_train = np.where(data2 == 0, 0, hadamard_train)
        # hadamard_train = pd.DataFrame(hadamard_train)

        # L, S, cost = rae.fit(data2, data2_ori, sess = sess, learning_rate=learning_rate_alpha, batch_size =batch_size,inner_iteration =inner_iteration,iteration=1, verbose=True)


        # # Process data3---------------------------------------------------------------
        # data3 = delete_percentage(fraction, data3)

        # hadamard_train = np.ones(data3.shape)
        # hadamard_train = np.where(data3 == 0, 0, hadamard_train)
        # hadamard_train = pd.DataFrame(hadamard_train)

        # L, S, cost = rae.fit(data3, data3_ori, sess = sess, learning_rate=learning_rate_alpha, batch_size =batch_size,inner_iteration =inner_iteration,iteration=1, verbose=True)


        # # Process data4---------------------------------------------------------------
        # data4 = delete_percentage(fraction, data4)

        # hadamard_train = np.ones(data4.shape)
        # hadamard_train = np.where(data4 == 0, 0, hadamard_train)
        # hadamard_train = pd.DataFrame(hadamard_train)

        # L, S, cost = rae.fit(data4, data4_ori, sess = sess, learning_rate=learning_rate_alpha, batch_size =batch_size,inner_iteration =inner_iteration,iteration=1, verbose=True)
        # print ("after training on nw1: cost = ", cost)

        # # Process data5---------------------------------------------------------------
        # data5 = delete_percentage(fraction, data5)

        # hadamard_train = np.ones(data5.shape)
        # hadamard_train = np.where(data5 == 0, 0, hadamard_train)
        # hadamard_train = pd.DataFrame(hadamard_train)

        # L, S, cost = rae.fit(data5, data5_ori, sess = sess, learning_rate=learning_rate_alpha, batch_size =batch_size,inner_iteration =inner_iteration,iteration=1, verbose=True)
        # print ("after training on nw2: cost = ", cost)


        # # Process data6---------------------------------------------------------------
        # data6 = delete_percentage(fraction, data6)

        # hadamard_train = np.ones(data6.shape)
        # hadamard_train = np.where(data6 == 0, 0, hadamard_train)
        # hadamard_train = pd.DataFrame(hadamard_train)

        # L, S, cost = rae.fit(data6, data6_ori, sess = sess, learning_rate=learning_rate_alpha, batch_size =batch_size,inner_iteration =inner_iteration,iteration=1, verbose=True)
        # print ("after training on nw3: cost = ", cost)




        # Process test data---------------------------------------------------------------
        data_test = delete_percentage(fraction, data_test)

        hadamard_test = np.ones(data_test.shape)
        hadamard_test = np.where(data_test == 0,0, hadamard_test)
        hadamard_test = pd.DataFrame(hadamard_test)

        data_test = pd.DataFrame(data_test)
        # data_test = np.array(data_test)

        # # reconstruct using autoencoder to predict entries
        # h = rae.transform(data_test, sess=sess)
        # # R : reconstructed matrix
        # R = rae.getRecon(data_test, sess=sess)

        # print(type(R))
        # exit()
        # R = pd.DataFrame(R)
        R = data_test.copy()


        data_test_original = pd.DataFrame(data_test_original)

        # # Correction code:
        # """
        # All diagonal entries are set to 0
        # All off diagonal entries predicted <=1 are set to 1
        # """
        for i in range(len(R)):
            for j in R.columns:
                if i != j:
                	if R.iloc[i, j] <= 1.0:
                		R.iloc[i, j] = 1
                	else:
                		continue

        for i in range(len(R)):
            for j in R.columns:
                if i == j:
                    R.iloc[i, j] = 0
                else:
                    continue



        R = pd.DataFrame(R)
        data_test = pd.DataFrame(data_test)


        data_test_original = pd.DataFrame(data_test_original)

        # # # save the recovered matrix
        # ss = str(w[0]) + '_' + str(w[1]) + '_' + str(w[2])
        # if fraction == 40 or fraction == 99:
        #     R.to_csv('/content/drive/MyDrive/PhD work/Projects/parameter estimation/virgili more results/stage 2/' + ss + '/R_' + str(fraction) + '.csv', index = False)
        # # save original Facebook network
        # data_test_original.to_csv('./Original_fb.csv')




##################################################################
        print ("-------------- Calculating error only for unobserved entries--------------------")

        [r,c] = data_test.shape
        # vectorize matrices - placeholders
        hop = []
        ori = []
        # meane = []
        # abse = []


        hadamard_test = np.array(hadamard_test)

        ##################################################################
        # trivial 1 test:
        # replace all missing entries with 1
        for i in range(r):
            for j in range(c):
                if hadamard_test[i,j] == 0:
                    R.iloc[i,j] = 1



        p = 0
        for i in range(r):
          for j in range(c):
            if hadamard_test[i,j] == 0:   # considers error on only unobserved entries
                hop.append(R.iloc[i,j])
                ori.append(data_test_original.iloc[i,j])
            p = p+1

        # #  mean and absolute hop error calculation----------------
        # mean_err: mean error
        # abs_err: AHDE - Absolute hop distance error
        hop = np.array(hop)
        ori = np.array(ori)
        x = np.round(hop-ori)

        # print ("numerator:", np.sum(abs(x)))
        # print ("sum of unobserved entries:", np.sum(ori))
        # print ("b: total unobserved entries:", len(ori))

        mean_err = (np.sum(abs(x)))/(np.sum(ori))        
        mean_err = mean_err*100
        mean_std = np.std(abs(x))

        abs_err = (np.sum(abs(x)))/(len(ori))  # divided by the number of unobserved entries
        abs_std = np.std(abs(x))

        print (mean_err, abs_err, mean_std, abs_std)

        return (mean_err, abs_err, mean_std, abs_std)