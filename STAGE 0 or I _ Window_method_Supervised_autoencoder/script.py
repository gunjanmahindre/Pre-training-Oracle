"""
AUTOMATION SCRIPT: Supervided Pretrained Autoencoders for Inference in Networks

@author: Gunjan Mahindre
Version: 1.0
Date last modified: Sept. 27 2020

Description: 
    Run the main autoencoder code for various percentages of deletion 
    Run the code over "iter" number of iterations
    Calculates Mean error, Absolute Hop Distance Error (AHDE) averaged over all iterations.

"""

"""
1. create windows
iterate through the windows
2. create these networks.. do 3 Power law
3. train
4. test - only over observed entries - on virgili network

% = 60, 80, 90, 99, 99.5, 99.9

plot : each window as seperate graph.. 
cross check whether the window of actual average node degree performs best..

"""
# IMPORT MODULES REQUIRED

import os
import random
import numpy as np
import pandas as pd
# import RobustDeepAutoencoder as auto
from RobustDeepAutoencoder import *
from RobustDeepAutoencoder import RDAE
import DeepAE as DAE
import networkx as nx

# exit()



# 1. create windows

# windows = [
#     [1,2,3],
#     [3,4,5],
#     [5,6,7],
#     [7,8,9],
#     [9,10,11],
#     [11,12,13],
#     [13,14,15]
# ]

# windows = [
#     [5,7,9]
# ]



windows = [
    [1,5,10],
    [10,15,20],
    [20,25,30],
    [30,35,40],
    [40,45,50],
    [50,55,60],
    [60,65,70],
    [70,75,80],
    [80,85,90],
    [90,95,100],
    [1,2,3],
    [3,4,5],
    [5,6,7],
    [7,8,9],
    [9,10,11],
    [11,12,13],
    [13,14,15]
]

# windows = [
#     [1,2,3]
# ]


for ww in range(len(windows)):
	w = windows[ww]

	print ("current window:---------------------------------------------------------------------------------------- ", w)
	# create Directory for this window----------------
	directory = str(w[0]) + '_' + str(w[1]) + '_' + str(w[2])
	# Parent Directory path  
	parent_dir = "/content/drive/MyDrive/PhD work/Projects/parameter estimation/virgili more results/"
	# Path  
	path = os.path.join(parent_dir, directory)   
	# Create the directory   
	os.mkdir(path)  
	print("Directory '% s' created" % directory) 


	print ("RESULTS FOR SUPERVISED AUTOENCODERS ")

	mean_results = []
	abs_results = []
	m_STD_results = []
	a_STD_results = []

	# frac_list = [20, 80, 99.9]
	frac_list = [20, 40, 60, 80, 90, 99, 99.5, 99.9]

	for fraction in frac_list:
		[mean_err, abs_err, mean_std, abs_std] = main_code(fraction, w)
		print ("Fraction--------------------------------", fraction)

		mean_results.append(mean_err)
		abs_results.append(abs_err)
		m_STD_results.append(mean_std)
		a_STD_results.append(abs_std)





	# save each result in a text file
	filename = '/mean_error.txt'
	np.savetxt(path + filename, mean_results)
	filename = '/abs_error.txt'
	np.savetxt(path + filename, abs_results)
	filename = '/mean_STD.txt'
	np.savetxt(path + filename, m_STD_results)
	filename = '/abs_STD.txt'
	np.savetxt(path + filename, a_STD_results)

	print (frac_list)

exit()
