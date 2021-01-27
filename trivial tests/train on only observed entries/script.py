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

print ('here 0')



import os
import random
import numpy as np
import pandas as pd
# import RobustDeepAutoencoder as auto
from RobustDeepAutoencoder import *
from RobustDeepAutoencoder import RDAE
import DeepAE as DAE
import networkx as nx



# 1. create windows


# 1. create windows

windows = [
	[12,13,14]
]

print ('here 1')


for ww in range(len(windows)):
	w = windows[ww]

	print ("current window:---------------------------------------------------------------------------------------- ", w)
	# create Directory for this window----------------
	directory = str(w[0]) + '_' + str(w[1]) + '_' + str(w[2])
	# Parent Directory path  
	# parent_dir = "C:\\Users\\gunjan\\Google Drive\\PhD work/Projects\\parameter estimation\\virgili\\"
	parent_dir = "/content/drive/MyDrive/PhD work/Projects/parameter estimation/bigger facebokk results/train on only observed entries/"
	# parent_dir = "/content/drive/MyDrive/PhD work/Projects/parameter estimation/virgili/"
	
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

	frac_list = [20, 40, 60, 80, 90, 99, 99.5, 99.9]
	# frac_list = [20]

	print ('here 2')

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