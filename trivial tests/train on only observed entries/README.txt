script.py:
	is the main script that runs everything. 


RobustDeepAutoencoder.py:
	has code for loading the networks, autoencoder class definition, training the autoencoder, error calculation
	The delete_percentage(fraction, data_matrix) function in this file will vary for directed and undirected graphs.
	Undirected graphs:
		lines 
		    if data_matrix[i, j] == 0:
			data_matrix[j, i] = 0
		 should not be commented. i.e., for every (i,j) entry, we also delete (j,i) entry.
	Directed graphs:
		these lines should be commented.


DeepAE.py:
	has autoencoder initialization
