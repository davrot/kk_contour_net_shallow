Folder orientation_tuning:


1. orientation_tuning_curve:
* generates the original tuning curve by convolving the Gabor patches with the weight matrices of C1
* Gabor patches file: gabor_dict_32o_8p.py

2. fitkarotte:
* implements the fitting procedure of the 3 von Mises functions
* plots the fitted tuning curves 

3. fit_statistics:
* contains all statistical test for the 20 trained CNNs of each stimulus condition
* calls the 'plot_fit_statistics' function to plot the data