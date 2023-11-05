Folder "weights_correlation": 

File:

1. create_gabor_dict:
* contains the code to generate the Gabor dictionary used for the weights of convolutional layer 1
* 32 Gabors: 8 orientations, 4 phases 
* Gabors have a diameter of 11 pixels

2. draw_input_fields:
* used to calculate how much of the input the kernel of each CNN layer has access to
* draws these sizes into a chosen image from the dataset 

3. all_cnns_mean_correlation:
* includes the code to plot the correlation matrices seen in the written thesis
* includes statistical test
* includes code to plot every single correlation matrix of the 20 CNNs



In folder "weight visualization":

1. plot_as_grid:
* visualizes the weights and bias (optional)

2. plot_weights:
* loads model
* choose layer to visualize weights from (+ bias optionally)