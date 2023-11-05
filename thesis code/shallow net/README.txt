Folder shallow net:

1. config.json:
* includes all cnn parameters and configurations 
* example for architecture: 32-8-8 (c1-c2-c3)

2. corner_loop_final.sh
* bash script to train the 20 cnns of one cnn architecture 

Folder functions: 
* contains the files do build the cnn, set the seeds, create a logging file, train and test the cnns 
* based on ---> Github: https://github.com/davrot/kk_contour_net_shallow.git