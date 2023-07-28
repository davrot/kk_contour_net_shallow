import torch
import scipy
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import logging


# this code calculates the mean number of Gabor patches inside a stimulus for each class 

logging.basicConfig(filename='AngularAvrg.txt', filemode='w', format='%(message)s', level=logging.INFO)

avg_avg_size: list = []
n_contours = 0
x_range = [140, 940]
y_range = [140, 940]
for i in range(10):
  path = f"/data_1/kk/StimulusGeneration/Alicorn/Angular/Ang0{i}0_n10000"
  files = glob.glob(path + os.sep + "*.mat")
  
  n_files = len(files)
  print(f"Going through {n_files} contour files...")
  logging.info(f"Going through {n_files} contour files...")
  varname=f"Table_intr_crn0{i}0"
  varname_dist=f"Table_intr_crn0{i}0_dist"
  for i_file in range(n_files):
      # get path, basename, suffix...
      full = files[i_file]
      path, file = os.path.split(full)
      base, suffix = os.path.splitext(file)
  
      # load file
      print(full)
      mat = scipy.io.loadmat(full)
      if "dist" in full:
          posori = mat[varname_dist]
      else:
          posori = mat[varname]
      
      sec_dim_sizes = []      #[posori[i][0].shape[1] for i in range(posori.shape[0])]
      
      for s in range(posori.shape[0]):
        # Extract the entry
        entry = posori[s][0]
        
        # Get the x and y coordinates
        x = entry[1]
        y = entry[2]
        
        # Find the indices of the coordinates that fall within the specified range
        idx = np.where((x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1]))[0]
        
        # Calculate the size of the second dimension while only considering the coordinates within the specified range
        sec_dim_size = len(idx)
        
        # Append the size to the list
        sec_dim_sizes.append(sec_dim_size)

      avg_size = np.mean(sec_dim_sizes)
      print(f"Average 2nd dim of posori: {avg_size}")
      logging.info(f"Average 2nd dim of posori: {avg_size}")
      avg_avg_size.append(avg_size)
      n_contours += posori.shape[0]
      
      print(f"...overall {n_contours} contours so far.")
      logging.info(f"...overall {n_contours} contours so far.")


# calculate avg number Gabors over whole condition
overall = np.mean(avg_avg_size)
print(f"OVERALL average 2nd dim of posori: {overall}")
logging.info(f"OVERALL average 2nd dim of posori: {overall}")