"""
    Python script to demonstrate how to load a dataset using the 
    functions in the library and some tips on manipulating them
"""

# required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the function to load the required dataset
# there are a group of datasets which you can import
# for different problems different types of datasets are provided

# Supervised dataset
from AimL.Datasets.SupervisedDataset import load_practical

# Unsupervised dataset
from AimL.Datasets.UnsupervisedDataset import load_credit_details

# use the function to load the data set
# the function returns a Pandas DataFrame
df1 = load_practical(PLOT=True)

# you can add the PLOT option in the functions to visualize the imported data
df2 = load_credit_details()



