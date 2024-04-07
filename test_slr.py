"""
    Python script to demonstrate how to load a dataset using the 
    functions in the library and some tips on manipulating them
"""

# required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data set
from AimL.Datasets.SupervisedDataset import load_practical

# import the simple linear regression model
from AimL.SupervisedLearning.GLM.SLR import simple_linear_regression

# apply the model to the data
df = load_practical(PLOT=True)

x = np.array(df["x"])
y = np.array(df["y"])

#! Apply the Simple Linear Regression Model to the given data set
slr_params = simple_linear_regression(x, y, reg_title="x", res_title="y")


