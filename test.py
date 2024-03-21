# import the required python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data set
from AimL.Datasets import load_practical

# import the simple linear regression model
from AimL.SupervisedLearning.GLM.SLR import simple_linear_regression

# apply the model to the data
df = load_practical(PLOT=True)

x = np.array(df["x"])
y = np.array(df["y"])

slr_params = simple_linear_regression(x, y, reg_title="x", res_title="y")

