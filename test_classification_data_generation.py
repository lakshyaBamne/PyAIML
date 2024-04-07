"""
    Script demonstrates how to generate random classification data
    for use in various algorithms of Machine Learning and Deep Learning
"""

# required libraries
import numpy as np
import matplotlib.pyplot as plt

# import the class to generate data
from AimL.Datasets import GenerateClassificationData

#! generate data and visualize it

# variables to state the true curve generating the classification data
# and number of points to be generated
FUNCTION = "POLYNOMIAL-7"
NUM = 1000

# instantiate the class and generate data stored as attributes in the object
data = GenerateClassificationData(FUNCTION, NUM)

data.visualize_generated_data()
