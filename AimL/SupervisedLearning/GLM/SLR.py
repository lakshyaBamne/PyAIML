"""
    Module to implement Simple Linear Regression
    -> One regressor
    -> One response
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simple_linear_regression(regressor: np.array, response: np.array) -> dict:
    """
        Function to implement the simple linear regression model
        also output the evaluation criteria and learned model parameters
    """

