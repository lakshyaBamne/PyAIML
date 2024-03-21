"""
    Module to load the data set named Credit Details as a dataframe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_credit_details(**kwargs) -> pd.DataFrame:
    """
        Function to load the unsupervised dataset - credit details
    """

    path = "Aiml/Datasets/unsupervised/CreditDetails.csv"

    # since this dataset is stored in a csv file we just read it
    # using pandas inbuilt function
    df = pd.read_csv(path)

    # visualise the data
    #! we can later change this to visualize the data better
    if "PLOT" in kwargs.keys():
        plot_dataset(df)

    return df

def plot_dataset(df: pd.DataFrame) -> None:
    """
        Function to make a plot of the credit card data in a scatterplot matrix
    """
    pd.plotting.scatter_matrix(df)
    
    plt.show()


