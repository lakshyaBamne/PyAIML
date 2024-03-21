"""
    Module to load the data set named practical as a dataframe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_practical(**kwargs) -> pd.DataFrame:
    """
        Function loads the practical data set as a data frame
    """

    path = "AimL/Datasets/supervised/practical.txt"

    with open(path, "r") as f:
        lines = f.readlines()

    dataset_title = "Practical Data"
    head = ("x", "y")
    x = []
    y = []

    for line in lines[1:]:
        xtemp, ytemp = list(map(float, line[:-1].split("\t")))
        x.append(xtemp)
        y.append(ytemp)

    #! plot the data set if required
    if "PLOT" in kwargs.keys():
        plot_dataset(x, y, dataset_title, head)

    # convert the data to a data frame and return to the user
    df_dict = {
        f"{head[0]}" : np.array(x),
        f"{head[1]}" : np.array(y)
    }
    
    df = pd.DataFrame(df_dict)

    return df

def plot_dataset(x: list, y: list, title: str, head: tuple[str]) -> None:
    """
        Function to plot the loaded dataset
    """

    # show the data to the user using a scatterplot
    fig, slr = plt.subplots(figsize=(10,10))

    slr.scatter(x, y, color="red", label="Population Points", s=[1 for _ in range(len(x))])
    slr.set_title(title)
    slr.set_xlabel(head[0])
    slr.set_ylabel(head[1])
    
    slr.legend()

    plt.show()
