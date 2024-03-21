"""
    Module to implement Simple Linear Regression
    -> One regressor
    -> One response
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simple_linear_regression(regressor: np.ndarray, response: np.ndarray, **kwargs) -> dict:
    """
        Function to implement the simple linear regression model on the given data

        @regressor : numpy nd-array -> regressor variable
        @response: numpy nd-array -> response variable
    """
    #! number of regressors
    n = np.size(regressor, axis=0)

    #! first we need to calculate the required quantities
    # regressor mean
    # response mean
    # Sxy
    # Sxx
    xbar = np.sum(regressor)/n
    ybar = np.sum(response)/n
    Sxy = np.sum((regressor-xbar)*(response-ybar))
    Sxx = np.sum((regressor-2)**2)

    #! simple linear regression model parameters
    b1 = Sxy/Sxx
    b0 = ybar - b1*xbar

    #! estimated response values for the regressors
    yi_hat = b0 + b1*regressor

    #! now we calculate various evaluation parameters
    # SStotal
    # SSerr
    # SSreg
    SStot = np.sum((response-ybar)**2)
    SSerr = np.sum((response-yi_hat)**2)
    SSreg = np.sum((yi_hat-ybar)**2)

    #! Calculate the Coefficient of determination and correlation coeeficient
    Rsq = SSreg / SStot
    corr_coeff = np.sqrt(Rsq)

    if b1<0:
        corr_coeff = -corr_coeff
    elif b1 > 0:
        pass
    else:
        corr_coeff = 0

    reg_params = {
        "regressor_title" : kwargs["reg_title"],
        "response_title" : kwargs["res_title"],
        "b0" : b0,
        "b1" : b1,
        "SStot" : SStot,
        "SSerr" : SSerr,
        "SSreg" : SSreg,
        "Rsq" : Rsq,
        "corr_coeff" : corr_coeff
    }

    #! Visualize the model along with various judgement criteria
    plot_regression(regressor, response, reg_params)

    #! log the regression parameters to the terminal
    # log_regression_parameters(reg_params)

    return reg_params

def get_mean_responses(regressor: np.ndarray, response: np.ndarray) -> list[list]:
    """
        Function to get the means of responses at each regressor in the dataset
    """
    distinct_regressor = list(set(regressor))

    mean_responses = []
    for i in distinct_regressor:
        ms = [response[j] for j in range(len(response)) if regressor[j]==i]
        mean_responses.append(np.mean(ms))

    return [distinct_regressor, mean_responses]

def extract_random_samples(regressor: np.ndarray, response: np.ndarray, num: int) -> tuple[np.array]:
    """
        Function to extract random samples from the given population
        of regressors and responses
    """
    sample_indices = random.sample([i for i in range(len(regressor))], k=num)

    sample_regressors = np.array([regressor[i] for i in range(len(regressor)) if i in sample_indices])
    sample_responses = np.array([response[i] for i in range(len(regressor)) if i in sample_indices])

    return (np.array(sample_regressors), np.array(sample_responses))

def log_regression_parameters(regression_params: dict) -> None:
    """
        Function to print the various parameters along with their significances
    """
    print(regression_params)

def plot_regression(regressor: np.ndarray, response: np.ndarray, reg_params: dict) -> None:
    """
        Function to make a plot to visualise the regression model
        along with various judgement criteria for the model
    """

    fig = plt.figure(figsize=(10,10), layout="constrained")
    slr = fig.subplot_mosaic(
        [
            ['slr', 'slr'],
            ['slr', 'slr'],
            ['slr', 'slr'],
            ['params', 'eval']
        ]
    )

    #! add regression plot
    xi = np.linspace(min(regressor), max(regressor), num=1000, endpoint=True)
    yi = reg_params["b0"] + reg_params["b1"]*xi
    
    slr["slr"].scatter(regressor, response, color="red", label="Sample Data", s=[1 for _ in range(len(regressor))])
    slr["slr"].plot(xi, yi, color="black", label="SLR Line")
    slr["slr"].set_title("Simple Linear Regression")
    slr["slr"].set_xlabel(reg_params["regressor_title"])
    slr["slr"].set_ylabel(reg_params["response_title"])

    #! add regression parameters
    slr["params"].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    slr["params"].set_title("Regression Parameters")
    
    params_str = f"Intercept($b_0$)={round(reg_params['b0'],3)}\nSlope($b_1$)={round(reg_params['b1'],3)}\n\n\n"
    reg_line = f"Regression Line : {round(reg_params['b0'],3)} + {round(reg_params['b1'],3)}x"
    slr["params"].text(0.5, 0.5, params_str+reg_line, horizontalalignment="center", verticalalignment="center", fontsize=10)
    
    #! add evaluation criteria 
    slr["eval"].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    slr["eval"].set_title("Evaluation Criteria")

    eval_str = f"SStotal={round(reg_params['SStot'],3)}\nSSreg={round(reg_params['SSreg'],3)}\nSSerr={round(reg_params['SSerr'],3)}\n\n\n\nRsq={round(reg_params['Rsq'],3)}\nr={round(reg_params['corr_coeff'],3)}"
    slr["eval"].text(0.5, 0.5, eval_str, horizontalalignment="center", verticalalignment="center", fontsize=10)

    fig.legend()
    plt.show()



