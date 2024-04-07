# import the required python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from AimL.Datasets import GenerateClassificationData

FUNCTION = "POLYNOMIAL-3"
NUM = 1000

data = GenerateClassificationData(FUNCTION, NUM)

data.visualize_generated_data()

