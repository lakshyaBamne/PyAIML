"""
    Module to generate a classification data set based on a given function
"""
import numpy as np
import random
import matplotlib.pyplot as plt

class GenerateClassificationData:
    """
        Class contains randomly generated data for classification
    """

    def __init__(self, function: str, num_points: int):
        self.function = function
        self.n = num_points

        # generate random data 
        self.generate_random_points()

        # generate labels for the random points based on the given function
        if self.function[:10] == "POLYNOMIAL":
            # for polynomial classification function user needs to input
            # POLYNOMIAL-<DEGREE> :- where DEGREE is any integer such that overflow is not encountered
            self.generate_polynomial_data(int(self.function[11:]))
        else:
            raise ValueError
        
    def generate_random_points(self):
        self.x = list(np.random.uniform(-10, 10, size=self.n))
        self.y = list(np.random.uniform(-10, 10, size=self.n))

    def get_polynomial_value(self, coeff: list, x: float):
        val = 0
        for i in range(len(coeff)):
            val += coeff[i]*(x**i)
        return val

    def generate_polynomial_data(self, degree: int):
        # we need some coefficients to represent a polynomial
        self.coeff = list(np.random.uniform(size=degree+1))

        # now we can add true labels for the random data
        self.true_labels = [
            1 if self.y[i]-self.get_polynomial_value(self.coeff, self.x[i])>=0 
            else 0 
            for i in range(self.n)
        ]

    def visualize_generated_data(self):
        
        # extract data for class-1
        class1_x = [self.x[i] for i in range(self.n) if self.true_labels[i]==1]
        class1_y = [self.y[i] for i in range(self.n) if self.true_labels[i]==1]

        # extract data for class-2
        class2_x = [self.x[i] for i in range(self.n) if self.true_labels[i]==0]
        class2_y = [self.y[i] for i in range(self.n) if self.true_labels[i]==0]

        # generate data for the function itself
        func_x = list(np.linspace(-10, 10, num=1000, endpoint=True))
        func_y = [self.get_polynomial_value(self.coeff, x) for x in func_x]

        sizes_class1 = [1 for _ in range(len(class1_x))]
        sizes_class2 = [1 for _ in range(len(class2_x))]

        # plot the data
        fig, ax = plt.subplots()

        ax.scatter(class1_x, class1_y, color="red", label="Class-1", s=sizes_class1)
        ax.scatter(class2_x, class2_y, color="blue", label="Class-2", s=sizes_class2)
        ax.plot(func_x, func_y, color="black", label="True classifying curve")

        ax.set_xlim(left=-10, right=10)
        ax.set_ylim(bottom=-10, top=10)

        ax.set_title(self.function)

        ax.legend()
        plt.show()


