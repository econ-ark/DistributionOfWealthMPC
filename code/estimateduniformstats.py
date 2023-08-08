import matplotlib.pyplot as plt 
import numpy as np
from HARK.distribution import Uniform
import re
import math

class EstUniformDist:
    """
    Base class to be ran following the structural estimation procedure for a HA 
    macro model using the estimation.py file from the DistributionOfWealthMPC repository.
    This class will extract the estimated values describing the distribution
    for a given parameter which best matches the distribution of wealth from the data. It
    then creates an object and attaches these estimated values to it. 
    """
    def __init__(self, results, num_types=7):
        self.num_types = num_types
        self.results = results if results is not None else ""

        self.center_value, self.spread_value = self.parse_line()

        self.bottom = self.center_value - self.spread_value
        self.top = self.center_value + self.spread_value

    def parse_line(self):
        """
        Function which takes a text file containing the output of a standard structural estimation
        exercise using the HARK toolkit (specifically, the DistributionOfWealthMPC repository)
        and reads the first line which describes the solution to the problem. 

        Note: This is specific to the assumption of a uniform distribution of 
        the specified preference parameter being estimated.
        """
        with open(self.results, 'r') as file:
            line = file.readline()
            center_index = line.find("center=")
            spread_index = line.find("spread=")
        
            if center_index == -1 or spread_index == -1:
                return None
        
            center_start = center_index + len("center=")
            comma_index = line.find(",", center_start)
        
            spread_start = spread_index + len("spread=")
        
            result_center = line[center_start:comma_index]
            result_spread = line[spread_start:].strip()
        
            return float(result_center), float(result_spread)
        

    def graph(self):
        """
        After extracting and assigning the values from the estimation, this class is defined so that
        it creates an object which inherits these attributes. It then creates a vector of values for 
        the estimated distribution given the number of types of households defined in the estimation 
        procedure (in general, this is 7). From there, a graph of this distribution is produced. 

        Note: for the uniform distirbution, this is pretty trivial. However, since other distributional
        assumptions may be of interest, seeing what those graphs may look like is important. This base 
        case is a general computational exercise to be extended to those more interesting cases.
        """

        est_dstn = (
                        Uniform(self.bottom, self.top)
                        .discretize(self.num_types)
                        .atoms.flatten()
)

        plt.gca().set_ylim([0, 2])
        plt.hist(est_dstn,bins=self.num_types)
        plt.show()


    def compute_moments(self):
        """
        Uses the output of the estimation to compute up to the fourth moment of the uniform distribution.
        Store the result of each one of these computations as an attribute of the class to be called later.
        """
        self.mean = (1/2) * (self.bottom + self.top)
        self.variance = (1/12) * (self.top - self.bottom)**2
        self.std_dev = math.sqrt(self.variance)
        self.skewness = 0
        self.gen_kurtosis = 9/5
        self.excess_kurtosis = self.gen_kurtosis - 3

        return [self.mean, self.std_dev, self.skewness, self.excess_kurtosis]

    def show_moments(self):
        """
        Provides an output of the first four moments for the estimated disrtribution of the 
        parameter of interest. 
        """

        values = self.compute_moments()
        keys = ["Mean", "Standard Deviation", "Skewness", "Kurtosis"]
        moments_dict = dict(zip(keys,values))

        return moments_dict
        
