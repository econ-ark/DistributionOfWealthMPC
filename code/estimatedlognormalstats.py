import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm
import math


class EstLogNormalDist:
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

        self.mu = np.log(self.center_value) - 0.5 * self.spread_value**2
        self.sigma = self.spread_value

    def parse_line(self):
        """
        Function which takes a text file containing the output of a standard structural estimation
        exercise using the HARK toolkit (specifically, the DistributionOfWealthMPC repository)
        and reads the first line which describes the solution to the problem.

        Note: This is specific to the assumption of a uniform distribution of
        the specified preference parameter being estimated.
        """
        with open(self.results, "r") as file:
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

    def compute_moments(self):
        """
        Uses the output of the estimation to compute up to the fourth moment of the lognormal distribution.
        Store the result of each one of these computations as an attribute of the class to be called later.
        """
        self.mean = math.exp(self.mu + (self.sigma**2) * 0.5)
        self.variance = (math.exp(self.sigma**2) - 1) * math.exp(
            2 * self.mu + (self.sigma) ** 2
        )
        self.std_dev = math.sqrt(self.variance)
        self.skewness = (math.exp(self.sigma**2) + 2) * (
            math.sqrt(math.exp(self.sigma**2) - 1)
        )
        self.gen_kurtosis = (
            math.exp(4 * self.sigma**2)
            + 2 * math.exp(3 * self.sigma**2)
            + 3 * math.exp(2 * self.sigma**2)
            - 6
        )
        self.excess_kurtosis = self.gen_kurtosis - 3

        return [self.mean, self.std_dev, self.skewness, self.excess_kurtosis]

    def show_moments(self):
        """
        Provides an output of the first four moments for the estimated disrtribution of the
        parameter of interest.
        """

        values = self.compute_moments()
        keys = ["Mean", "Standard Deviation", "Skewness", "Kurtosis"]
        moments_dict = dict(zip(keys, values))

        return moments_dict

    def graph(self):
        # Generate random samples from lognormal distribution
        samples = np.random.lognormal(self.mu, self.sigma, 100000)

        # Plot the histogram of the samples
        plt.hist(samples, bins=self.num_types, density=True, alpha=0.7, color="skyblue")

        # Plot the probability density function (PDF) of the lognormal distribution
        x = np.linspace(0.90, np.max(samples), 1000)
        pdf = lognorm.pdf(x, s=self.sigma, scale=np.exp(self.mu))
        plt.plot(x, pdf, "r", linewidth=2)

        # Set the labels and title of the plot
        plt.xlabel("Value")
        plt.ylabel("Probability Density")
        plt.title("Lognormal Distribution")

        # Show the plot
        plt.show()
