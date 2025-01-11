import numpy as np
import matplotlib.pyplot as plt


class SPD:
    def __init__(self, dates):
        """
        Represents a summed probability density (SPD).

        Parameters:
        - dates: A list of `Date` objects to sum.
        """
        self.dates = dates
        self.summed = None

        for date in self.dates:
            date.calibrate()

    def sum(self):
        """
        Sums the probability densities of all calibrated dates.
        """
        min_age = max(date.cal_date[0, 0] for date in self.dates)
        max_age = min(date.cal_date[-1, 0] for date in self.dates)
        age_range = np.arange(min_age, max_age)

        probs = np.zeros_like(age_range, dtype=float)

        for date in self.dates:
            probs += np.interp(age_range,
                               date.cal_date[:, 0], date.cal_date[:, 1], left=0, right=0)

        self.summed = np.column_stack((age_range, probs))

    def plot(self):
        """
        Plots the summed probability density.
        """
        if self.summed is None:
            raise ValueError("Summation must be performed before plotting.")
        plt.plot(self.summed[:, 0], self.summed[:, 1], color='black')
        plt.gca().invert_xaxis()
        plt.xlabel('Calibrated age (BP)')
        plt.ylabel('Probability density')
        plt.show()
