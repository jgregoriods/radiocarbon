import numpy as np
import matplotlib.pyplot as plt

from .calibration_curves import CALIBRATION_CURVES


class Date:
    def __init__(self, c14age: int, c14sd: int):
        """
        Represents a radiocarbon date.

        Parameters:
        - c14age: Radiocarbon age (years BP).
        - c14sd: Standard deviation of the radiocarbon age.
        """
        self.c14age = c14age
        self.c14sd = c14sd
        self.cal_date = None

    def calibrate(self, curve: str = 'intcal20'):
        """
        Calibrates the radiocarbon date against a selected calibration curve.

        Parameters:
        - curve: Name of the calibration curve to use.
        """
        if curve not in CALIBRATION_CURVES:
            raise ValueError(f"Curve '{curve}' is not available.")

        calibration_curve = CALIBRATION_CURVES[curve]
        time_range = (self.c14age + 1000, self.c14age - 1000)

        # Select the relevant portion of the calibration curve
        selection = calibration_curve[
            (calibration_curve[:, 0] < time_range[0]) & (
                calibration_curve[:, 0] > time_range[1])
        ]

        probs = np.exp(-((self.c14age - selection[:, 1])**2 / (
            2 * (self.c14sd**2 + selection[:, 2]**2)))) / np.sqrt(self.c14sd**2 + selection[:, 2]**2)

        # get the range of the calibration curve
        calbp = selection[:, 0][probs > 1e-6]
        probs = probs[probs > 1e-6]

        calbp_interp = np.arange(calbp.min(), calbp.max() + 1)
        probs_interp = np.interp(calbp_interp, calbp[::-1], probs[::-1])

        normalized_probs = probs_interp / np.sum(probs_interp)

        self.cal_date = np.column_stack(
            (calbp_interp, probs_interp, normalized_probs))

    def hpd(self, level: float = 0.954):
        """
        Calculates the highest posterior density (HPD) region.

        Parameters:
        - level: Confidence level for the HPD region (default is 95.4%).

        Returns:
        - A numpy array containing the HPD region.
        """
        if self.cal_date is None:
            raise ValueError("Calibration must be performed before calculating HPD.")

        # Sort by probability and calculate cumulative sum
        sorted_cal = self.cal_date[np.argsort(self.cal_date[:, 2])[::-1]]
        cumulative_probs = np.cumsum(sorted_cal[:, 2])

        hpd_region = sorted_cal[cumulative_probs < level]

        hpd_set = set(hpd_region[:, 0])
        hpd_probs = [p if cal in hpd_set else 0 for cal, p in zip(self.cal_date[:, 0], self.cal_date[:, 2])]

        return np.column_stack((self.cal_date[:, 0], hpd_probs))

    def plot(self, level: float = 0.954):
        """
        Plots the calibrated date with the HPD region.

        Parameters:
        - level: Confidence level for the HPD region (default is 95.4%).
        """
        if self.cal_date is None:
            raise ValueError("Calibration must be performed before plotting.")

        hpd_region = self.hpd(level)
        plt.plot(self.cal_date[:, 0], self.cal_date[:, 2], color='black')
        plt.fill_between(hpd_region[:, 0], 0, hpd_region[:, 1], color='black', alpha=0.1)
        plt.gca().invert_xaxis()
        plt.xlabel('Calibrated age (BP)')
        plt.ylabel('Probability density')
        plt.show()
