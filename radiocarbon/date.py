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
            raise ValueError(
                "Calibration must be performed before calculating HPD.")

        # Sort by probability and calculate cumulative sum
        sorted_cal = self.cal_date[np.argsort(self.cal_date[:, 2])[::-1]]
        cumulative_probs = np.cumsum(sorted_cal[:, 2])

        hpd_region = sorted_cal[cumulative_probs < level]

        hpd_set = sorted(hpd_region[:, 0])
        hpd_probs = [p for cal, p in zip(
            self.cal_date[:, 0], self.cal_date[:, 2]) if cal in hpd_set]

        res = np.column_stack((hpd_set, hpd_probs))

        segments = []
        j = 0
        for i in range(1, len(res)):
            if res[i][0] - res[i - 1][0] > 1:
                segments.append(res[j:i])
                j = i

        if j < len(res):
            segments.append(res[j:])

        return segments

    def plot(self, level: float = 0.954, age = 'BP'):
        """
        Plots the calibrated date with the HPD region.

        Parameters:
        - level: Confidence level for the HPD region (default is 95.4%).
        """
        if self.cal_date is None:
            raise ValueError("Calibration must be performed before plotting.")

        hpd_region = self.hpd(level)
        cal_date = self.cal_date.copy()
        if age == 'AD':
            cal_date[:, 0] = 1950 - cal_date[:, 0]
            for segment in hpd_region:
                segment[:, 0] = 1950 - segment[:, 0]
        plt.plot(cal_date[:, 0], cal_date[:, 2], color='black')
        for segment in hpd_region:
            plt.fill_between(segment[:, 0], 0,
                             segment[:, 1], color='black', alpha=0.1)
        if age == 'BP':
            plt.gca().invert_xaxis()

        bounds = []
        for segment in hpd_region:
            if age == 'AD':
                bounds.append((int(segment[-1][0]), int(segment[0][0])))
            else:
                bounds.append((int(segment[0][0]), int(segment[-1][0])))

        cum_probs = [np.round(np.sum(segment[:, 1]) * 100, 2) for segment in hpd_region]

        text = '\n'.join([f'{b[0]}-{b[1]} ({p}%)' for b, p in zip(bounds, cum_probs)])
        plt.text(0.05, 0.95, text, horizontalalignment='left',
                 verticalalignment='top', transform=plt.gca().transAxes)

        plt.xlabel('Calibrated age (BP)')
        plt.ylabel('Probability density')
        plt.show()

    def __repr__(self):
        hpd = self.hpd()
        bounds = []
        for segment in hpd:
            bounds.append((int(segment[0][0]), int(segment[-1][0])))
        if self.cal_date is None:
            return f"Radiocarbon date: {self.c14age} +/- {self.c14sd} BP"
        else:
            # return both
            return f"Radiocarbon date: {self.c14age} +/- {self.c14sd} BP\nCalibrated date: {', '.join([f'{b[0]}-{b[1]}' for b in bounds])} cal BP (95.4%)"
