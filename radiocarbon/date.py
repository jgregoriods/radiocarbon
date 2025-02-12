import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Union

from .calibration_curves import CALIBRATION_CURVES


class Date:
    """
    Represents a radiocarbon date and provides methods for calibration.

    Attributes:
        c14age (int): Radiocarbon age in years BP.
        c14sd (int): Standard deviation of the radiocarbon age.
        cal_date (Optional[np.ndarray]): Calibrated date as a numpy array with columns:
                                         [calibrated age, probability, normalized probability].
    """

    def __init__(self, c14age: int, c14sd: int):
        """
        Initializes a radiocarbon date.

        Args:
            c14age (int): Radiocarbon age in years BP.
            c14sd (int): Standard deviation of the radiocarbon age.
        """
        self.c14age = c14age
        self.c14sd = c14sd
        self.cal_date: Optional[np.ndarray] = None

    def calibrate(self, curve: Optional[str] = 'intcal20') -> None:
        """
        Calibrates the radiocarbon date against a specified calibration curve.

        Args:
            curve (Optional[str]): Name of the calibration curve to use. Defaults to 'intcal20'.
        """
        if curve not in CALIBRATION_CURVES:
            raise ValueError(f"Curve '{curve}' is not available.")

        calibration_curve = CALIBRATION_CURVES[curve]
        time_range = (self.c14age + 1000, self.c14age - 1000)

        # Select the relevant portion of the calibration curve
        selection = calibration_curve[
            (calibration_curve[:, 0] < time_range[0]) & (calibration_curve[:, 0] > time_range[1])
            ]

        # Calculate probabilities
        probs = np.exp(-((self.c14age - selection[:, 1])**2 / (
            2 * (self.c14sd**2 + selection[:, 2]**2)))) / np.sqrt(self.c14sd**2 + selection[:, 2]**2)

        # Filter out negligible probabilities
        calbp = selection[:, 0][probs > 1e-6]
        probs = probs[probs > 1e-6]

        # Interpolate
        calbp_interp = np.arange(calbp.min(), calbp.max() + 1)
        probs_interp = np.interp(calbp_interp, calbp[::-1], probs[::-1])
        
        # Normalize probabilities
        normalized_probs = probs_interp / np.sum(probs_interp)

        self.cal_date = np.column_stack((calbp_interp, probs_interp, normalized_probs))

    def mean(self) -> float:
        """
        Calculates the mean calibrated date.

        Returns:
            float: The mean calibrated date.
        """
        if self.cal_date is None:
            raise ValueError(
                "Calibration must be performed before calculating the mean.")

        return np.round(np.sum(self.cal_date[:, 0] * self.cal_date[:, 2]))

    def median(self) -> float:
        """
        Calculates the median calibrated date.

        Returns:
            float: The median calibrated date.
        """
        if self.cal_date is None:
            raise ValueError(
                "Calibration must be performed before calculating the median.")

        return np.round(np.interp(0.5, np.cumsum(self.cal_date[:, 2]), self.cal_date[:, 0]))

    def hpd(self, level: float = 0.954) -> List[np.ndarray]:
        """
        Calculates the highest posterior density (HPD) region.

        Args:
            level (float): Confidence level for the HPD region. Defaults to 0.954 (95.4%).

        Returns:
            List[np.ndarray]: A list of numpy arrays, each representing a segment of the HPD region.
        """
        if self.cal_date is None:
            raise ValueError(
                "Calibration must be performed before calculating HPD.")

        sorted_cal = self.cal_date[np.argsort(self.cal_date[:, 2])[::-1]]
        cumulative_probs = np.cumsum(sorted_cal[:, 2])

        hpd_region = sorted_cal[cumulative_probs < level]

        # Split the HPD region into continuous segments
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

    def plot(self, level: float = 0.954, age: str = 'BP') -> None:
        """
        Plots the calibrated date with the HPD region.

        Args:
            level (float): Confidence level for the HPD region. Defaults to 0.954 (95.4%).
            age (str): Age format to display. Options are 'BP' (default) or 'AD'.
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

        plt.xlabel(f'Calibrated age ({age})')
        plt.ylabel('Probability density')
        plt.show()

    def __repr__(self) -> str:
        """
        Returns a string representation of the radiocarbon date.

        Returns:
            str: A string representation of the radiocarbon date.
        """
        if self.cal_date is None:
            return f"Radiocarbon date: {self.c14age} +/- {self.c14sd} BP"

        hpd = self.hpd()
        bounds = [(int(segment[0][0]), int(segment[-1][0])) for segment in hpd]
        return f"Radiocarbon date: {self.c14age} +/- {self.c14sd} BP\nCalibrated date: {', '.join([f'{b[0]}-{b[1]}' for b in bounds])} cal BP (95.4%)"


class Dates:
    """
    Represents a collection of radiocarbon dates and provides methods for batch calibration.

    Attributes:
        dates (List[Date]): A list of Date objects.
        curves (Optional[List[str]]): A list of calibration curve names corresponding to each date.
    """

    def __init__(self, c14ages: List[int], c14sds: List[int], curves: Optional[List[str]] = None):
        """
        Initializes a collection of radiocarbon dates.

        Args:
            c14ages (List[int]): List of radiocarbon ages in years BP.
            c14sds (List[int]): List of standard deviations of the radiocarbon ages.
            curves (Optional[List[str]]): List of calibration curve names for each date.
        """
        if len(c14ages) != len(c14sds):
            raise ValueError("The number of radiocarbon ages and standard deviations must be equal.")

        if curves is not None and len(c14ages) != len(curves):
            raise ValueError("The number of curves must match the number of radiocarbon dates.")

        self.dates = [Date(age, sd) for age, sd in zip(c14ages, c14sds)]
        self.curves = curves

    def calibrate(self) -> None:
        """
        Calibrates all radiocarbon dates in the collection.
        """
        for i, date in enumerate(self.dates):
            curve = 'intcal20' if self.curves is None else self.curves[i]
            date.calibrate(curve)

    def __getitem__(self, i: int) -> Date:
        """
        Returns the radiocarbon date at the specified index.
        """
        return self.dates[i]

    def __len__(self) -> int:
        """
        Returns the number of radiocarbon dates in the collection.
        """
        return len(self.dates)

    def __repr__(self) -> str:
        """
        Returns a string representation of the collection of radiocarbon dates.
        """
        return '\n'.join([date.__repr__() for date in self.dates])

    def __iter__(self):
        """
        Returns an iterator over the radiocarbon dates.
        """
        return iter(self.dates)


class Bins:
    """
    Represents a collection of radiocarbon dates binned by a specified bin size.

    Attributes:
        dates (Dates): A Dates object containing the radiocarbon dates.
        labels (List[str]): A list of labels (ideally site names) corresponding to each date.
        bin_size (int): Size of the bins in years.
        bins (Dates): A Dates object containing the binned radiocarbon dates.
    """
    def __init__(self, dates: Dates, labels: List[str], bin_size: int = 100):
        """
        Initializes a collection of binned radiocarbon dates.

        Args:
            dates (Dates): A Dates object containing the radiocarbon dates.
            labels (List[str]): A list of labels (ideally site names) corresponding to each date.
            bin_size (int): Size of the bins in years.
        """
        self.dates = dates
        self.labels = labels
        self.bin_size = bin_size
        self.bins = self._bin_dates()

    def _bin_dates(self) -> Dates:
        """
        Bins the radiocarbon dates.

        Returns:
            Dates: A Dates object containing the binned radiocarbon dates.
        """
        sites = {}
        for i, label in enumerate(self.labels):
            if label not in sites:
                sites[label] = {}
            bin_key = self.dates[i].c14age // self.bin_size
            if bin_key not in sites[label]:
                sites[label][bin_key] = []
            sites[label][bin_key].append(self.dates[i])
        filtered_ages = []
        filtered_errors = []
        for label in sites:
            for bin_key in sites[label]:
                filtered_ages.append(sites[label][bin_key][0].c14age)
                filtered_errors.append(sites[label][bin_key][0].c14sd)
        return Dates(filtered_ages, filtered_errors)

