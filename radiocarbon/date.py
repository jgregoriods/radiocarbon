import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict

import scipy.cluster.hierarchy as sch
from typing import List, Optional, Tuple, Dict, Union

from .calibration_curves import CALIBRATION_CURVES


class Date:
    """
    Represents a radiocarbon date and provides methods for calibration.

    Attributes:
        c14age (int): Radiocarbon age in years BP.
        c14sd (int): Standard deviation of the radiocarbon age.
        curve (str): Name of the calibration curve to use.
        cal_date (Optional[np.ndarray]): Calibrated date as a numpy array with columns:
                                         [calibrated age, probability, normalized probability].
        bin_id (Optional[str]): Id of the bin the date belongs to.
    """

    def __init__(self, c14age: int, c14sd: int, curve: Optional[str] = None, bin_id: Optional[str] = None):
        """
        Initializes a radiocarbon date.

        Args:
            c14age (int): Radiocarbon age in years BP.
            c14sd (int): Standard deviation of the radiocarbon age.
            curve (Optional[str]): Name of the calibration curve to use.
        """

        if curve and curve.lower() not in CALIBRATION_CURVES:
            raise ValueError(f"Curve '{curve}' is not available.")

        self.c14age = c14age
        self.c14sd = c14sd
        self.curve = curve.lower() if curve else 'intcal20'
        self.cal_date: Optional[np.ndarray] = None
        self.bin_id = bin_id

    def calibrate(self) -> 'Date':
        """
        Calibrates the radiocarbon date.
        """
        calibration_curve = CALIBRATION_CURVES[self.curve]
        time_range = (self.c14age + 1000, self.c14age - 1000)

        selection = calibration_curve[
            (calibration_curve[:, 0] < time_range[0]) & (calibration_curve[:, 0] > time_range[1])
            ]

        probs = np.exp(-((self.c14age - selection[:, 1])**2 / (
            2 * (self.c14sd**2 + selection[:, 2]**2)))) / np.sqrt(self.c14sd**2 + selection[:, 2]**2)

        calbp = selection[:, 0][probs > 1e-6]
        probs = probs[probs > 1e-6]

        calbp_interp = np.arange(calbp.min(), calbp.max() + 1)
        probs_interp = np.interp(calbp_interp, calbp[::-1], probs[::-1])
        
        normalized_probs = probs_interp / np.sum(probs_interp)

        self.cal_date = np.column_stack((calbp_interp, probs_interp, normalized_probs))

        return self

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

    def to_json(self) -> str:
        """
        Converts the radiocarbon date to a JSON string.

        Returns:
            str: A JSON string representing the radiocarbon date.
        """
        return json.dumps({
            'c14age': self.c14age,
            'c14sd': self.c14sd,
            'curve': self.curve,
            'cal_date': self.cal_date.tolist() if self.cal_date is not None else None,
            'bin_id': self.bin_id
        })

    @staticmethod
    def from_json(json_str: str) -> 'Date':
        """
        Creates a Date object from a JSON string.

        Args:
            json_str (str): A JSON string representing the radiocarbon date.

        Returns:
            Date: A Date object.
        """
        data = json.loads(json_str)
        date = Date(data['c14age'], data['c14sd'], data['curve'], data['bin_id'])
        if data['cal_date']:
            date.cal_date = np.array(data['cal_date'])
        return date

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

    def __init__(self, dates: List[Date]):
        """
        Initializes a collection of radiocarbon dates.

        Args:
            dates (List[Date]): A list of Date objects.
        """
        self.dates = dates

    def calibrate(self) -> 'Dates':
        """
        Calibrates all radiocarbon dates in the collection.
        """
        for date in self.dates:
            date.calibrate()
        return self

    @staticmethod
    def from_df(df: 'pd.DataFrame', age_col: str, sd_col: str, curve_col: Optional[str] = None) -> 'Dates':
        """
        Creates a Dates object from a pandas DataFrame.

        Args:
            df (pd.DataFrame): A pandas DataFrame containing the radiocarbon dates.
            age_col (str): Name of the column containing the radiocarbon ages.
            sd_col (str): Name of the column containing the standard deviations of the radiocarbon ages.
            curve_col (Optional[str]): Name of the column containing the calibration curve names.

        Returns:
            Dates: A Dates object containing the radiocarbon dates.
        """
        df[age_col] = df[age_col].astype(int)
        df[sd_col] = df[sd_col].astype(int)

        c14ages = df[age_col].tolist()
        c14sds = df[sd_col].tolist()
        curves = df[curve_col].tolist() if curve_col is not None else None

        date_list = []
        for i in range(len(c14ages)):
            date_list.append(Date(c14ages[i], c14sds[i], curves[i] if curves else None))

        return Dates(date_list)

    def bin(self, labels: List[str], h: int = 100) -> 'Dates':
        """
        Bins the radiocarbon dates by a specified bin size.
        """
        if len(self.dates) != len(labels):
            raise ValueError("The number of dates and labels must be equal.")

        for date in self.dates:
            if date.cal_date is None:
                date.calibrate()

        sites = defaultdict(list)
        for i, label in enumerate(labels):
            sites[label].append(self.dates[i])

        binned_dates = []

        for site in sites:
            date_array = np.array([date.median() for date in sites[site]]).reshape(-1, 1)
            if len(date_array) == 1:
                site_date = sites[site][0]
                site_date.bin_id = f'{site}_1'
                binned_dates.append(site_date)
                continue
            linkage_matrix = sch.linkage(date_array, method='ward')
            clusters = sch.fcluster(linkage_matrix, t=h, criterion='distance')
            for i, date in enumerate(sites[site]):
                date.bin_id = f'{site}_{clusters[i]}'
                binned_dates.append(date)

        return self

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

    def __str__(self) -> str:
        """
        Returns a string representation of the collection of radiocarbon dates.
        """
        return '\n'.join([date.__repr__() for date in self.dates])

    def __iter__(self):
        """
        Returns an iterator over the radiocarbon dates.
        """
        return iter(self.dates)
