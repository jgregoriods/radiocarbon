import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Union
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from .calibration_curves import CALIBRATION_CURVES
from .date import Date, Dates


def uncalibrate(cal_years: np.ndarray, cal_probs: np.ndarray, curve: np.ndarray = CALIBRATION_CURVES['intcal20']):
    cal_to_rcarbon = interp1d(
        curve[:, 0],
        curve[:, 1],
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )

    c14_ages = cal_to_rcarbon(cal_years)
    binned = np.round(c14_ages).astype(int)

    rcarbon_probs_dict = defaultdict(float)
    for age, prob in zip(binned, cal_probs):
        rcarbon_probs_dict[age] += prob

    rcarbon_years = np.array(list(rcarbon_probs_dict.keys()))
    rcarbon_probs = np.array([rcarbon_probs_dict[y] for y in rcarbon_years])
    rcarbon_probs = rcarbon_probs / rcarbon_probs.sum()

    return rcarbon_years, rcarbon_probs


class SPD:
    """
    Represents a Summed Probability Density (SPD) for a collection of radiocarbon dates.

    Attributes:
        dates (Dates): A collection of `Date` objects.
        summed (Optional[np.ndarray]): The summed probability density as a numpy array
                                       with columns: [age, probability].
    """

    def __init__(self, dates: Dates):
        """
        Initializes the SPD instance.

        Args:
            dates (Dates): A collection of `Date` objects to sum.
        """
        if not dates:
            raise ValueError("The list of dates cannot be empty.")

        self.dates = dates
        self.bins = defaultdict(list)
        self.summed: Optional[np.ndarray] = None

        for date in self.dates:
            if not hasattr(date, 'calibrate') or not callable(date.calibrate):
                raise TypeError("Each date must have a `calibrate` method.")
            date.calibrate()

    def sum(self) -> 'SPD':
        """
        Sums the probability densities of all calibrated dates.
        """
        if not self.dates:
            raise ValueError("No dates provided for summation.")

        min_age = min(date.cal_date[0, 0] for date in self.dates)
        max_age = max(date.cal_date[-1, 0] for date in self.dates)
        age_range = np.arange(min_age, max_age)

        probs = np.zeros_like(age_range, dtype=float)

        self.bins = defaultdict(list)
        bin_id = 0
        for date in self.dates:
            if date.bin_id:
                self.bins[date.bin_id].append(date)
            else:
                self.bins[bin_id].append(date)
                bin_id += 1

        for _, bin_dates in self.bins.items():
            bin_probs = np.zeros_like(age_range, dtype=float)
            for date in bin_dates:
                bin_probs += np.interp(
                    age_range, date.cal_date[:, 0], date.cal_date[:, 1], left=0, right=0
                )
            probs += bin_probs / len(bin_dates)

        self.summed = np.column_stack((age_range, probs))
        return self

    def plot(self, age: str = 'BP') -> None:
        """
        Plots the summed probability density.

        Args:
            age (str): The age format to use ('BP' or 'AD'). Default is 'BP'.
        """
        if self.summed is None:
            raise ValueError("Summation must be performed before plotting.")

        cal_dates = self.summed[:, 0].copy()

        if age == 'AD':
            cal_dates = 1950 - cal_dates

        plt.plot(cal_dates, self.summed[:, 1], color="black")
        plt.fill_between(cal_dates, self.summed[:, 1], color="lightgray")

        if age == 'BP':
            plt.gca().invert_xaxis()

        plt.xlabel(f"Calibrated Age ({age})")
        plt.ylabel("Probability Density")
        plt.title("Summed Probability Density (SPD)")
        plt.show()

    def to_json(self) -> str:
        """
        Converts the SPD to a JSON string.

        Returns:
            str: The JSON representation of the SPD.
        """
        return json.dumps({
            'dates': [date.to_json() for date in self.dates],
            'summed': self.summed.tolist() if self.summed is not None else None
        })

    @staticmethod
    def from_json(json_string: str) -> 'SPD':
        """
        Creates an SPD object from a JSON string.

        Args:
            json_string (str): The JSON string to parse.

        Returns:
            SPD: The SPD object parsed from the JSON string.
        """
        data = json.loads(json_string)
        dates = Dates([Date.from_json(date) for date in data['dates']])
        spd = SPD(dates)
        spd.summed = np.array(data['summed']) if data['summed'] is not None else None
        return spd


class SimSPD:
    """
    Represents a simulated Summed Probability Density (SimSPD).

    Attributes:
        date_range (Tuple[int, int]): Range of years to simulate (start, end).
        n_dates (int): Number of dates to simulate per iteration.
        n_iter (int): Number of iterations for the simulation.
        errors (List[int]): List of errors for simulated dates.
        curves (List[str]): List of calibration curves to use.
        probs (List[float]): List of probabilities for sampling dates.
        spds (List[SPD]): List of simulated SPDs.
    """

    def __init__(
            self,
            date_range: Tuple[int, int],
            n_dates: int,
            n_iter: int = 1000,
            errors: List[List[int]] = None,
            curves: List[List[str]] = None,
            probs: List[float] = None
    ):
        """
        Initializes the SimSPD instance.

        Args:
            date_range (Tuple[int, int]): Range of years to simulate (start, end).
            n_dates (int): Number of dates to simulate per iteration.
            n_iter (int): Number of iterations for the simulation. Default is 1000.
            errors (List[int]): List of errors for simulated dates.
            curves (List[str]): List of calibration curves to use.
            probs (List[float]): List of probabilities for sampling dates.
        """
        if not isinstance(date_range, tuple) or len(date_range) != 2:
            raise ValueError("date_range must be a tuple of (start, end).")
        if n_dates <= 0 or n_iter <= 0:
            raise ValueError("n_dates and n_iter must be positive integers.")

        self.date_range = date_range
        self.n_dates = n_dates
        self.n_iter = n_iter
        self.errors = errors
        self.curves = curves
        self.probs = probs if probs is not None else np.ones(self.date_range[1] - self.date_range[0]) / (self.date_range[1] - self.date_range[0])
        self.spds: List[SPD] = []
        self.prob_matrix: Optional[np.ndarray] = None
        self.summary_stats: Optional[np.ndarray] = None

    def _generate_random_dates(self, method: str = 'calsample') -> List[Date]:
        """
        Generates random `Date` objects based on the specified model.

        Returns:
            List[Date]: A list of randomly generated `Date` objects.
        """
        X = np.arange(self.date_range[0], self.date_range[1])

        curves = [np.random.choice(curve) for curve in self.curves] if self.curves else ['intcal20'] * self.n_dates
        errors = [np.random.choice(error) for error in self.errors] if self.errors else np.random.randint(0, 100, self.n_dates)

        if method == 'calsample':
            years = np.random.choice(X, self.n_dates, replace=True, p=self.probs)
            c14ages = [CALIBRATION_CURVES[curve][np.argmin(np.abs(CALIBRATION_CURVES[curve][:, 0] - year)), 1] for year, curve in zip(years, curves)]
        else:
            c14ages = []
            uncal_probs = dict()
            for i in range(self.n_dates):
                if curves[i] not in uncal_probs:
                    uncal_probs[curves[i]] = uncalibrate(X, self.probs, CALIBRATION_CURVES[curves[i]])
                c14ages.append(np.random.choice(uncal_probs[curves[i]][0], p=uncal_probs[curves[i]][1]))

        return [Date(c14age, error, curve) for c14age, error, curve in zip(c14ages, errors, curves)]

    def simulate_spds(self, method: str = 'calsample') -> np.ndarray:
        """
        Simulates SPDs and calculates percentile bounds.
        """
        if method not in ["uncalsample", "calsample"]:
            raise ValueError("Method must be 'uncalsample' or 'calsample'.")

        self.spds = [self._create_spd(self._generate_random_dates(method))
                     for _ in range(self.n_iter)]

        min_age = min(spd.summed[0, 0] for spd in self.spds)
        max_age = max(spd.summed[-1, 0] for spd in self.spds)
        age_range = np.arange(min_age, max_age)

        self.prob_matrix = self._create_probability_matrix(age_range)
        self.summary_stats = self._calculate_stats(self.prob_matrix)

    def _create_spd(self, dates: List[Date]) -> SPD:
        """
        Creates and sums an SPD for a given set of dates.

        Args:
            dates (List[Date]): List of `Date` objects.

        Returns:
            SPD: The resulting SPD object.
        """
        spd = SPD(dates)
        spd.sum()
        return spd

    def _create_probability_matrix(self, age_range: np.ndarray) -> np.ndarray:
        """
        Creates a matrix of probabilities for all SPDs.

        Args:
            age_range (np.ndarray): Array of age values.

        Returns:
            np.ndarray: A 2D matrix with probabilities for each SPD.
        """
        prob_matrix = np.zeros((len(age_range), self.n_iter + 1))
        prob_matrix[:, 0] = age_range

        for i, spd in enumerate(self.spds):
            prob_matrix[:, i + 1] = np.interp(
                age_range, spd.summed[:, 0], spd.summed[:, 1]
            )

        return prob_matrix

    def _calculate_stats(self, prob_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates summary statistics (mean and standard deviation) for the probability matrix.

        Args:
            prob_matrix (np.ndarray): A 2D matrix with probabilities for each SPD.

        Returns:
            np.ndarray: A 2D array with mean and standard deviation for each age.
        """
        means = np.mean(prob_matrix[:, 1:], axis=1)
        stds = np.std(prob_matrix[:, 1:], axis=1)
        return np.column_stack((means, stds))


class SPDTest:
    """
    Tests an SPD by simulating a series of summed probability densities (SPDs)
    and comparing the real SPD with the simulation's confidence intervals.
    """

    def __init__(self, spd: SPD, date_range: Optional[Tuple[int, int]] = None):
        """
        Initializes the SPDTest instance.

        Args:
            spd (SPD): The real SPD object to test.
            date_range (Optional[Tuple[int, int]]): Range of years for simulation.
        """
        if not isinstance(spd, SPD):
            raise TypeError(
                "The provided object must be an instance of the SPD class.")
        if spd.summed is None:
            raise ValueError(
                "The provided SPD must have its probabilities summed.")

        self.spd = spd
        self.simulations: Optional[SimSPD] = None

        self.date_range = date_range if date_range else (
            int(min(date.median() for date in spd.dates)),
            int(max(date.median() for date in spd.dates)),
        )

        self.n_dates = len(spd.bins)
        self.n_iter = 0
        self.intervals: Dict[str, List[Tuple[int, int]]] = {}
        self.model = None

        self.lower_percentile = None
        self.upper_percentile = None

        self.p_value = None

    def run_test(self, n_iter: int = 1000, model: str = 'exp', method: str = 'calsample', probs: List[float] = None) -> 'SPDTest':
        """
        Runs simulations using the same time range and number of dates as the real SPD.

        Args:
            n_iter (int): Number of iterations for the simulation. Default is 1000.
            model (str): Model for date generation ('uniform', 'linear', 'exp' or 'custom'). Default is 'exp'.
        """

        errors = [[date.c14sd for date in bin] for bin in self.spd.bins.values()]
        curves = [[date.curve for date in bin] for bin in self.spd.bins.values()]

        if model == 'exp':
            ages = self.spd.summed[:, 0]
            spd_values = self.spd.summed[:, 1] + 1e-10

            sel_ages = ages[(ages > self.date_range[0]) & (ages < self.date_range[1])]
            x = np.arange(len(sel_ages))
            y = spd_values[(ages > self.date_range[0]) & (ages < self.date_range[1])]

            a, b = curve_fit(lambda x, a, b: np.exp(a + b * x), x, y, p0=(-1.0, -0.1))[0]
            probs = np.exp(a + b * np.arange(self.date_range[0], self.date_range[1]))
            probs /= probs.sum()

        elif model == 'linear':
            ages = self.spd.summed[:, 0]
            spd_values = self.spd.summed[:, 1]

            sel_ages = ages[(ages > self.date_range[0]) & (ages < self.date_range[1])]
            x = np.arange(len(sel_ages))
            y = spd_values[(ages > self.date_range[0]) & (ages < self.date_range[1])]

            m, b = np.polyfit(x, y, deg=1)
            probs = b + m * np.arange(self.date_range[0], self.date_range[1])
            probs /= probs.sum()

        elif model == 'uniform':
            probs = np.ones(self.date_range[1] - self.date_range[0])
            probs /= probs.sum()

        elif model == 'custom':
            if probs is None:
                raise ValueError("Custom model requires a list of probabilities.")
            probs = probs

        else:
            raise ValueError("Model not supported yet. Choose between 'uniform', 'linear', 'exp' or 'custom'.")

        self.simulations = SimSPD(
            date_range=self.date_range,
            n_dates=self.n_dates,
            n_iter=n_iter,
            errors=errors,
            curves=curves,
            probs=probs
        )

        self.model = model
        self.n_iter = n_iter
        self.simulations.simulate_spds(method)
        self.intervals["above"], self.intervals["below"] = self._extract_intervals()
        self.p_value = self._calculate_p_value()

        return self

    def _get_percentile_bounds(self, prob_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the lower and upper percentile bounds for the simulated SPDs.

        Args:
            prob_matrix (np.ndarray): A 2D matrix with probabilities for each SPD.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - lower_percentile: Lower percentile bounds.
                - upper_percentile: Upper percentile bounds.
        """
        lower_percentile = np.percentile(prob_matrix[:, 1:], 2.5, axis=1)
        upper_percentile = np.percentile(prob_matrix[:, 1:], 97.5, axis=1)
        return lower_percentile, upper_percentile

    def _extract_intervals(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Identifies intervals where the observed SPD is above or below the confidence envelope.

        Returns:
            Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]: A tuple containing:
                - above_intervals: List of intervals where SPD is above the envelope.
                - below_intervals: List of intervals where SPD is below the envelope.
        """
        observed_ages = self.spd.summed[:, 0]
        observed_probs = self.spd.summed[:, 1]

        age_range = self.simulations.prob_matrix[:, 0]
        self.lower_percentile, self.upper_percentile = self._get_percentile_bounds(self.simulations.prob_matrix)

        lower_ci = np.interp(observed_ages, age_range, self.lower_percentile)
        upper_ci = np.interp(observed_ages, age_range, self.upper_percentile)

        above_intervals, below_intervals = [], []
        current_interval = []
        is_above = None

        for i, (age, prob, low, high) in enumerate(zip(observed_ages, observed_probs, lower_ci, upper_ci)):
            if prob > high:
                if not current_interval or not is_above:
                    current_interval = [age, age]
                    is_above = True
                else:
                    current_interval[1] = age
            elif prob < low:
                if not current_interval or is_above:
                    current_interval = [age, age]
                    is_above = False
                else:
                    current_interval[1] = age
            else:
                if current_interval:
                    if is_above:
                        above_intervals.append(tuple(current_interval))
                    else:
                        below_intervals.append(tuple(current_interval))
                    current_interval = []

        if current_interval:
            if is_above:
                above_intervals.append(tuple(current_interval))
            else:
                below_intervals.append(tuple(current_interval))

        return above_intervals, below_intervals

    def _calculate_p_value(self):
        """
        Calculates the p-value for the observed SPD. The p-value is the proportion of
        simulated SPDs with a sum of z-scores in the significant regions that is greater
        than the sum of z-scores for the observed SPD (Timpson et al., 2014).

        Returns:
            float: The p-value for the observed SPD.
        """
        indices = np.where((self.spd.summed[:, 0] > self.date_range[0]) & (self.spd.summed[:, 0] < self.date_range[1]))[0]
        observed_ages = self.spd.summed[:, 0][indices]
        observed_probs = self.spd.summed[:, 1][indices]

        sim_indices = np.where((self.simulations.prob_matrix[:, 0] > self.date_range[0]) & (self.simulations.prob_matrix[:, 0] < self.date_range[1]))[0]
        prob_matrix = self.simulations.prob_matrix[sim_indices, :]
        age_range = prob_matrix[:, 0]
        mean, std = self.simulations.summary_stats[:, 0][sim_indices], self.simulations.summary_stats[:, 1][sim_indices]

        interp_mean = np.interp(observed_ages, age_range, mean)
        interp_std = np.interp(observed_ages, age_range, std)

        observed_z_scores = np.abs((observed_probs - interp_mean) / interp_std)
        observed_z_sum = np.sum(observed_z_scores[observed_z_scores > 1.96])

        score_sums = []
        for i in range(self.n_iter):
            sim_spd = prob_matrix[:, i + 1]
            z_scores = np.abs((sim_spd - mean) / std)
            z_sum = np.sum(z_scores[z_scores > 1.96])
            score_sums.append(z_sum)

        p_val = np.mean(np.array(score_sums) > observed_z_sum)
        return p_val

    def plot(self, age: str = 'BP'):
        """
        Plots the real SPD overlaid on the simulated confidence intervals.

        Args:
            age (str): The age format to use ('BP' or 'AD'). Default is 'BP'.
        """
        if self.simulations is None:
            raise ValueError("Simulations must be run before plotting.")

        cal_dates = self.spd.summed[:, 0].copy()
        sim_dates = self.simulations.prob_matrix[:, 0].copy()
        start_date = self.date_range[1]
        end_date = self.date_range[0]

        if age == 'AD':
            cal_dates = 1950 - cal_dates
            sim_dates = 1950 - sim_dates
            start_date = 1950 - start_date
            end_date = 1950 - end_date

        plt.fill_between(
            sim_dates,
            self.lower_percentile,
            self.upper_percentile,
            color="lightgray",
            label="95% CI",
        )

        plt.plot(
            cal_dates,
            self.spd.summed[:, 1],
            color="black",
            label="SPD",
        )

        if age == 'AD':
            above_intervals = [(1950 - end, 1950 - start) for start, end in self.intervals['above']]
            below_intervals = [(1950 - end, 1950 - start) for start, end in self.intervals['below']]
        else:
            above_intervals = self.intervals['above']
            below_intervals = self.intervals['below']

        for i, (start, end) in enumerate(above_intervals):
            plt.fill_betweenx(
                [0, self.spd.summed[:, 1].max()],
                start,
                end,
                color="red",
                alpha=0.3,
                label="Above CI" if i == 0 else None,
            )

        for i, (start, end) in enumerate(below_intervals):
            plt.fill_betweenx(
                [0, self.spd.summed[:, 1].max()],
                start,
                end,
                color="blue",
                alpha=0.3,
                label="Below CI" if i == 0 else None,
            )

        if age == 'BP':
            plt.gca().invert_xaxis()
        plt.xlim(start_date, end_date)
        plt.xlabel(f"Calibrated Age ({age})")
        plt.ylabel("Probability Density")
        plt.title(f"SPD with Simulated CI ({self.model} model)")
        plt.legend()
        plt.show()

    def __str__(self) -> str:
        """
        Returns the string representation of the SPDTest instance.

        Returns:
            str: The string representation of the SPDTest object.
        """
        positive_intervals = ', '.join(
            f"{int(start)} BP - {int(end)} BP" for start, end in self.intervals['above'] if self.date_range[0] <= start <= self.date_range[1] or self.date_range[0] <= end <= self.date_range[1]
        )
        negative_intervals = ', '.join(
            f"{int(start)} BP - {int(end)} BP" for start, end in self.intervals['below'] if self.date_range[0] <= start <= self.date_range[1] or self.date_range[0] <= end <= self.date_range[1]
        )
        return f"SPD Model Test\n----------------\n" \
               f"Model: {self.model}\n" \
               f"Number of dates: {self.n_dates}\n" \
               f"Number of simulations: {self.n_iter}\n" \
               f"Date range: {self.date_range[0]} - {self.date_range[1]} BP\n" \
               f"Positive deviations: {positive_intervals}\n" \
               f"Negative deviations: {negative_intervals}\n" \
               f"Global p-value: {self.p_value}"

