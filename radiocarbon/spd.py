import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Union

from .calibration_curves import CALIBRATION_CURVES
from .date import Date, Dates


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
        self.summed: Optional[np.ndarray] = None

        # Ensure all dates are calibrated
        for date in self.dates:
            if not hasattr(date, 'calibrate') or not callable(date.calibrate):
                raise TypeError("Each date must have a `calibrate` method.")
            date.calibrate()

    def sum(self) -> None:
        """
        Sums the probability densities of all calibrated dates.
        """
        if not self.dates:
            raise ValueError("No dates provided for summation.")

        # Define the age range for summation
        min_age = min(date.cal_date[0, 0] for date in self.dates)
        max_age = max(date.cal_date[-1, 0] for date in self.dates)
        age_range = np.arange(min_age, max_age)

        probs = np.zeros_like(age_range, dtype=float)

        # Sum the probability densities
        for date in self.dates:
            probs += np.interp(
                age_range, date.cal_date[:, 0], date.cal_date[:, 1], left=0, right=0
            )

        self.summed = np.column_stack((age_range, probs))

    def plot(self) -> None:
        """
        Plots the summed probability density.
        """
        if self.summed is None:
            raise ValueError("Summation must be performed before plotting.")

        plt.plot(self.summed[:, 0], self.summed[:, 1], color="black")
        plt.fill_between(self.summed[:, 0], self.summed[:, 1], color="lightgray")
        plt.gca().invert_xaxis()
        plt.xlabel("Calibrated Age (BP)")
        plt.ylabel("Probability Density")
        plt.title("Summed Probability Density (SPD)")
        plt.show()


class SimSPD:
    """
    Represents a simulated Summed Probability Density (SimSPD).

    Attributes:
        date_range (Tuple[int, int]): Range of years to simulate (start, end).
        n_dates (int): Number of dates to simulate per iteration.
        n_iter (int): Number of iterations for the simulation.
        model (str): Model for date generation ('uniform' or 'exp').
        lam (float): Lambda parameter for the exponential model.
        errors (List[int]): List of errors for simulated dates.
        spds (List[SPD]): List of simulated SPDs.
    """

    def __init__(
            self,
            date_range: Tuple[int, int],
            n_dates: int,
            n_iter: int = 1000,
            errors = List[int],
            model: str = 'exp',
            lam: float = 1
    ):
        """
        Initializes the SimSPD instance.

        Args:
            date_range (Tuple[int, int]): Range of years to simulate (start, end).
            n_dates (int): Number of dates to simulate per iteration.
            n_iter (int): Number of iterations for the simulation. Default is 1000.
            errors (List[int]): List of errors for simulated dates.
            model (str): Model for date generation ('uniform' or 'exp'). Default is 'exp'.
            lam (float): Lambda parameter for the exponential model. Default is 1.0.
        """
        if not isinstance(date_range, tuple) or len(date_range) != 2:
            raise ValueError("date_range must be a tuple of (start, end).")
        if n_dates <= 0 or n_iter <= 0:
            raise ValueError("n_dates and n_iter must be positive integers.")

        self.date_range = date_range
        self.n_dates = n_dates
        self.n_iter = n_iter
        self.model = model
        self.lam = lam
        self.errors = errors
        self.spds: List[SPD] = []

    def _generate_random_dates(self) -> List[Date]:
        """
        Generates random `Date` objects based on the specified model.

        Returns:
            List[Date]: A list of randomly generated `Date` objects.
        """

        # Uniform model
        if self.model == 'uniform':
            years = np.random.choice(
                np.arange(self.date_range[0], self.date_range[1] + 1), self.n_dates, replace=True
            )

        # Exponential model
        elif self.model == 'exp':
            probs = np.exp(-self.lam * np.arange(self.date_range[0], self.date_range[1]))
            years = np.random.choice(
                np.arange(self.date_range[0], self.date_range[1]), self.n_dates, replace=True,
                p=probs / probs.sum()
            )

        # Unsupported model
        else:
            raise ValueError("Model not supported yet. Choose between 'uniform' and 'exp'.")

        curve = CALIBRATION_CURVES["intcal20"]
        c14ages = [curve[np.argmin(np.abs(curve[:, 0] - year)), 1] for year in years]

        # Randomly sample errors
        errors = np.random.choice(self.errors, self.n_dates)

        return [Date(c14age, error) for c14age, error in zip(c14ages, errors)]

    def simulate_spds(self) -> np.ndarray:
        """
        Simulates SPDs and calculates percentile bounds.

        Returns:
            np.ndarray: A 2D array with age range, lower percentile, and upper percentile.
        """
        self.spds = [self._create_spd(self._generate_random_dates())
                     for _ in range(self.n_iter)]

        min_age = min(spd.summed[0, 0] for spd in self.spds)
        max_age = max(spd.summed[-1, 0] for spd in self.spds)
        age_range = np.arange(min_age, max_age)

        prob_matrix = self._create_probability_matrix(age_range)

        lower_percentile = np.percentile(prob_matrix[:, 1:], 2.5, axis=1)
        upper_percentile = np.percentile(prob_matrix[:, 1:], 97.5, axis=1)

        return np.column_stack((age_range, lower_percentile, upper_percentile))

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
        self.simulations: Optional[np.ndarray] = None

        self.date_range = date_range if date_range else (
            int(min(date.median() for date in spd.dates)),
            int(max(date.median() for date in spd.dates)),
        )

        self.n_dates = len(spd.dates)

    def simulate(self, n_iter: int = 1000, model: str = 'exp') -> None:
        """
        Runs simulations using the same time range and number of dates as the real SPD.

        Args:
            n_iter (int): Number of iterations for the simulation. Default is 1000.
            model (str): Model for date generation ('uniform' or 'exp'). Default is 'exp'.
        """
        errors = [date.c14sd for date in self.spd.dates]
        if model == 'exp':
            ages = self.spd.summed[:, 0]
            probs = self.spd.summed[:, 1] + 1e-10

            x = ages[(ages > self.date_range[0]) & (ages < self.date_range[1])]
            y = probs[(ages > self.date_range[0]) & (ages < self.date_range[1])]

            lam = -np.polyfit(x, np.log(y), 1)[0]
            sim_spd = SimSPD(
                date_range=self.date_range,
                n_dates=self.n_dates,
                n_iter=n_iter,
                errors=errors,
                model=model,
                lam=lam
            )
        elif model == 'uniform':
            sim_spd = SimSPD(
                date_range=self.date_range,
                n_dates=self.n_dates,
                n_iter=n_iter,
                errors=errors,
                model=model
            )
        else:
            raise ValueError("Model not supported yet. Choose between 'uniform' and 'exp'.")

        self.simulations = sim_spd.simulate_spds()

    def _extract_intervals(self):
        """
        Identifies intervals where the observed SPD is above or below the confidence envelope.

        Returns:
            Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]: A tuple containing:
                - above_intervals: List of intervals where SPD is above the envelope.
                - below_intervals: List of intervals where SPD is below the envelope.
        """
        observed_ages = self.spd.summed[:, 0]
        observed_probs = self.spd.summed[:, 1]
        lower_ci = np.interp(observed_ages, self.simulations[:, 0], self.simulations[:, 1])
        upper_ci = np.interp(observed_ages, self.simulations[:, 0], self.simulations[:, 2])

        above_intervals, below_intervals = [], []
        current_interval = []
        is_above = None

        for i, (age, prob, low, high) in enumerate(zip(observed_ages, observed_probs, lower_ci, upper_ci)):
            if prob > high:  # Above the envelope
                if not current_interval or not is_above:
                    current_interval = [age, age]
                    is_above = True
                else:
                    current_interval[1] = age
            elif prob < low:  # Below the envelope
                if not current_interval or is_above:
                    current_interval = [age, age]
                    is_above = False
                else:
                    current_interval[1] = age
            else:  # Within the envelope
                if current_interval:
                    if is_above:
                        above_intervals.append(tuple(current_interval))
                    else:
                        below_intervals.append(tuple(current_interval))
                    current_interval = []

        # Add any ongoing interval
        if current_interval:
            if is_above:
                above_intervals.append(tuple(current_interval))
            else:
                below_intervals.append(tuple(current_interval))

        return above_intervals, below_intervals

    def plot(self):
        """
        Plots the real SPD overlaid on the simulated confidence intervals.
        """
        if self.simulations is None:
            raise ValueError("Simulations must be run before plotting.")

        # Plot confidence intervals
        plt.fill_between(
            self.simulations[:, 0],  # Age range
            self.simulations[:, 1],  # Lower CI
            self.simulations[:, 2],  # Upper CI
            color="lightgray",
            label="95% CI",
        )

        # Plot the real SPD
        plt.plot(
            self.spd.summed[:, 0],  # Age range
            self.spd.summed[:, 1],  # Probability density
            color="black",
            label="SPD",
        )

        above_intervals, below_intervals = self._extract_intervals()

        # Highlight above intervals in red (only add one legend entry)
        for i, (start, end) in enumerate(above_intervals):
            plt.fill_betweenx(
                [0, self.spd.summed[:, 1].max()],
                start,
                end,
                color="red",
                alpha=0.3,
                label="Above CI" if i == 0 else None,
            )

        # Highlight below intervals in blue (only add one legend entry)
        for i, (start, end) in enumerate(below_intervals):
            plt.fill_betweenx(
                [0, self.spd.summed[:, 1].max()],
                start,
                end,
                color="blue",
                alpha=0.3,
                label="Below CI" if i == 0 else None,
            )

        plt.gca().invert_xaxis()
        plt.xlim(self.date_range[1], self.date_range[0])
        plt.xlabel("Calibrated Age (BP)")
        plt.ylabel("Probability Density")
        plt.title("SPD with Simulated Confidence Intervals")
        plt.legend()
        plt.show()

