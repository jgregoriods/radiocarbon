# Radiocarbon Date Calibration and Analysis

![PyPI](https://img.shields.io/pypi/v/radiocarbon)
![PyPI - Downloads](https://img.shields.io/pypi/dm/radiocarbon)

This package provides tools for calibrating radiocarbon dates, calculating Summed Probability Distributions (SPDs), and performing statistical tests on SPDs using simulated data (Timpson et al. 2014).
Functionality is similar to that provided by the R package `rcarbon` (Crema et al. 2016, 2017).

## Features

- **Radiocarbon Date Calibration**: Calibrate individual or multiple radiocarbon dates using calibration curves (e.g., IntCal20, ShCal20).
- **Summed Probability Distributions (SPDs)**: Calculate SPDs for a collection of radiocarbon dates.
- **Simulated SPDs**: Generate simulated SPDs to test hypotheses or assess the significance of observed SPDs.
- **Statistical Testing**: Compare observed SPDs with simulated SPDs to identify significant deviations.
- **Visualization**: Plot calibrated dates, SPDs, and confidence intervals.

## Installation

To install the package, you can use the following command:

```bash
pip install radiocarbon
```

## Usage

### Calibrating Radiocarbon Dates

```python
from radiocarbon import Date, Dates

# Create a single radiocarbon date
date = Date(c14age=3000, c14sd=30, curve="intcal20").calibrate()

# Calibrate multiple dates
dates = Dates(c14ages=[3000, 3200, 3100], c14sds=[30, 25, 35], curves=["intcal20", "intcal20", "shcal20"]).calibrate()

# Plot a single calibrated date
date.plot()
```

Supposing you have a CSV file with radiocarbon dates, you can read the file and calibrate the dates as follows:

```python
import pandas as pd
from radiocarbon import Dates

# Read dates from a CSV file
df = pd.read_csv("dates.csv")

# Create a Dates object from the DataFrame
dates = Dates.from_df(df, "c14age", "c14sd", "curve").calibrate()
```

### Calculating Summed Probability Distributions (SPDs)

```python
from radiocarbon import SPD

# Create an SPD from a collection of dates
spd = SPD(dates).sum()

# Plot the SPD
spd.plot()
```

### Simulating SPDs and Testing

```python
from radiocarbon import SPDTest

# Test an observed SPD against simulations
spd_test = SPDTest(spd, date_range=(3000, 3500)).run_test(n_iter=1000, model="uniform")
spd_test.plot()
```
