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

```
from radiocarbon import Date, Dates

# Create a single radiocarbon date
date = Date(c14age=3000, c14sd=30, curve="intcal20")
date.calibrate()

# Calibrate multiple dates
dates = Dates(c14ages=[3000, 3200, 3100], c14sds=[30, 25, 35], curves=["intcal20", "intcal20", "shcal20"])
dates.calibrate()

# Plot a calibrated date
date.plot()
```

Supposing you have a CSV file with radiocarbon dates, you can read the file and calibrate the dates as follows:

```
import pandas as pd
from radiocarbon import Dates

# Read dates from a CSV file
df = pd.read_csv("dates.csv")

# Create a Dates object from the DataFrame
dates = Dates(df["c14age"], df["c14sd"], df["curve"])
dates.calibrate()
```

### Calculating Summed Probability Distributions (SPDs)

```
from radiocarbon import SPD

# Create an SPD from a collection of dates
spd = SPD(dates)
spd.sum()

# Plot the SPD
spd.plot()
```

### Simulating SPDs and Testing

```
from radiocarbon import SPDTest

# Test an observed SPD against simulations
spd_test = SPDTest(spd, date_range=(3000, 3500))
spd_test.simulate(n_iter=1000, model="uniform")
spd_test.plot()
```

### Binning Dates

Binning can be performed to account for oversampling.

```
from radiocarbon import Dates, Bins, SPD

# Create a Dates object
df = pd.read_csv("dates.csv")
dates = Dates(df["c14age"], df["c14sd"], df["curve"])

# Bin the dates by site using a window of 100 years
bins = Bins(dates, labels=df["site"], bin_size=100)

# Calculate the SPD from the binned dates
spd = SPD(bins.bins)
spd.sum()
spd.plot()
```
