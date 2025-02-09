# Radiocarbon Date Calibration and Analysis

![PyPI](https://img.shields.io/pypi/v/radiocarbon)
![PyPI - Downloads](https://img.shields.io/pypi/dm/radiocarbon)

This package provides tools for calibrating radiocarbon dates, calculating Summed Probability Distributions (SPDs), and performing statistical tests on SPDs using simulated data.

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
date = Date(c14age=3000, c14sd=30)
date.calibrate(curve="intcal20")

# Calibrate multiple dates
dates = Dates(c14ages=[3000, 3200, 3100], c14sds=[30, 25, 35])
dates.calibrate()

# Plot a calibrated date
date.plot()
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
