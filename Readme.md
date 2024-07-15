# Repository with code for Multiple Fabric Assessment 2.0

Code by University of Cote'd Azur - Alessandro Araldi & Giovanni Fusco

Will make its way to pysal/momepy.

## Methods workflow

1. `01_street_sightlines.ipynb` defines the sightlines

2. `02_sightlines_plot.ipynb`

3. `03_sightlines_DEM.ipynb` enhances sightlines with z coords and computes slope

4. `04_street_indicators.ipynb` collects and produces street-based metrics from the point-based measures

## Repository workflow

If you commit changes in any of the four notebooks, please begin the commit message with the corresponding
number of the method (eg. 04 Cleanup of imports)

## Status

- [x] `01_street_sightlines.ipynb` running
- [x] `02_sightlines_plot.ipynb` running
- [ ] `03_sightlines_DEM.ipynb` running
> ~~We are interested only in urban morphometrics so we might skip the third part.~~
- [x] `04_street_indicators.ipynb` running

## Notes

`03_sightlines_DEM.ipynb` is not needed to run the other three methods