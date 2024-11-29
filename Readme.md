# Streetscape Morphometrics: Expanding Momepy to analyze urban form from the street point of view

This repository contains the development code, example code and example data for the _Streetscape Morphometrics: Expanding Momepy to analyze urban form from the street point of view_ academic paper.

- Alessandro Araldi (1) [alessandro.araldi@univ-cotedazur.fr]
- Martin Fleischmann (2) [martin.fleischmann@natur.cuni.cz]
- Giovanni Fusco (1) [giovanni.fusco@univ-cotedazur.fr]
- Marek Novotný (2) [marek.novotny@natur.cuni.cz]

1. Université Côte d’Azur-CNRS-AMU-AU, ESPACE, 98 Bd Edouard Herriot, 06200, Nice, France
2. Department of Social Geography and Regional Development, Charles University, Albertov 6  128 00 Praha 2, Czech Republic

## Streetscape implementation

The final version of the streetscape morphometrics is included in the version 0.9.1 of `momepy`. Please see the `streetscape.py` module within [`pysal/momepy` repository](https://github.com/pysal/momepy).

## Running the example

You can reproduce the `example-france.ipynb` using the environment defined by [Pixi](https://pixi.sh).

Install Pixi if you don't have it and then install the locked environment:

```sh
pixi install
```

Use the environment to open Jupyter Lab and execute the notebook.

```sh
pixi run jupyter lab
```

## Output files

The files in the `data/output` folder are generated using the `momepy.Streetscape` class and are saved as Apache GeoParquet files, to reduce their footprint. GeoParquet can be opened using `geopandas.read_parquet` and converted to any other file type as needed.


## License

The data is available under the [Open License 2.0](https://geoservices.ign.fr/cgu-licences) © IGN - 2024.

The code is available under MIT License.