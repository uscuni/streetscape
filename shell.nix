{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    # nativeBuildInputs is usually what you want -- tools you need to run
    nativeBuildInputs = with pkgs.buildPackages; [
      python311Packages.pandas
      python311Packages.geopandas
      python311Packages.matplotlib
      python311Packages.rtree
      python311Packages.shapely
      python311Packages.numpy
      python311Packages.toml
      python311Packages.scikit-learn
      python311Packages.seaborn
      python311Packages.jupyter
      python311Packages.rasterio
      ];
}
