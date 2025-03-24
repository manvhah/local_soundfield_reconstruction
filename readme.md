# Sound field reconstruction

Repo to generate results for the paper

![Spatial reconstruction of sound fields using local and data-driven functions 
https://doi.org/10.1121/10.0008975 ](https://doi.org/10.1121/10.0008975)

A fun exercise to train a dictionary on measured data.

## 1) download data

    make data

## 2) set up python environment

    make env

## 3) generate figures

    make figs 

Commands might not always run through. In case, check the Makefile for the commands and run them manually.

For those interested in the _reconstruction quality_ of unknown sound fields (more than dictionary learning fun here), a ![convolutional plane wave model](https://github.com/manvhah/convolutional_plane_waves) provides higher accuracy and better extrapolation.
