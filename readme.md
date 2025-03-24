# Sound field reconstruction

Repo to generate results for the paper

![Spatial reconstruction of sound fields using local and data-driven functions 
https://doi.org/10.1121/10.0008975 ](https://doi.org/10.1121/10.0008975)

A dictionary is trained on measured sound field data in one room, and tests show that it generalizes to another room. All reverberant sound fields share the same spatial statistics, which is why this works. In fact, the atoms might as well be prescribed (or trained on) analytical data, as for example the dictionary with sinc-correlation. 
Besides the fun exercise of testing the theory, the benefit of training for the reconstruction quality is minimal, also since this application is very general (non-specific acoustic scenarios). For those interested in the _reconstruction quality_ of unknown sound fields (more than dictionary learning fun here), a ![convolutional plane wave model](https://github.com/manvhah/convolutional_plane_waves) provides higher accuracy and better extrapolation. Kind of a handcrafted CNN with locally planar wave filters (fulfilling the Helmholtz equation).

All the fancy plots are in the paper, but for a high-level and short overview, you can also check out my ![https://orbit.dtu.dk/en/publications/distributed-microphone-and-array-processing-in-rooms](thesis). 


## 1) download data

    make data

## 2) set up python environment

    make env

## 3) generate figures

    make figs 

Commands might not always run through. In case, check the Makefile for the commands and run them manually.
