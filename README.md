# Python Package for the Rosenbluth Fokker Planck (RFP) Equation

> Currently, heavily under renovation (refactoring) from my old code (in my other private repository)

## Description

This package is refactored version of a part of the `pystops_ml` code.
I've separated only data generation and training part from `pystops_ml`.

Unlike `pystops_ml`, this module doesn't utilize distributed training. (`DDP` feature not needed)

This code is part of my paper, Data-Driven Stochastic Particle Scheme for Collisional Plasma Simulations.

- Preprint is available at [SSRN](https://ssrn.com/abstract=4108990)

## Dependencies

- Global
- `python >=3.10`
- `torch >= 1.13.1`

- Personal project
  - `pymaxed` (for the Maximum Entropy Distribution)
  - `pymytools` (miscellaneous tools including data IO)
