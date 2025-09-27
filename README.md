Event driven molecular dynamics code in 3D written in Python.

Accompanying the main EDMD code is functionality for the Lubachevsky-Stillinger procedure for generating initial conditions.

The code is a simplified implementation of the algorithm detailed in "Efficient event-driven simulations of hard spheres," F. Smallenberg, Eur. Phys. J. E (2022) 45:22

These codes were used to generate the data in the manuscript
“Non-Equilibrium Dynamics of Hard Spheres in the Fluid, Crystalline, and Glassy Regimes,” M. Kafker, X. Arsiwalla, Phys. Rev. E 112, 034103 (2025) (Editor’s Suggestion), https://arxiv.org/abs/2504.04599, (2025).

One runs the code using "hs_edmd.py" by providing initial positions, initial velocities, simulation duration, box size, as well as a number of flags. The code calculates trajectories of particles, and also optionally computes the causal graph, see Phys. Rev. E paper above.
