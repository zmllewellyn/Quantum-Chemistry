
# Art of Scientific Computing Code

The draft code used to generate the plots for Zac Llewellyns draft report.

Needed packages
- Numpy
- Scipy
- Matplotlib
- Numba
- Adaptive

Although knowledge of Numba isn't necessary for running this code it is recommended. Knowledge of Adaptive is strongly recommended though.

Radial_dependence.py uses the adaptive package to generate the data for the line plots of the internuclear seperation vs ground state energy. It can be run in the background and generates the .learner file containing the data. The Radial_plotting file however takes this .learner file and then plots it.

