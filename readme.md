## Inverted Pendulum Walking Model Research

This repository contains the code used for research into using an inverted pendulum (IP) model to understand walking stability. This research augments the IP model by adding a dynamic center of pressure (CoP) and using a novel basin-switching regulator to parametrically control the CoP to maximize robustness. 

Below is a list of the different results that can be obtained: 

* `basinswitch_res_basin.py`: Recovery basin results for the model with basin-switching. 
* `basinswitch_res_traj.py`: Trajectory results for the model with basin-switching. 
* `timevarcop_res_basin.py`: Recovery basin and convergence plot results for the model. 
* `timevarcop_res_traj.py`: Trajectory results for the model. 
* `walkingmodel_basin_1.py`: Generate trajectories for various initial conditions, which in turn, can be used to create recovery basins or convergence plots. 



