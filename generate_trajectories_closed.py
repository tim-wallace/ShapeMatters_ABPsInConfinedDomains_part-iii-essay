import numpy as np
import math
import scipy.io
from utils import *

import time

"""
This script generates Ntraj trajectories for an ellipse in an closed channel domain, with all of the same parameters,
which are then uploaded to matlab for further analysis.

The x coordinate is tracked in this case.
"""

start = time.time()

system_params = {
  "U": 1,
  "T": 100000,
  "N": 10000000,
  "L": 1,
  "a": 0.55,
  "b": 0.05,
  "X_rot": -0.25,
  "D_X": 0.1,
  "D_Y": 0.1,
  "D_theta": 0.01,
  "initial_configuration": [0,0,0]
}
seed = 1234
np.random.seed(seed)
Ntraj = 1

trajectories = np.empty((Ntraj, system_params["N"] + 1, 3))

starttraj = time.time()

for i in range(Ntraj):
    trajectories[i,:,:] = ellipse_simulation_rejection_channel(system_params=system_params)

endtraj = time.time()

scipy.io.savemat(f'trajectory_data_final/channel/closed/rejection/Ntraj={Ntraj}_T={system_params["T"]}_dt={system_params["T"]/system_params["N"]}_dth={system_params["D_theta"]}.mat', dict(x = trajectories[:, :, 0], 
                                                y= trajectories[:, :, 1], 
                                                theta = trajectories[:, :, 2],
                                                system_params = system_params))

end = time.time()

print(f'simulation runtime = {endtraj -starttraj}s')
print(f'total runtime = {end -start}s')
