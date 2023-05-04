import numpy as np
import math
import scipy.io
from utils import *

import time

"""
This script generates Ntraj trajectories for an ellipse in a square domain, with all of the same parameters,
which are then uploaded to matlab for further analysis.
"""

start = time.time()

system_params = {
  "U": 10,
  "T": 10000,
  "N": 10000000,
  "L": 20,
  "a": 0.5,
  "b": 0.05,
  "X_rot": 0.2,
  "D_X": 0.05,
  "D_Y": 0.05,
  "D_theta": 1,
  "initial_configuration": [0,0,0]
}

np.random.seed(4321)
Ntraj = 20

trajectories = np.empty((Ntraj, system_params["N"] + 1, 3))

starttraj = time.time()

for i in range(Ntraj):
    trajectories[i,:,:] = ellipse_simulation_rejection_square(system_params=system_params)
    print(f'{i} Simulations have taken {time.time()-starttraj}s')

endtraj = time.time()
    
scipy.io.savemat(f'trajectory_data_final/square/rejection/Ntraj={Ntraj}_T={system_params["T"]}_dt={system_params["T"]/system_params["N"]}_dth={system_params["D_theta"]}_DX={system_params["D_X"]}_DY={system_params["D_Y"]}.mat', dict(x = trajectories[:, :, 0], 
                                                y= trajectories[:, :, 1], 
                                                theta = trajectories[:, :, 2],
                                                system_params = system_params))

end = time.time()

print(f'simulation runtime = {endtraj -starttraj}s')
print(f'total runtime = {end -start}s')
