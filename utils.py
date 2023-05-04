import numpy as np
import math
from intersect import intersection
"""
This first section is all about boundary/wall distance functions
"""

def ellipse_wdf(theta, system_params):
    # returns the value of the wall distance function for a given theta, for an ellipse

    a, b, X_rot = system_params["a"], system_params["b"], system_params["X_rot"]

    if abs(X_rot) > a:
        raise Exception("X_rot is outside the ellipse")
    else:
        pass

    return np.sqrt(a ** 2 * np.sin(theta) ** 2 + b ** 2 * np.cos(theta) ** 2) + X_rot * np.sin(theta)

def ellipse_wdf_diff(theta, system_params):
    # returns the differentiated value of the WDF for a given theta

    a, b, X_rot = system_params["a"], system_params["b"], system_params["X_rot"]

    if abs(X_rot) > a:
        raise Exception("X_rot is outside the ellipse")
    else:
        pass

    return X_rot * np.cos(theta) + 1/2 * (a*a - b*b) * np.sin(2 * theta) / np.sqrt(a ** 2 * np.sin(theta) ** 2 + b ** 2 * np.cos(theta) ** 2)

def ellipse_boundary(theta, system_params):
    # Chen-Thiffeault (2.20)
    # returns the values ([lower, upper]) of the boundary in (y, theta) configuration space for an ellipse in a channel of width L, centered at y = 0

    return [-0.5 * system_params["L"] + ellipse_wdf(theta, system_params), 0.5 * system_params["L"] - ellipse_wdf(theta + np.pi, system_params)]

def ellipse_check_channel(configuration, system_params):
    # returns True if configuration (= [x, y, theta]), lies within the allowed configuration space, returns False if not

    y, theta = configuration[1], configuration[2]
    boundary = ellipse_boundary(theta, system_params)

    if y < boundary[1] and y > boundary[0]:
        return True
    else:
        return False 

def ellipse_check_circle(configuration, system_params):
    # returns True if configuration (= [x, y, theta]), lies within the allowed configuration space, returns false if not
    # A check that the curvature of the ellipse is greater than that of the circle will be done elsewhere

    x, y, theta = configuration[0], configuration[1], configuration[2]
    R = system_params["R"]
    phi = math.atan2(y,x)
    r = math.sqrt(x**2 + y**2)

    if r < R - ellipse_wdf(theta - phi - math.pi/2, system_params):
        return True
    else:
        return False

def ellipse_check_square(configuration, system_params):


    x, y, theta = configuration[0], configuration[1], configuration[2]
    y_boundary = ellipse_boundary(theta, system_params)
    x_boundary = ellipse_boundary(theta + np.pi/2, system_params) # works on symmetry grounds

    if y < y_boundary[1] and y > y_boundary[0] and x < x_boundary[1] and x > x_boundary[0]:
        return True
    else:
        return False

"""
This section is all about the simulation of trajectories (Note there is a lot of copy and paste)
"""

def normalise_angles(angles):
    # Function to normalise the angles to the range (-pi, +pi]

    signs = np.sign(angles)

    Ntraj = np.shape(angles)[0]
    
    for j in range(Ntraj):
        for i in range(np.shape(angles)[1]):
            if signs[j, i] == 1:
                angles[j, i] += - math.floor(angles[j, i] / (2*math.pi)) * 2 * math.pi
            elif signs[j, i] == -1:
                angles[j, i] += - math.ceil(angles[j, i] / (2*math.pi)) * 2 * math.pi
            else:
                pass
            if np.abs(angles[j, i]) < math.pi:
                pass
            elif np.abs(angles[j, i]) >= math.pi:
                angles[j, i] += -2 * math.pi * signs[j, i]
            else:
                raise Exception("Something wrong in normalising the angles")
    
    return angles

"""
Channel Geometry
"""

def ellipse_simulation_rejection_channel(system_params):
    """
    Returns a simulated trajectory of an ellipse as a numpy array of shape (N+1, 3)
    """

    # loading in system parameters
    T = system_params["T"]
    N = system_params["N"]
    U = system_params["U"]
    initial_configuration = system_params["initial_configuration"]

    dt = T / N # timestep

    # diffusion parameters
    A = np.sqrt(2 * system_params["D_X"] * dt)
    B = np.sqrt(2 * system_params["D_Y"] * dt)
    C = np.sqrt(2 * system_params["D_theta"] * dt)

    # initialising trajectory array
    trajectory = np.empty((N + 1, 3))

    if ellipse_check_channel(initial_configuration, system_params) == True:
        trajectory[0, :] = initial_configuration
    else:
        raise Exception("inital configuration outside valid configuration space")

    # calculating gaussian random variables for simulation
    gauss1 = np.random.normal(loc=0.0, scale=1.0, size=(N))
    gauss2 = np.random.normal(loc=0.0, scale=1.0, size=(N))
    gauss3 = np.random.normal(loc=0.0, scale=1.0, size=(N))

    # beginning simulation
    for i in range(0, N):
        x, y, theta = trajectory[i, 0], trajectory[i, 1], trajectory[i ,2]
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        # Chen-Thiffeault (3.2)
        dx = (U * dt + A * gauss1[i]) * cos_theta - sin_theta * B * gauss2[i]
        dy = (U * dt + A * gauss1[i]) * sin_theta + cos_theta * B * gauss2[i]
        dtheta = C * gauss3[i]

        trajectory[i + 1, 0] = x + dx
        trajectory[i + 1, 1] = y + dy
        trajectory[i + 1, 2] = theta + dtheta

        # implementing rejection algorithm
        if ellipse_check_channel(configuration = trajectory[i + 1, :], system_params = system_params) == True:
            pass
        else:
            fail_count = 0
            while ellipse_check_channel(configuration = trajectory[i + 1, :], system_params = system_params) == False:
                # generate new gaussian random numbers
                newgauss1, newgauss2, newgauss3 = np.random.normal(loc = 0.0, scale = 1.0), np.random.normal(loc = 0.0, scale = 1.0), np.random.normal(loc = 0.0, scale = 1.0)
            
                dx = (U * dt + A * newgauss1) * cos_theta - sin_theta * B * newgauss2
                dy = (U * dt + A * newgauss1) * sin_theta + cos_theta * B * newgauss2
                dtheta = C * newgauss3

                trajectory[i + 1, 0] = x + dx
                trajectory[i + 1, 1] = y + dy
                trajectory[i + 1, 2] = theta + dtheta

                fail_count += 1
                if fail_count > 10000:
                    raise Exception("When implementing rejection sampling method, failed more than 1000 times to find a valid next step")
            else:
                pass

    return trajectory

def ellipse_simulation_reflection_channel(system_params):
    """
    Returns a simulated trajectory of an ellipse as a numpy array of shape (N+1, 3)

    The reflection algorithm is as follows:
    - simulate as normal until an invalid configuration is reached.
    - reflect the configuration
    - if this reflection is valid, proceed
    - if not use rejection method to obtain next configuration.
    - keep track of number of rejection steps
    """

    # loading in system parameters
    T = system_params["T"]
    N = system_params["N"]
    U = system_params["U"]
    initial_configuration = system_params["initial_configuration"]

    dt = T / N # timestep

    # diffusion parameters
    A = np.sqrt(2 * system_params["D_X"] * dt)
    B = np.sqrt(2 * system_params["D_Y"] * dt)
    C = np.sqrt(2 * system_params["D_theta"] * dt)

    # initialising trajectory array
    trajectory = np.empty((N + 1, 3))

    if ellipse_check_channel(initial_configuration, system_params) == True:
        trajectory[0, :] = initial_configuration
    else:
        raise Exception("inital configuration outside valid configuration space")


    # calculating gaussian random variables for simulation
    gauss1 = np.random.normal(loc=0.0, scale=1.0, size=(N))
    gauss2 = np.random.normal(loc=0.0, scale=1.0, size=(N))
    gauss3 = np.random.normal(loc=0.0, scale=1.0, size=(N))

    rejection_count = 0
    reflection_count = 0

    # beginning simulation
    for i in range(0, N):
        
        x, y, theta = trajectory[i, 0], trajectory[i, 1], trajectory[i ,2]
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        # Chen-Thiffeault (3.2)
        dx = (U * dt + A * gauss1[i]) * cos_theta - sin_theta * B * gauss2[i]
        dy = (U * dt + A * gauss1[i]) * sin_theta + cos_theta * B * gauss2[i]
        dtheta = C * gauss3[i]

        trial_configuration = [x+dx,y+dy,theta+dtheta]

        # implementing reflection algorithm
        if ellipse_check_channel(configuration = trial_configuration, system_params = system_params) == True:
            trajectory[i+1,:] = trial_configuration
        else:
            reflection_count += 1
            reflected_configuration = generate_reflected_configuration_channel2(trajectory[i,:], trial_configuration, system_params)
            if ellipse_check_channel(reflected_configuration, system_params) == True:
                trajectory[i+1,:] = reflected_configuration
            else:
                fail_count = 0
                rejection_count += 1
                trajectory[i+1,:] = reflected_configuration # We already know this won't pass ellipse_channel check, therefore it will trigger the while clause
                while ellipse_check_channel(configuration = trajectory[i + 1, :], system_params = system_params) != True:
                    # generate new gaussian random numbers
                    newgauss1, newgauss2, newgauss3 = np.random.normal(loc = 0.0, scale = 1.0), np.random.normal(loc = 0.0, scale = 1.0), np.random.normal(loc = 0.0, scale = 1.0)
                
                    dx = (U * dt + A * newgauss1) * cos_theta - sin_theta * B * newgauss2
                    dy = (U * dt + A * newgauss1) * sin_theta + cos_theta * B * newgauss2
                    dtheta = C * newgauss3

                    trajectory[i + 1, 0] = x + dx
                    trajectory[i + 1, 1] = y + dy
                    trajectory[i + 1, 2] = theta + dtheta

                    fail_count += 1
                    if fail_count > 10000:
                        raise Exception("When implementing rejection sampling method, failed more than 1000 times to find a valid next step")
                else:
                    pass
    
    print(f'Number of rejection steps required = {rejection_count}')
    print(f'Number of reflections required = {reflection_count}')
    print(f'rejection percentage = {100.0 * rejection_count/reflection_count}')
            
    return trajectory

def generate_reflected_configuration_channel2(configuration, failed_configuration, system_params):

    x0, y0, theta0 = configuration[0], configuration[1], configuration[2]
    x1, y1, theta1 = failed_configuration[0], failed_configuration[1], failed_configuration[2]

    # if ellipse_check_channel(configuration = [x0,y0,theta0], system_params=system_params) == False:
    #     raise Exception("Configuration given to reflection algorithm already outside configuration space")

    #Finding the intersction point(s) of the trajectory and boundary
    s = np.linspace(0,1,30)

    theta = theta0 + s*(theta1 - theta0)
    y_traj = y0 + s*(y1 - y0)

    y_lower = ellipse_boundary(theta, system_params)[0]
    y_upper = ellipse_boundary(theta, system_params)[1]

    intersection_lower = intersection(theta, y_traj, theta, y_lower)
    intersection_upper = intersection(theta, y_traj, theta, y_upper)

    """
    Cases, that would be triggered by a failed configuration fromma starting configuration, open domain only:
        - 1 intersection with either boundary
        - even intersections with one and odd intersections with the other. this is very unlikely, especially as dt gets smaller. Suggest use of dt must be order of magnitude less than distnace between boundary walls
    if even intersections for both boundaries, then it's still in the domain and we assume it takes a path around the wall instead. this should not cause the reflection algorithm to run.
    """

    print(f'lower_intersection = {intersection_lower}')
    print(f'upper intersection = {intersection_upper}')

    if np.shape(intersection_lower)[1] == 1:
        reflected_configuration = reflect_lower(original_configuration=configuration, failed_configuration=failed_configuration, intersection_point=intersection_lower, system_params=system_params)
    elif np.shape(intersection_upper)[1] == 1 and np.shape(intersection_lower)[1] == 0:
        reflected_configuration = reflect_upper(original_configuration=configuration, failed_configuration=failed_configuration, intersection_point=intersection_upper, system_params=system_params)
    else:
        reflected_configuration = [0, system_params["L"], 0] # a configuration that isn't valid - this triggers rejection sampling for this iteration step

    return reflected_configuration

def reflect_lower(original_configuration, failed_configuration, intersection_point, system_params):
    # Defining an orientation as [theta,y] position vector
    orientation1 = [failed_configuration[2], failed_configuration[1]]
    crit_orientation = np.asarray(intersection_point, dtype=np.float64)

    n_ = np.asarray([-ellipse_wdf_diff(crit_orientation[0], system_params), 1.0], dtype=np.float64) # unnormalised normal vector
    mod_n_ = np.sqrt(n_[0]*n_[0] + n_[1]*n_[1])
    N_ = n_ / mod_n_ # Normalised normal vector

    # print(f'N_ = {N_}')
    projection = (orientation1[0] - crit_orientation[0]) * N_[0] + (orientation1[1] - crit_orientation[1]) * N_[1]
    reflected_orientation = orientation1 - 2 * (projection) * N_

    # print(f'projection = {projection}')

    return [failed_configuration[0], reflected_orientation[1], reflected_orientation[0]]

def reflect_upper(original_configuration, failed_configuration, intersection_point, system_params):
    # Defining an orientation as [theta,y] "position" vector
    orientation1 = [failed_configuration[2], failed_configuration[1]]
    crit_orientation = np.asarray(intersection_point, dtype=np.float64)

    n_ = np.asarray([-ellipse_wdf_diff(crit_orientation[0] + np.pi, system_params), -1.0], dtype=np.float64) # Unnormalised normal vector
    mod_n_ = np.sqrt(n_[0]*n_[0] + n_[1]*n_[1])
    N_ = n_ / mod_n_ # Normalised normal vector

    projection = (orientation1[0] - crit_orientation[0]) * N_[0] + (orientation1[1] - crit_orientation[1]) * N_[1]
    reflected_orientation = orientation1 - 2 * (projection) * N_ 

    return [failed_configuration[0], reflected_orientation[1], reflected_orientation[0]]

def generate_reflected_configuration_channel(configuration, failed_configuration, system_params, M = 32):

    x0, y0, theta0 = configuration[0], configuration[1], configuration[2]
    x1, y1, theta1 = failed_configuration[0], failed_configuration[1], failed_configuration[2]

    # if ellipse_check_channel(configuration = [x0,y0,theta0], system_params=system_params) == False:
    #     raise Exception("Configuration given to reflection algorithm already outside configuration space") 
    
    #Find the critical point on the boundary
    ts = np.linspace(0, 1, num = M)
    count = 0

    for t in ts: # There is a problem here when doing multiple reflections. Our configuration does not lie in the allowed configuration space if our critical t is ts[count], therefore we must use ts[count - 1]
        y = y0 + t * (y1 - y0)
        theta = theta0 + t * (theta1 - theta0)
        if ellipse_check_channel([x0, y, theta], system_params) == False: # NOTE: x coordinate makes no difference in this geometry so we don't bother here
            break
        else:
            count += 1

    t_crit = ts[count - 1]
    crit_configuration = [x0 + t_crit * (x1 - x0), y0 + t_crit * (y1 - y0), theta0 + t_crit * (theta1 - theta0)]

    if  y1 < ellipse_boundary(theta1, system_params)[0]: # i.e. we are below the (y,theta) allowed region) NOTE: we are only allowed to do this in the open channel domain, although the choice of y to test the boundary would make the problem more managable in that case
        n_ = [ellipse_wdf_diff(crit_configuration[2], system_params), -1]
    elif y1 > ellipse_boundary(theta1, system_params)[1]:
        n_ = [-ellipse_wdf_diff(crit_configuration[2] + np.pi, system_params), -1]
    else:
        raise Exception("failed configuration didn't fail the boundary test")
    
    projection_unnormalised = (n_[0] * (theta1-theta0)*(1-t_crit) + n_[1] * (y1 - y0)*(1-t_crit))
    new_configuration = [x1, y0 - 2 * projection_unnormalised * n_[1] / (n_[0]*n_[0] + n_[1]*n_[1]), theta0 - 2 * projection_unnormalised * n_[0] / (n_[0]*n_[0] + n_[1]*n_[1])]

    return new_configuration 

"""
Square Geometry
"""

def ellipse_simulation_rejection_square(system_params):
    """
    Returns a simulated trajectory of an ellipse as a numpy array of shape (N+1, 3)
    """

    # loading in system parameters
    T = system_params["T"]
    N = system_params["N"]
    U = system_params["U"]
    initial_configuration = system_params["initial_configuration"]

    dt = T / N # timestep

    # diffusion parameters
    A = np.sqrt(2 * system_params["D_X"] * dt)
    B = np.sqrt(2 * system_params["D_Y"] * dt)
    C = np.sqrt(2 * system_params["D_theta"] * dt)

    # initialising trajectory array
    trajectory = np.empty((N + 1, 3))

    if ellipse_check_square(initial_configuration, system_params) == True:
        trajectory[0, :] = initial_configuration
    else:
        raise Exception("inital configuration outside valid configuration space")


    # calculating gaussian random variables for simulation
    gauss1 = np.random.normal(loc=0.0, scale=1.0, size=(N))
    gauss2 = np.random.normal(loc=0.0, scale=1.0, size=(N))
    gauss3 = np.random.normal(loc=0.0, scale=1.0, size=(N))

    # beginning simulation
    for i in range(0, N):
        x, y, theta = trajectory[i, 0], trajectory[i, 1], trajectory[i ,2]
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        # Chen-Thiffeault (3.2)
        dx = (U * dt + A * gauss1[i]) * cos_theta - sin_theta * B * gauss2[i]
        dy = (U * dt + A * gauss1[i]) * sin_theta + cos_theta * B * gauss2[i]
        dtheta = C * gauss3[i] 

        trajectory[i + 1, 0] = x + dx
        trajectory[i + 1, 1] = y + dy
        trajectory[i + 1, 2] = theta + dtheta

        # implementing rejection algorithm
        if ellipse_check_square(configuration = trajectory[i + 1, :], system_params = system_params) == True:
            pass
        else:
            fail_count = 0
            while ellipse_check_square(configuration = trajectory[i + 1, :], system_params = system_params) == False:
                # generate new gaussian random numbers
                newgauss1, newgauss2, newgauss3 = np.random.normal(loc = 0.0, scale = 1.0), np.random.normal(loc = 0.0, scale = 1.0), np.random.normal(loc = 0.0, scale = 1.0)
            
                dx = (U * dt + A * newgauss1) * cos_theta - sin_theta * B * newgauss2
                dy = (U * dt + A * newgauss1) * sin_theta + cos_theta * B * newgauss2
                dtheta = C * newgauss3

                trajectory[i + 1, 0] = x + dx
                trajectory[i + 1, 1] = y + dy
                trajectory[i + 1, 2] = theta + dtheta

                fail_count += 1
                if fail_count > 1000:
                    raise Exception("When implementing rejection sampling method, failed more than 1000 times to find a valid next step")
            else:
                pass


    return trajectory

"""
Circle Geometry
"""

def ellipse_simulation_rejection_circle(system_params):
    """
    Returns a simulated trajectory of an ellipse as a numpy array of shape (N+1, 3)
    """

    # loading in system parameters
    T = system_params["T"]
    N = system_params["N"]
    U = system_params["U"]
    initial_configuration = system_params["initial_configuration"]

    dt = T / N # timestep

    # diffusion parameters
    A = np.sqrt(2 * system_params["D_X"] * dt)
    B = np.sqrt(2 * system_params["D_Y"] * dt)
    C = np.sqrt(2 * system_params["D_theta"] * dt)

    # initialising trajectory array
    trajectory = np.empty((N + 1, 3))

    if ellipse_check_circle(initial_configuration, system_params) == True:
        trajectory[0, :] = initial_configuration
    else:
        raise Exception("inital configuration outside valid configuration space")


    # calculating gaussian random variables for simulation
    gauss1 = np.random.normal(loc=0.0, scale=1.0, size=(N))
    gauss2 = np.random.normal(loc=0.0, scale=1.0, size=(N))
    gauss3 = np.random.normal(loc=0.0, scale=1.0, size=(N))

    # beginning simulation
    for i in range(0, N):
        x, y, theta = trajectory[i, 0], trajectory[i, 1], trajectory[i ,2]
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        # Chen-Thiffeault (3.2)
        dx = (U * dt + A * gauss1[i]) * cos_theta - sin_theta * B * gauss2[i]
        dy = (U * dt + A * gauss1[i]) * sin_theta + cos_theta * B * gauss2[i]
        dtheta = C * gauss3[i] 

        trajectory[i + 1, 0] = x + dx
        trajectory[i + 1, 1] = y + dy
        trajectory[i + 1, 2] = theta + dtheta

        # implementing rejection algorithm
        if ellipse_check_circle(configuration = trajectory[i + 1, :], system_params = system_params) == True:
            pass
        else:
            fail_count = 0
            while ellipse_check_circle(configuration = trajectory[i + 1, :], system_params = system_params) == False:
                # generate new gaussian random numbers
                newgauss1, newgauss2, newgauss3 = np.random.normal(loc = 0.0, scale = 1.0), np.random.normal(loc = 0.0, scale = 1.0), np.random.normal(loc = 0.0, scale = 1.0)
            
                dx = (U * dt + A * newgauss1) * cos_theta - sin_theta * B * newgauss2
                dy = (U * dt + A * newgauss1) * sin_theta + cos_theta * B * newgauss2
                dtheta = C * newgauss3

                trajectory[i + 1, 0] = x + dx
                trajectory[i + 1, 1] = y + dy
                trajectory[i + 1, 2] = theta + dtheta

                fail_count += 1
                if fail_count > 10000:
                    raise Exception("When implementing rejection sampling method, failed more than 1000 times to find a valid next step")
            else:
                pass


    return trajectory