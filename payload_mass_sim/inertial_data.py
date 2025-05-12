import numpy as np

class InertialData:
    def __init__(self, inertial_direction:np.ndarray, mass_accel_curve=None, accel_grav:float=9.81):
        self.inertial_direction = np.array(inertial_direction)
        self.accel_grav = accel_grav # could change if in different units

        # gives the load factor n as a function of mass
        if mass_accel_curve is None:
            mass_accel_curve = lambda m : 1.0
        self.mass_accel_curve = mass_accel_curve

    def get_load_factor(self, mass:float):
        return self.mass_accel_curve(mass)