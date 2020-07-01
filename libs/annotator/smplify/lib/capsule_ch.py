"""
Copyright 2016 Max Planck Society, Federica Bogo, Angjoo Kanazawa. All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPLify license here:
     http://smplify.is.tue.mpg.de/license

This script implements a Capsule object, used in the body approximation implemented
in capsule_body.py. Capsule sizes depend on body shape (and are differentiable with respect to it).
Capsules are the basis to compute an approximation based on spheres, used to compute efficiently
the interpenetration error term in sphere_collisions.py.
"""

import numpy as np
import chumpy as ch
from opendr.geometry import Rodrigues

# faces for the capsules. Useful only for visualization purposes
cap_f = np.asarray(
    [[0, 7, 6], [1, 7, 9], [0, 6, 11], [0, 11, 13], [0, 13, 10], [1, 9, 16],
     [2, 8, 18], [3, 12, 20], [4, 14, 22], [5, 15, 24], [1, 16, 19],
     [2, 18, 21], [3, 20, 23], [4, 22, 25], [5, 24, 17], [16, 17, 26],
     [22, 23, 32], [48, 18, 28], [49, 20, 30], [24, 25, 34], [25, 22, 50],
     [28, 19, 47], [30, 21, 48], [32, 23, 49], [17, 24, 51], [26, 17, 51],
     [34, 25, 50], [23, 20, 49], [21, 18, 48], [19, 16, 47], [51, 24, 34],
     [24, 15, 25], [15, 4, 25], [50, 22, 32], [22, 14, 23], [14, 3, 23],
     [20, 21, 30], [20, 12, 21], [12, 2, 21], [18, 19, 28], [18, 8, 19],
     [8, 1, 19], [47, 16, 26], [16, 9, 17], [9, 5, 17], [10, 15, 5],
     [10, 13, 15], [13, 4, 15], [13, 14, 4], [13, 11, 14], [11, 3, 14],
     [11, 12, 3], [11, 6, 12], [6, 2, 12], [9, 10, 5], [9, 7, 10], [7, 0, 10],
     [6, 8, 2], [6, 7, 8], [7, 1, 8], [29, 36, 41], [31, 37, 44], [33, 38, 45],
     [35, 39, 46], [27, 40, 42], [42, 46, 43], [42, 40, 46], [40, 35, 46],
     [46, 45, 43], [46, 39, 45], [39, 33, 45], [45, 44, 43], [45, 38, 44],
     [38, 31, 44], [44, 41, 43], [44, 37, 41], [37, 29, 41], [41, 42, 43],
     [41, 36, 42], [36, 27, 42], [26, 40, 27], [26, 51, 40], [51, 35, 40],
     [34, 39, 35], [34, 50, 39], [50, 33, 39], [32, 38, 33], [32, 49, 38],
     [49, 31, 38], [30, 37, 31], [30, 48, 37], [48, 29, 37], [28, 36, 29],
     [28, 47, 36], [47, 27, 36], [51, 34, 35], [50, 32, 33], [49, 30, 31],
     [48, 28, 29], [47, 26, 27]])

elev = np.asarray(
    [0., 0.5535673, 1.01721871, 0., -1.01721871, -0.5535673, 0.52359324,
     0.31415301, 0.94246863, 0., -0.31415301, 0., 0.52359547, -0.52359324,
     -0.52359547, -0.94246863, 0.31415501, -0.31415501, 1.57079633, 0.94247719,
     0.31415501, 0.94247719, -0.94247719, -0.31415501, -0.94247719,
     -1.57079633, -0.31415624, 0., 0.94248124, 1.01722122, 0.94247396,
     0.55356579, -0.31415377, -0.55356579, -1.57079233, -1.01722122,
     0.52359706, 0.94246791, 0., -0.94246791, -0.52359706, 0.52359371, 0., 0.,
     0.31415246, -0.31415246, -0.52359371, 0.31415624, 1.57079233, 0.31415377,
     -0.94247396, -0.94248124])

az = np.asarray(
    [-1.57079633, -0.55358064, -2.12435586, -2.67794236, -2.12435586,
     -0.55358064, -1.7595018, -1.10715248, -1.10714872, -0.55357999,
     -1.10715248, -2.12436911, -2.48922865, -1.7595018, -2.48922865,
     -1.10714872, 0., 0., 0., 0., 3.14159265, 3.14159265, 3.14159265,
     3.14159265, 0., 0., 0., 0.46365119, 0., 1.01724226, 3.14159265,
     2.58801549, 3.14159265, 2.58801549, 3.14159265, 1.01724226, 0.6523668,
     2.03445078, 2.58801476, 2.03445078, 0.6523668, 1.38209652, 1.01722642,
     1.57080033, 2.03444394, 2.03444394, 1.38209652, 0., 3.14159265,
     3.14159265, 3.14159265, 0.])

# vertices for the capsules
v = np.vstack(
    [np.cos(az) * np.cos(elev), np.sin(az) * np.cos(elev), np.sin(elev)]).T


class Capsule(object):

    def __init__(self, t, rod, rad, length):
        assert (hasattr(t, 'dterms'))
        # the translation should be a chumpy object (differentiable wrt shape)
        self.t = t  # translation of the axis
        self.rod = rod  # rotation of the axis in Rodrigues form
        # the radius should be a chumpy object (differentiable wrt shape)
        assert (hasattr(rad, 'dterms'))
        self.rad = rad  # radius of the capsule
        # the length should be a chumpy object (differentiable wrt shape)
        assert (hasattr(length, 'dterms'))
        self.length = length  # length of the axis
        axis0 = ch.vstack([0, ch.abs(self.length), 0])
        self.axis = ch.vstack((t.T, (t + Rodrigues(rod).dot(axis0)).T))
        v0 = ch.hstack([v[:26].T * rad, (v[26:].T * rad) + axis0])
        self.v = ((t + Rodrigues(rod).dot(v0)).T)
        self.set_sphere_centers()

    def set_sphere_centers(self, floor=False):
        # sphere centers are evenly spaced along the capsule axis length
        if floor:
            n_spheres = int(np.floor(self.length / (2 * self.rad) - 1))
        else:
            n_spheres = int(np.ceil(self.length / (2 * self.rad) - 1))

        centers = [self.axis[0].r, self.axis[1].r]

        if n_spheres >= 1:
            step = self.length.r / (n_spheres + 1)
            for i in xrange(n_spheres):
                centers.append(self.axis[0].r + (self.axis[1].r - self.axis[
                    0].r) * step * (i + 1) / self.length.r)

        self.centers = centers
