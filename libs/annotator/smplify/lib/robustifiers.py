"""
Copyright 2016 Max Planck Society, Matthew Loper. All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPLify license here:
     http://smplify.is.tue.mpg.de/license

This script implements the Geman-McClure robustifier as chumpy object.
"""

#!/usr/bin/env python
import numpy as np
import scipy
import scipy.sparse as sp
from chumpy import Ch

__all__ = ['GMOf']


def GMOf(x, sigma):
    """Given x and sigma in some units (say mm),
    returns robustified values (in same units),
    by making use of the Geman-McClure robustifier."""

    result = SignedSqrt(x=GMOfInternal(x=x, sigma=sigma))
    return result


class SignedSqrt(Ch):
    dterms = ('x', )
    terms = ()

    def compute_r(self):
        return np.sqrt(np.abs(self.x.r)) * np.sign(self.x.r)

    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            result = (.5 / np.sqrt(np.abs(self.x.r)))
            result = np.nan_to_num(result)
            result *= (self.x.r != 0).astype(np.uint32)
            return sp.spdiags(result.ravel(), [0], self.x.r.size,
                              self.x.r.size)


class GMOfInternal(Ch):
    dterms = 'x', 'sigma'

    def on_changed(self, which):
        if 'sigma' in which:
            assert (self.sigma.r > 0)

        if 'x' in which:
            self.squared_input = self.x.r**2.

    def compute_r(self):
        return (self.sigma.r**2 *
                (self.squared_input /
                 (self.sigma.r**2 + self.squared_input))) * np.sign(self.x.r)

    def compute_dr_wrt(self, wrt):
        if wrt is not self.x and wrt is not self.sigma:
            return None

        squared_input = self.squared_input
        result = []
        if wrt is self.x:
            dx = self.sigma.r**2 / (self.sigma.r**2 + squared_input
                                    ) - self.sigma.r**2 * (squared_input / (
                                        self.sigma.r**2 + squared_input)**2)
            dx = 2 * self.x.r * dx
            result.append(
                scipy.sparse.spdiags(
                    (dx * np.sign(self.x.r)).ravel(), [0],
                    self.x.r.size,
                    self.x.r.size,
                    format='csc'))
        if wrt is self.sigma:
            ds = 2 * self.sigma.r * (squared_input / (
                self.sigma.r**2 + squared_input)) - 2 * self.sigma.r**3 * (
                    squared_input / (self.sigma.r**2 + squared_input)**2)
            result.append(
                scipy.sparse.spdiags(
                    (ds * np.sign(self.x.r)).ravel(), [0],
                    self.x.r.size,
                    self.x.r.size,
                    format='csc'))

        if len(result) == 1:
            return result[0]
        else:
            return np.sum(result).tocsc()
