"""
Copyright 2016 Max Planck Society, Federica Bogo, Angjoo Kanazawa. All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPLify license here:
     http://smplify.is.tue.mpg.de/license

This script implements the pose prior based on a mixture of Gaussians.
To simplify the log-likelihood computation, the sum in the mixture of Gaussians
is approximated by a max operator (see the paper for more details).
"""

import os
import numpy as np
import chumpy as ch


class MaxMixtureComplete(ch.Ch):
    """Define the MaxMixture class."""
    # x is the input vector we want to evaluate the prior on;
    # means, precs and weights are the parameters of the mixture
    dterms = 'x'
    terms = 'means', 'precs', 'weights'

    def on_changed(self, which):
        # on_changed is called before any call to r or dr_wrt,
        # therefore it can be used also for initialization
        # setup means, precs and loglikelihood expressions
        if 'means' in which or 'precs' in which or 'weights' in which:
            # This is just the mahalanobis part.
            self.loglikelihoods = [np.sqrt(0.5) * (self.x - m).dot(s)
                                   for m, s in zip(self.means, self.precs)]

        if 'x' in which:
            self.min_component_idx = np.argmin(
                [(logl**2).sum().r[0] - np.log(w[0])
                 for logl, w in zip(self.loglikelihoods, self.weights)])

    def compute_r(self):
        min_w = self.weights[self.min_component_idx]
        # Add the sqrt(-log(weights))
        return ch.concatenate((self.loglikelihoods[self.min_component_idx].r,
                               np.sqrt(-np.log(min_w))))

    def compute_dr_wrt(self, wrt):
        # the call to dr_wrt returns a jacobian 69 x 72,
        # when wrt has 72 elements (pose vector)
        # here we intercept the call and return a 70 x 72 matrix,
        # with an additional row of zeroes (these are the jacobian
        # entries corresponding to sqrt(-log(weights))
        import scipy.sparse as sp

        dr = self.loglikelihoods[self.min_component_idx].dr_wrt(wrt)
        if dr is not None:
            # extract rows, cols and data, and return a new matrix with
            # the same values but 1 additional row
            Is, Js, Vs = sp.find(dr)
            dr = sp.csc_matrix(
                (Vs, (Is, Js)), shape=(dr.shape[0] + 1, dr.shape[1]))

        return dr


class MaxMixtureCompleteWrapper(object):
    """Convenience wrapper to match interface spec."""

    def __init__(self, means, precs, weights, prefix):
        self.means = means
        self.precs = precs  # Already "sqrt"ed
        self.weights = weights
        self.prefix = prefix

    def __call__(self, x):
        # wrapping since __call__ couldn't be defined directly for a chumpy
        # object
        return (MaxMixtureComplete(
            x=x[self.prefix:],
            means=self.means,
            precs=self.precs,
            weights=self.weights))


class MaxMixtureCompletePrior(object):
    """Prior density estimation."""

    def __init__(self, n_gaussians=8, prefix=3):
        self.n_gaussians = n_gaussians
        self.prefix = prefix
        self.prior = self.create_prior_from_cmu()

    def create_prior_from_cmu(self):
        """Load the gmm from the CMU motion database."""
        from os.path import dirname
        import cPickle as pickle
        with open(
                os.path.join(
                    dirname(dirname(__file__)), 'models', 'gmm_%02d.pkl' %
                    self.n_gaussians)) as f:
            gmm = pickle.load(f)

        precs = ch.asarray([np.linalg.inv(cov) for cov in gmm['covars']])
        chols = ch.asarray([np.linalg.cholesky(prec) for prec in precs])

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                            for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)

        self.weights = ch.asarray(gmm['weights'] / (const *
                                                    (sqrdets / sqrdets.min())))

        return (MaxMixtureCompleteWrapper(
            means=gmm['means'],
            precs=chols,
            weights=self.weights,
            prefix=self.prefix))

    def get_gmm_prior(self):
        """Getter implementation."""
        return self.prior
