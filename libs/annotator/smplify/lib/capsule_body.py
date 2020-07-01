"""
Copyright 2016 Max Planck Society, Federica Bogo, Angjoo Kanazawa. All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPLify license here:
     http://smplify.is.tue.mpg.de/license

This script implements an approximation of the body by means of capsules (20 in total).
Capsules can be further simplified into spheres (with centers along the capsule axis and
radius corresponding to the capsule radius) to efficiently compute an interpenetration error term
(as in sphere_collisions.py).
"""

import numpy as np
import chumpy as ch
import scipy.sparse as sp

from .capsule_ch import Capsule

joint2name = ['pelvis', 'leftThigh', 'rightThigh', 'spine', 'leftCalf',
              'rightCalf', 'spine1', 'leftFoot', 'rightFoot', 'spine2', 'neck',
              'leftShoulder', 'rightShoulder', 'head', 'leftUpperArm',
              'rightUpperArm', 'leftForeArm', 'rightForeArm', 'leftHand',
              'rightHand']

# the orientation of each capsule
rots0 = ch.asarray(
    [[0, 0, np.pi / 2], [0, 0, np.pi], [0, 0, np.pi], [0, 0, np.pi / 2],
     [0, 0, np.pi], [0, 0, np.pi], [0, 0, np.pi / 2], [np.pi / 2, 0, 0],
     [np.pi / 2, 0, 0], [0, 0, np.pi / 2], [0, 0, 0], [0, 0, -np.pi / 2],
     [0, 0, np.pi / 2], [0, 0, 0], [0, 0, -np.pi / 2], [0, 0, np.pi / 2],
     [0, 0, -np.pi / 2], [0, 0, np.pi / 2], [0, 0, -np.pi / 2],
     [0, 0, np.pi / 2]])

# groups hands and fingers, feet and toes
# each comment line provides the body part corresonding to the capsule
# and the corresponding id
mujoco2segm = [[0],  # hip 0
               [1],  # leftThigh 1
               [2],  # rightThigh 2
               [3],  # spine 3
               [4],  # leftCalf 4
               [5],  # rightCalf 5
               [6],  # spine1 6
               [7, 10],  # leftFoot + leftToes 7
               [8, 11],  # rightFoot + rightToes 8
               [9],  # spine2 9
               [12],  # neck 10
               [13],  # leftShoulder 11
               [14],  # rightShoulder 12
               [15],  # head 13
               [16],  # leftUpperArm 14
               [17],  # rightUpperArm 15
               [18],  # leftForeArm 16
               [19],  # rightForeArm 17
               [20, 22],  # leftHand + leftFingers 18
               [21, 23]]  # rightHand + rightFingers 19

# sets pairs of ids, corresponding to capsules that should not
# penetrate each other
collisions = [
    [0, 16],  # hip and leftForeArm
    [0, 17],  # hip and rightForeArm
    [0, 18],  # hip and leftHand
    [0, 19],  # hip and rightHand
    [3, 16],  # spine and leftForeArm
    [3, 17],  # spine and rightForeArm
    [3, 18],  # spine and leftHand
    [3, 19],  # spine and rightHand
    [4, 5],  # leftCalf and rightCalf
    [6, 16],  # spine1 and leftForeArm
    [6, 17],  # spine1 and rightForeArm
    [6, 18],  # spine1 and leftHand
    [6, 19],  # spine1 and rightHand
    [7, 5],  # leftFoot and rightCalf
    [8, 7],  # rightFoot and leftFoot
    [8, 4],  # rightFoot and leftCalf
    [9, 16],  # spine2 and leftForeArm
    [9, 17],  # spine2 and rightForeArm
    [9, 18],  # spine2 and leftHand
    [9, 19],  # spine2 and rightHand
    [11, 16],  # leftShoulder and leftForeArm
    [12, 17],  # rightShoulder and rightForeArm
    [18, 19],  # leftHand and rightHand
]


def get_capsules(model, wrt_betas=None, length_regs=None, rad_regs=None):
    from opendr.geometry import Rodrigues
    if length_regs is not None:
        n_shape_dofs = length_regs.shape[0] - 1
    else:
        n_shape_dofs = model.betas.r.size
    segm = np.argmax(model.weights_prior, axis=1)
    J_off = ch.zeros((len(joint2name), 3))
    rots = rots0.copy()
    mujoco_t_mid = [0, 3, 6, 9]
    if wrt_betas is not None:
        # if we want to differentiate wrt betas (shape), we must have the
        # regressors...
        assert (length_regs is not None and rad_regs is not None)
        # ... and betas must be a chumpy object
        assert (hasattr(wrt_betas, 'dterms'))
        pad = ch.concatenate(
            (wrt_betas, ch.zeros(n_shape_dofs - len(wrt_betas)), ch.ones(1)))
        lengths = pad.dot(length_regs)
        rads = pad.dot(rad_regs)
    else:
        lengths = ch.ones(len(joint2name))
        rads = ch.ones(len(joint2name))
    betas = wrt_betas if wrt_betas is not None else model.betas
    n_betas = len(betas)
    # the joint regressors are the original, pre-optimized ones
    # (middle of the part frontier)
    myJ_regressor = model.J_regressor_prior
    myJ0 = ch.vstack(
        (ch.ch.MatVecMult(myJ_regressor, model.v_template[:, 0] +
                          model.shapedirs[:, :, :n_betas].dot(betas)[:, 0]),
         ch.ch.MatVecMult(myJ_regressor, model.v_template[:, 1] +
                          model.shapedirs[:, :, :n_betas].dot(betas)[:, 1]),
         ch.ch.MatVecMult(myJ_regressor, model.v_template[:, 2] +
                          model.shapedirs[:, :, :n_betas].dot(betas)[:, 2]))).T
    # with small adjustments for hips, spine and feet
    myJ = ch.vstack(
        [ch.concatenate([myJ0[0, 0], (
            .6 * myJ0[0, 1] + .2 * myJ0[1, 1] + .2 * myJ0[2, 1]), myJ0[9, 2]]),
         ch.vstack([myJ0[i] for i in range(1, 7)]), ch.concatenate(
             [myJ0[7, 0], (1.1 * myJ0[7, 1] - .1 * myJ0[4, 1]), myJ0[7, 2]]),
         ch.concatenate(
             [myJ0[8, 0], (1.1 * myJ0[8, 1] - .1 * myJ0[5, 1]), myJ0[8, 2]]),
         ch.concatenate(
             [myJ0[9, 0], myJ0[9, 1], (.2 * myJ0[9, 2] + .8 * myJ0[12, 2])]),
         ch.vstack([myJ0[i] for i in range(10, 24)])])
    capsules = []
    # create one capsule per mujoco joint
    for ijoint, segms in enumerate(mujoco2segm):
        if wrt_betas is None:
            vidxs = np.asarray([segm == k for k in segms]).any(axis=0)
            verts = model.v_template[vidxs].r
            dims = (verts.max(axis=0) - verts.min(axis=0))
            rads[ijoint] = .5 * ((dims[(np.argmax(dims) + 1) % 3] + dims[(
                np.argmax(dims) + 2) % 3]) / 4.)
            lengths[ijoint] = max(dims) - 2. * rads[ijoint].r
        # the core joints are different, since the capsule is not in the joint
        # but in the middle
        if ijoint in mujoco_t_mid:
            len_offset = ch.vstack([ch.zeros(1), ch.abs(lengths[ijoint]) / 2.,
                                    ch.zeros(1)]).reshape(3, 1)
            caps = Capsule(
                (J_off[ijoint] + myJ[mujoco2segm[ijoint][0]]).reshape(
                    3, 1) - Rodrigues(rots[ijoint]).dot(len_offset),
                rots[ijoint], rads[ijoint], lengths[ijoint])
        else:
            caps = Capsule(
                (J_off[ijoint] + myJ[mujoco2segm[ijoint][0]]).reshape(3, 1),
                rots[ijoint], rads[ijoint], lengths[ijoint])
        caps.id = ijoint
        capsules.append(caps)
    return capsules


def set_sphere_centers(capsule, floor=True):
    if floor:
        n_spheres = int(np.floor(capsule.length.r / (2 * capsule.rad.r) - 1))
    else:
        n_spheres = int(np.ceil(capsule.length.r / (2 * capsule.rad.r) - 1))

    # remove "redundant" spheres for right and left thigh...
    if capsule.id == 1 or capsule.id == 2:
        centers = [capsule.axis[1].r]
    # ... and right and left upper arm
    elif capsule.id == 14 or capsule.id == 15:
        if n_spheres >= 1:
            centers = []
        else:
            centers = [capsule.axis[1].r]
    else:
        centers = [capsule.axis[0].r, capsule.axis[1].r]

    if n_spheres >= 1:
        step = capsule.length.r / (n_spheres + 1)
        for i in xrange(n_spheres):
            centers.append(capsule.axis[0].r + (capsule.axis[
                1].r - capsule.axis[0].r) * step * (i + 1) / capsule.length.r)

    capsule.centers = centers
    return capsule.centers


def capsule_dist(capsule0, capsule1, alpha=.3, increase_hand=True):
    range0 = range(capsule0.center_id,
                   capsule0.center_id + len(capsule0.centers))
    range1 = range(capsule1.center_id,
                   capsule1.center_id + len(capsule1.centers))
    cnt0 = ch.concatenate([[cid] * len(range1) for cid in range0])
    cnt1 = ch.concatenate([range1] * len(range0))
    if increase_hand:
        if (capsule0.id == 18) or (capsule0.id == 19) or (
                capsule1.id == 18) or (capsule1.id == 19):
            dst = (alpha * 1.2 * capsule0.rad.r)**2 + (alpha * 1.2 *
                                                       capsule1.rad.r)**2
        else:
            dst = (alpha * capsule0.rad.r)**2 + (alpha * capsule1.rad.r)**2
    else:
        dst = (alpha * capsule0.rad.r)**2 + (alpha * capsule1.rad.r)**2
    radiuss = np.hstack([dst] * len(cnt0)).squeeze()
    return (cnt0, cnt1, radiuss)


def get_capsule_bweights(vs):
    # "blend" weights for the capsule. They are binary
    rows = np.arange(vs.shape[0])
    cols = np.tile(np.hstack((range(10), range(12, 22))), (52, 1)).T.ravel()
    data = np.ones(vs.shape[0])
    caps_weights = np.asarray(
        sp.csc_matrix(
            (data, (rows, cols)), shape=(vs.shape[0], 24)).todense())
    return caps_weights


def get_sphere_bweights(sph_vs, capsules):
    rows = np.arange(sph_vs.shape[0])
    cols = []
    for cps, w in zip(capsules, range(10) + range(12, 22)):
        cols.append([w] * len(cps.centers))
    cols = np.hstack(cols)
    data = np.ones(sph_vs.shape[0])
    sph_weights = np.asarray(
        sp.csc_matrix(
            (data, (rows, cols)), shape=(sph_vs.shape[0], 24)).todense())
    return sph_weights
