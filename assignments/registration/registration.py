# Copyright (c) 2025 University of Bern. All rights reserved.
import numpy as np
import cas.registration.util as util

def paired_point_matching(source, target):
    """
    Calculates the transformation T that maps the source to the target point clouds.
    :param source: A N x 3 matrix with N 3D points.
    :param target: A N x 3 matrix with N 3D points.
    :return:
        T: 4x4 transformation matrix mapping source to target.
        R: 3x3 rotation matrix part of T.
        t: 1x3 translation vector part of T.
    """
    assert source.shape == target.shape
    T = np.eye(4)
    R = np.eye(3)
    t = np.zeros((1, 3))

    ## TODO: your code goes here

    # Target = Reference = r, source = floating = l

    # centroid
    ul = np.mean(source, axis=0)
    ur = np.mean(target, axis=0)

    plCen = source - ul
    prCen = target - ur

    # covariance matrix
    M = plCen.T @ prCen

    # SVD / rotation
    U, W, Vt = np.linalg.svd(M)
    R = Vt.T @ U.T

    # translation
    t = ur - R@ul

    # Transformation matrix
    T[:3,:3] = R
    T[:3,3] = t

    return T, R, t


def get_initial_pose(source, target):
    """
    Calculates an initial rough registration or optionally returns a hand-picked initial pose.
    :param source: A N x 3 point cloud.
    :param target: A N x 3 point cloud.
    :return: An initial 4 x 4 rigid transformation matrix.
    """
    T = np.eye(4)
    R = np.eye(3)

    ## TODO: Your code goes here

    t = np.array([600, 0, 0])

    R = np.array([[1, 0, 0],
                 [0, 0, -1],
                 [0, 1, 0]])

    #T = np.eye(4)
    #R = np.eye(3)

    T[:3,:3] = R
    T[:3,3] = t

    return T


def find_nearest_neighbor(source, target):
    """
    Finds the nearest neighbor in 'target' for every point in 'source'.
    :param source: A N x 3 point cloud.
    :param target: A N x 3 point cloud.
    :return: A tuple containing two arrays: the first array contains the
             distances to the nearest neighbor in 'target' for each point
             in 'source', and the second array contains the indices of
             these nearest neighbors in 'target'.
    """

    ## TODO: replace this by your code
    distances = np.zeros(source.shape[0])
    indices = np.zeros(source.shape[0], dtype=int)

    for i, point in enumerate(source):
        diff = target - point # difference x,y,z of each target point to actual point
        dists = np.linalg.norm(diff, axis=1) # calc distance of each row with norm
        indices[i] = np.argmin(dists) # stores index of smallest distance
        distances[i] = dists[indices[i]] # stores the minimal distance

    return distances, indices


def icp(source, target, init_pose=None, max_iterations=10, tolerance=0.0001):
    """
    Iteratively finds the best transformation mapping the source points onto the target.
    :param source: A N x 3 point cloud.
    :param target: A N x 3 point cloud.
    :param init_pose: Initial pose as a 4 x 4 transformation matrix.
    :param max_iterations: Maximum iterations.
    :param tolerance: Error tolerance.
    :return: The optimal 4 x 4 rigid transformation matrix, distances, and registration error.
    """

    # Initialisation
    T = np.eye(4)
    distances = 0
    error = np.finfo(float).max

    ## TODO: Your code goes here
    prev_error = np.inf
    T = init_pose

    src = util.make_homogenous(source)
    srcTransf = np.dot(T, src.T).T[:, :3]

    for i in range(max_iterations):

        # find nearest neighbors
        distances, indices = find_nearest_neighbor(srcTransf, target)
        matched_target = target[indices]

        # calculate new transformation
        delta_T, _, _ = paired_point_matching(srcTransf, matched_target)

        T = delta_T @ T

        src = util.make_homogenous(source)
        srcTransf = np.dot(T, src.T).T[:, :3]


        # compute RMSE
        rmse = np.sqrt(np.mean(distances**2))

        if np.abs(prev_error - rmse) < tolerance:
            break

        prev_error = rmse

    error = rmse

    return T, distances, error
