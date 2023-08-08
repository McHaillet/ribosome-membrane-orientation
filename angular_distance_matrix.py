import numpy as np
from numba import njit, prange

epsilon = 0.0000005


# ===================================NUMBA ACCELERATED ANGULAR DISTANCE MATRIX==========================================
@njit
def check_eps(value):
    if abs(value) < epsilon:
        return 0.0
    else:
        return float(value)


@njit
def fill_zmat(z):
    mat = np.identity(3)
    sinz = check_eps(np.sin(z))
    cosz = check_eps(np.cos(z))
    mat[0, 0] = cosz
    mat[0, 1] = -sinz
    mat[1, 0] = sinz
    mat[1, 1] = cosz
    return mat


@njit
def fill_xmat(x):
    mat = np.identity(3)
    sinx = check_eps(np.sin(x))
    cosx = check_eps(np.cos(x))
    mat[1, 1] = cosx
    mat[1, 2] = -sinx
    mat[2, 1] = sinx
    mat[2, 2] = cosx
    return mat


@njit
def diff(z1_1, x_1, z2_1, z1_2, x_2, z2_2):
    """
    Input and output of angular diff in radians.
    """
    z1mat_1 = fill_zmat(z1_1)
    xmat_1 = fill_xmat(x_1)
    z2mat_1 = fill_zmat(z2_1)
    rot_1 = np.dot(np.dot(z1mat_1, xmat_1), z2mat_1)

    z1mat_2 = fill_zmat(z1_2)
    xmat_2 = fill_xmat(x_2)
    z2mat_2 = fill_zmat(z2_2)
    rot_2 = np.linalg.inv(np.dot(np.dot(z1mat_2, xmat_2), z2mat_2))  # invert for diff angle

    rot = np.dot(rot_1, rot_2)

    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    cos_ang = .5 * (trace - 1)
    if cos_ang > 1:
        cos_ang = 1.
    if cos_ang < -1.:
        cos_ang = -1.
    the = np.arccos(cos_ang)

    return the


@njit(parallel=True)
def angular_distance(mat):
    newmat = np.zeros((mat.shape[0], mat.shape[1]), dtype=np.float32)  # output matrix shape

    for i in prange(mat.shape[0]):
        for j in range(mat.shape[1]):
            newmat[i, j] = diff(mat[i, j, 0], mat[i, j, 1], mat[i, j, 2],
                                mat[i, j, 3], mat[i, j, 4], mat[i, j, 5])

    for i in prange(mat.shape[0]):
        newmat[i, i] = .0

    return newmat


def angular_distance_matrix(rotations):
    """
    Calculate a angular distance matrix with the 'distance' bewteen all the rotations in the input.
    """
    # construct matrix of (n, n, 6) where is the number zxz of rotations in the input
    ang1 = np.tile(rotations[:, np.newaxis, :], (1, rotations.shape[0], 1))
    ang2 = np.tile(rotations[np.newaxis, :, :], (rotations.shape[0], 1, 1))
    mat = np.deg2rad(np.concatenate((ang1, ang2), axis=2))  # combine to matrix shape

    # pass to calculation of angular distance which returns a (n,n,1) matrix
    return np.rad2deg(angular_distance(mat)) / 180.
