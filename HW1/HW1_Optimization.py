import numpy as np
import torch
import numdifftools as nd


def compute(x: np.array, A: np.array):
    assert x.shape == (3, 1)
    assert A.shape == (3, 3)
    Ax = np.matmul(A, x)
    value = np.sin(np.product(Ax))
    cosAx = np.cos(np.product(Ax))
    phi_gradiant = cosAx * np.array([Ax[1, 0] * Ax[2, 0], Ax[0, 0] * Ax[2, 0], Ax[0, 0] * Ax[1, 0]])
    gradiant = np.matmul(A.transpose(), phi_gradiant)
    hessian_diag = np.array([[-np.sin(np.product(Ax)) * (Ax[1, 0] * pow(Ax[2, 0],2)), 0, 0], \
                             [0, -np.sin(np.product(Ax)) * (Ax[1, 0] * pow(Ax[2, 0],2)) , 0], \
                             [0, 0, -np.sin(np.product(Ax)) * (Ax[1, 0])* pow(Ax[2, 0],2)]])
    hessian_not_diag = np.array(
        [[0, -np.sin(np.product(Ax)) * (Ax[1, 0] * Ax[0, 0] * pow(Ax[2, 0],2)) + cosAx * Ax[2, 0], \
          -np.sin(np.product(Ax)) * ((pow(Ax[1, 0],2)) * Ax[0, 0] * Ax[2, 0]) + cosAx * Ax[1, 0]], \
         [0, 0, -np.sin(np.product(Ax)) * (Ax[1, 0] * pow(Ax[0, 0], 2) * Ax[2, 0]) + cosAx * Ax[0, 0]], \
         [0, 0, 0]])
    # note that the hessian is symmetric so I only had to compute the diagonal and the upper triangle
    return value, gradiant, hessian_diag + hessian_not_diag + hessian_not_diag.transpose()


print(compute(np.array([[2], [1], [1]]), np.array([[2, 0, 0], [0, 2, 0], [0, 0, 3]])))
print("end")


# Section 1.2.1 - Numerical Differentiation

def numerical_diff_gradient(func, vec: np.matrix, scalar):
    vec_len = vec.shape[0]
    assert vec.shape[1] == 1
    assert vec_len > 0
    gradient = np.matrix(np.zeros((vec_len, 1)))
    for i in range(vec_len):
        base_vector = np.matrix(np.zeros((vec_len, 1)))
        base_vector[i, 0] = 1
        gradient[i, 0] = (func(vec + scalar * base_vector) - func(vec - scalar * base_vector)) / (2 * scalar)
    return gradient


def numerical_diff_hessian(func, vec: np.matrix, scalar):
    vec_len = vec.shape[0]
    assert vec.shape[1] == 1
    assert vec_len > 0
    hessian = np.matrix(np.zeros(vec_len, vec_len))
    for i in range(vec_len):
        base_vector = np.matrix(np.zeros((vec_len, 1)))
        base_vector[i, 0] = 1
        hessian[: i] = (numerical_diff_gradient(func, vec + scalar * base_vector, scalar)
                        - numerical_diff_gradient(func, vec - scalar * base_vector, scalar)) / (2 * scalar)
    return hessian