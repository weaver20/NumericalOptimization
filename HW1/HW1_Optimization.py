import numpy as np


# Section 1.1.5 - Analytical Evaluation

# SubSection 1
def gradient_phi(mat: np.matrix, vec: np.matrix):
    grad_vec = np.matrix(np.array([[vec[1] * vec[2]], [vec[0] * vec[2]], [vec[0] * vec[1]]]))
    return np.cos(np.product(vec)) * grad_vec


def hessian_phi(mat: np.matrix, vec: np.matrix):
    hessian_mat = np.matrix([[0, vec[2], vec[1]],
                             [vec[2], 0, vec[0]],
                             [vec[1], vec[0], 0]])
    return -1 * np.sin(np.product(vec)) * hessian_mat


# SubSection 2
def gradient_h(vec: np.matrix):
    return np.cos(np.sin(np.product(vec))) * gradient_phi(np.matrix(np.identity(3)), vec)


def hessian_h(vec: np.matrix):
    # Calculating the 2nd derivative of h of phi
    derivative = (np.cos(np.product(vec)) ** 2 -
                 (1 + np.sin(np.product(vec)) ** 2) * np.sin(np.product(vec)) ** 2) / \
                (1 + np.sin(np.product(vec)) ** 2) ** 1.5

    return derivative * hessian_phi(np.matrix(np.identity(3)), vec)


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
