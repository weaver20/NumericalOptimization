import random

import numpy as np
import matplotlib.pyplot as plt


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

def numerical_diff_gradient(func, vector: np.matrix, epsilon):
    vec_len = vector.shape[0]
    assert vector.shape[1] == 1
    assert vec_len > 0
    gradient = np.matrix(np.zeros((vec_len, 1)), dtype=np.float128)
    for i in range(vec_len):
        base_vector = np.matrix(np.zeros((vec_len, 1)), dtype=np.float128)
        base_vector[i, 0] = 1
        func_plus = func(vector + (epsilon * base_vector))
        func_minus = func(vector - (epsilon * base_vector))
        gradient[i, 0] = ((func_plus - func_minus) / (2 * epsilon))
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


# Section 1.3 - Comparison plot
def compare_grad():
    vec = np.matrix(np.random.randint(0, 9, size=(3, 1)))
    epsilon = []
    values = []
    for i in range(60):
        epsilon.append(np.random.uniform(2e-60, 1))

    for i in range(60):
        grad = gradient_phi(np.matrix(np.identity(3)), vec)
        num_grad = numerical_diff_gradient(func_phi, vec, epsilon[i])
        x = gradient_phi(np.matrix(np.identity(3)), vec) - numerical_diff_gradient(func_phi, vec, epsilon[i])
        y = max(x)
        values.append(max(gradient_phi(np.matrix(np.identity(3)), vec)
                          - numerical_diff_gradient(func_phi, vec, epsilon[i])))
    plt.plot(epsilon, values)
    plt.xscale("exponent")
    plt.yscale("logarithmic")
    plt.show()


def func_phi(vec: np.matrix):
    return np.sin(np.product(vec))


compare_grad()
