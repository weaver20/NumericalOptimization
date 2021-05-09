import random

import numpy as np
import matplotlib.pyplot as plt

phi = lambda vec: np.sin(np.product(vec))
h = lambda x: np.sqrt(1 + np.sin(np.product(x)) ** 2)



# Section 1.1.5 - Analytical Evaluation

# SubSection 1
def gradient_phi(mat: np.matrix, vec: np.matrix):
    grad_vec = np.matrix(np.array([[vec[1] * vec[2]], [vec[0] * vec[2]], [vec[0] * vec[1]]]), dtype=np.float128)
    grad_vec = grad_vec.transpose()
    return np.cos(np.product(vec)) * grad_vec


def hessian_phi(mat: np.matrix, vec: np.matrix):
    nabla = np.zeros((3, 3))
    nabla[0, 0] = (-((vec[1] * vec[2]) ** 2) * np.sin(np.product(vec)))
    nabla[0, 1] = vec[2] * np.cos(np.product(vec)) - np.product(vec) * vec[2] * np.sin(np.product(vec))
    nabla[0, 2] = vec[1] * np.cos(np.product(vec)) - np.product(vec) * vec[1] * np.sin(np.product(vec))
    nabla[1, 0] = vec[2] * np.cos(np.product(vec)) - np.product(vec) * vec[2] * np.sin(np.product(vec))
    nabla[1, 1] = (-((vec[0] * vec[2]) ** 2) * np.sin(np.product(vec)))
    nabla[1, 2] = vec[0] * np.cos(np.product(vec)) - np.product(vec) * vec[0] * np.sin(np.product(vec))
    nabla[2, 0] = vec[1] * np.cos(np.product(vec)) - np.product(vec) * vec[1] * np.sin(np.product(vec))
    nabla[2, 1] = vec[0] * np.cos(np.product(vec)) - np.product(vec) * vec[0] * np.sin(np.product(vec))
    nabla[2, 2] = (-((vec[0] * vec[1]) ** 2) * np.sin(np.product(vec)))
    return mat.transpose() * nabla * mat


# SubSection 2
def gradient_h(vec: np.matrix):
    return 0.5 * np.sin(np.product(vec)) / ((1 + (np.sin(np.product(vec)) ** 2)) ** 0.5) \
           * gradient_phi(np.matrix(np.identity(3)), vec)


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


def numerical_diff_hessian(func, vec: np.matrix, epsilon):
    vec_len = vec.shape[0]
    assert vec.shape[1] == 1
    assert vec_len > 0
    hessian = np.matrix(np.zeros((vec_len, vec_len)), dtype=np.float128)
    for i in range(vec_len):
        base_vector = np.matrix(np.zeros((vec_len, 1)))
        base_vector[i, 0] = 1
        v1 = numerical_diff_gradient(func, vec + epsilon * base_vector, epsilon)
        v2 = numerical_diff_gradient(func, vec - epsilon * base_vector, epsilon)
        value = (v1 - v2)
        hessian[0:vec_len, i] = value / (2 * epsilon)
    return hessian


# Section 1.3 - Comparison plot
def compare_grad():
    vec = np.matrix(np.random.rand(3), dtype=np.float128).transpose()
    A = np.matrix(np.random.rand(3, 3), dtype=np.float128)
    epsilon = []
    values = []
    for i in range(61):
        epsilon.append(2 ** -i)

    # Comparison for f1 gradient
    analytical_grad = gradient_phi(A, vec)
    for i in range(61):
        numerical_grad = numerical_diff_gradient(phi, vec, epsilon[i])
        x = np.abs(analytical_grad - numerical_grad)
        values.append(np.linalg.norm(x, np.inf))
    show_plot("f1 gradient", values)

    # Comparison for f1 hessian
    analytical_hessian = hessian_phi(A, vec)
    values = []
    for i in range(61):
        numerical_hessian = numerical_diff_hessian(phi, A * vec, epsilon[i])
        x = np.abs(analytical_hessian - numerical_hessian)
        values.append(np.linalg.norm(x, np.inf))
    show_plot("f1 hessian", values)

    # Comparison for f2
    analytical_grad = gradient_h(vec)
    values = []
    for i in range(61):
        numerical_grad = numerical_diff_gradient(h, vec, epsilon[i])
        x = np.abs(analytical_grad - numerical_grad)
        y = max(x).item()
        values.append(y)
    show_plot("f2 gradient", values)


def show_plot(str, y_axis):
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.subplots_adjust(top=0.85)
    fig.suptitle(str, fontsize=14, fontweight='bold')
    ax.set_xlabel('epsilon')
    ax.set_ylabel('differentiation')
    plt.plot(list(range(61)), y_axis)
    plt.xscale("linear")
    plt.yscale("log")
    plt.show()


compare_grad()
