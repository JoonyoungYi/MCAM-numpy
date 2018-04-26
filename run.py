import random
import math

import numpy as np
from scipy.linalg import orth


def _make_lr_matrix(m, n, k):
    L = np.random.randn(m, k)
    R = np.random.randn(k, n)
    return np.dot(L, R)


def _get_masked_matrix(M, omega):
    M_max, M_min = np.max(M), np.min(M)
    M_ = M.copy()
    M_[(1 - omega).astype(np.int16)] = M_max * M_max
    return M_


def _get_V_from_U(M, U, omega):
    column = M.shape[1]
    rank = U.shape[1]
    V = np.empty((rank, column), dtype=M.dtype)

    for j in range(0, column):
        U_ = U.copy()
        U_[(1 - omega[:, j]).astype(np.int16), :] = 0
        V[:, j] = np.linalg.lstsq(U_, M[:, j], rcond=None)[0]
    return V


def _get_err(M, U, V, omega):
    error_matrix = M - np.dot(U, V)
    error_matrix[(1 - omega).astype(np.int16)] = 0
    return np.linalg.norm(error_matrix, 'fro') / np.count_nonzero(omega)


def _split_omega(omega, T):
    omegas = [np.zeros(omega.shape) for t in range(2 * T + 1)]
    row, col = omega.shape
    for i in range(row):
        for j in range(col):
            idx = random.randint(0, 2 * T)
            omegas[idx][i, j] = omega[i, j]
    return omegas


def _init_U(M, omega, p, k, mu):
    M[(1 - omega).astype(np.int16)] = 0
    M = M / p
    U, S, V = np.linalg.svd(M, full_matrices=False)

    U_hat = U.copy()
    clip_threshold = 2 * mu * math.sqrt(k / max(M.shape))
    U_hat[U_hat > clip_threshold] = 0
    U_hat = orth(U_hat)
    print("|U_hat-U|_F/|U|_F:",
          np.linalg.norm(np.subtract(U_hat, U), ord='fro') / np.linalg.norm(
              U, ord='fro'))
    return U_hat


def _solve(M, omega, p, k, T, mu):
    omegas = _split_omega(omega, T)
    U = _init_U(M[:, :], omegas[0], p, k, mu)
    print('')
    V = None
    for t in range(T):
        V = _get_V_from_U(M, U, omegas[t + 1])
        U = _get_V_from_U(M.T, V.T, omegas[T + t + 1].T).T

        err = _get_err(M, U, V, omega)
        print('>> t(%3d):' % t, err)
    print('')
    assert V is not None
    return np.dot(U, V)


def main(m, n, k, p, T, mu):
    M = _make_lr_matrix(m, n, k)
    omega = np.zeros((m, n))
    omega[np.random.rand(m, n) <= p] = 1
    cardinality_of_omega = np.count_nonzero(omega)
    omega = omega.astype(np.int16)
    M_rank = np.linalg.matrix_rank(M)
    print("RANK of M        :", M_rank)
    M_ = _get_masked_matrix(M, omega)

    X = _solve(M, omega, p, k, T, mu)
    X_rank = np.linalg.matrix_rank(X)
    print("RANK of X        :", X_rank)

    E = np.subtract(M, X)
    E_train = E.copy()
    np.place(E_train, 1 - omega, 0)
    print('TRAIN RMSE       :',
          np.linalg.norm(E_train, "fro") / cardinality_of_omega)
    E_test = E.copy()
    np.place(E_test, omega, 0)
    print('TEST  RMSE       :',
          np.linalg.norm(E_test, "fro") / (m * n - cardinality_of_omega))

    print("|X-M|_F/|M|_F    :",
          np.linalg.norm(np.subtract(M, X), ord='fro') / np.linalg.norm(
              M, ord='fro'))


if __name__ == '__main__':
    # Given
    m = 100
    n = 100
    p = 0.1

    # Hyper Parameters
    k = 2
    T = 5
    mu = 0.1
    main(m, n, k, p, T, mu)
