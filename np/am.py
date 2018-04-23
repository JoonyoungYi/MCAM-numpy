import math

import numpy as np


class AlternatingMinimization(object):
    """
    this class is for performing low rank matrix completion
    """

    @classmethod
    def _get_V_from_U(cls, M, U, mask):
        """
        compute matrix V from original matrix M and left matrix U.

        Args:
            M (numpy.array): given matrix
            U (numpy.array): left matrix
            mask (numpy.array): mask of hidden entries

        Returns:
            numpy.array: right matrix
        """
        column = M.shape[1]
        rank = U.shape[1]
        V = np.empty((rank, column), dtype=type(M))

        for j in range(0, column):
            U_ = U.copy()
            U_[mask[:, j], :] = 0
            V[:, j] = np.linalg.lstsq(U_, M[:, j], rcond=None)[0]
        return V

    @classmethod
    def _get_err(cls, M, U, V, mask):
        """
        compute training error

        Args:
            M (numpy.array): original matrix
            U (numpy.array): derived left matrix
            V (numpy.array): derived right matrix
            mask (numpy.array): mask of hidden entries

        Returns:
            float: training error
        """
        error_matrix = M - np.dot(U, V)
        error_matrix[mask] = 0
        return np.linalg.norm(error_matrix, 'fro') / np.count_nonzero(mask)

    @classmethod
    def run(cls, M, mask, k=1, early_stop=1e-6, delta=1e-6, max_iter=10000):
        """
        perform low rank matrix completion

        Args:
            M (numpy.array): given matrix
            mask (numpy.array): mask of hidden entries
            rank (int): rank to try
            early_stop (float): early stop criteria for training error.
            max_iter (int): maximum number of iterations per each rank

        Returns:
            numpy.array: best left side matrix(U) that can express matrix M
            numpy.array: best right side matrix(V) that can express matrix M
        """
        train_err = None
        U, S, V = np.linalg.svd(M, full_matrices=False)
        U, V, train_err = U[:, :k], np.dot(S, V), None

        for _ in range(max_iter):
            V = cls._get_V_from_U(M, U, mask)
            U = cls._get_V_from_U(M.T, V.T, mask.T).T

            err = cls._get_err(M, U, V, mask)
            print('>>', err)
            if err < early_stop:
                break
            if train_err and (-delta < train_err - err < delta):
                break
            train_err = err
        return train_err, U, V
