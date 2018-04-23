import numpy as np


class AlternatingMinimization(object):
    """ this class is for performing low rank matrix completion

    Attributes:
      _MAX_ERROR (float): default error boundary

    """
    _MAX_ERROR = 1e10

    @classmethod
    def _proper_max(cls, row_x, col_x, rank, max_rank):
        """
        determine proper max value

        Args:
            row_x (int): number of rows of given matrix
            col_x (str): number of columns of given matrix
            rank (int): number of singular values
            max_rank (str): user-provided maximum limit

        Returns:
            int: proper max rank
        """
        return min(rank, max_rank, int(min(row_x, col_x) / 10))

    @classmethod
    def _get_r_from_l(cls, matrix_x, matrix_l):
        """
        compute matrix R from original matrix and left matrix.

        Args:
            matrix_x (numpy.array): given matrix
            matrix_l (numpy.array): left matrix

        Returns:
            numpy.array: right matrix
        """
        column = matrix_x.shape[1]
        rank = matrix_l.shape[1]
        R = np.empty((rank, column), dtype=np.float16)
        for j in range(0, column):
            matrix_l_new = matrix_l.copy()
            matrix_l_new[matrix_x[:, j] == 0, :] = 0

            R[:, j] = np.dot(np.linalg.pinv(matrix_l_new), matrix_x[:, j])
        return R

    @classmethod
    def _get_r_from_l2(cls, matrix_x, matrix_l):
        """
        compute matrix R from original matrix and left matrix.

        Args:
            matrix_x (numpy.array): given matrix
            matrix_l (numpy.array): left matrix

        Returns:
            numpy.array: right matrix
        """
        column = matrix_x.shape[1]
        rank = matrix_l.shape[1]
        matrix_r = np.empty((rank, column))

        for j in range(0, column):
            matrix_l_new = matrix_l.copy()
            matrix_l_new[matrix_x[:, j] == 0, :] = 0

            q_class = np.dot(matrix_l_new.T, matrix_l_new)
            c_class = np.dot(matrix_x[:, j], matrix_l_new)
            q_det = np.linalg.det(q_class)
            if q_det < 1e-10:
                matrix_r[:, j] = np.dot(
                    np.linalg.pinv(q_class, 1e-10), c_class)
            else:
                matrix_r[:, j] = np.linalg.solve(q_class, c_class)
        return matrix_r

    @classmethod
    def _get_err(cls, matrix_x, matrix_y):
        """
        compute training error

        Args:
            matrix_x (numpy.array): original matrix
            matrix_y (numpy.array): derived matrix

        Returns:
            float: training error
        """
        error_matrix = matrix_x - matrix_y
        error_matrix[matrix_x == 0] = 0
        return np.linalg.norm(error_matrix, 'fro') / np.count_nonzero(matrix_x)

    @classmethod
    def _get_candidate(cls, matrix_x, matrix_l_rank_k, delta, max_iter):
        """
        get matrix_l of rank k and candidate matrix from given matrix_x and initial matrix_l of rank k

        Args:
            matrix_x (int): given matrix
            matrix_l_rank_k (str): matrix l that has k number of columns
            delta (int): training error gap for stop training if error is close enough
            max_iter (str): maximum number of iteration

        Returns:
            float: Description of return value
            numpy.array: best left side matrix that can express matrix_x
            numpy.array: best right side matrix that can express matrix_x
        """
        result = (cls._MAX_ERROR, None, None)

        for _ in range(0, max_iter):
            matrix_r_rank_kt = cls._get_r_from_l2(matrix_x, matrix_l_rank_k)
            matrix_l_rank_k = cls._get_r_from_l2(matrix_x.T,
                                                 matrix_r_rank_kt.T).T
            candidate = np.dot(matrix_l_rank_k, matrix_r_rank_kt)
            train_err = cls._get_err(matrix_x, candidate)

            if -delta < result[0] - train_err < delta:
                result = (train_err, matrix_l_rank_k, matrix_r_rank_kt)
                break
            if result[0] > train_err:
                result = (train_err, matrix_l_rank_k, matrix_r_rank_kt)
        return result

    @classmethod
    def run(cls, matrix_x, min_rank=1, max_rank=30, delta=1e-6,
            max_iter=10000):
        """
        perform low rank matrix completion

        Args:
            matrix_x (int): given matrix
            min_rank (str): minimum rank to try
            max_rank (int): maximum rank to try
            delta (str): training/test error gap for stop if error is close enough
            max_iter (int): maximum number of iterations per each rank

        Returns:
            numpy.array: best left side matrix that can express matrix_x
            numpy.array: best right side matrix that can express matrix_x
        """
        matrix_u, matrix_sigma, matrix_vt = np.linalg.svd(
            matrix_x, full_matrices=False)
        rank = cls._proper_max(matrix_x.shape[0], matrix_x.shape[1],
                               matrix_sigma.shape[0], max_rank)
        min_rank = min(min_rank, rank)
        result = (cls._MAX_ERROR, None, None)

        for k in range(min_rank, rank + 1):
            candidate_tuple = cls._get_candidate(
                matrix_x, matrix_u[:, :k].copy(), delta, max_iter)
            # self.log.info('RANK : %d \t(Training Error : %f)' % (k, err))
            if result[0] - candidate_tuple[0] > delta:
                result = candidate_tuple
            else:
                break

        return result[1], result[2]
