import random

import numpy as np

from am import AlternatingMinimization as AM

m = 100
n = 100
k = 2
p = 0.5

L = np.random.randn(m, k)
R = np.random.randn(k, n)
M = np.dot(L, R)

M_max = np.max(M)
M_min = np.min(M)

M_ = M.copy()
mask = np.random.rand(m, n) >= p
M_[mask] = 0

am = AM()
train_err, U, V = am.run(M_, k=k, mask=mask)
X = np.dot(U, V)
error = np.linalg.norm(
    np.subtract(M, X), ord='fro') / np.linalg.norm(
        M, ord='fro')
print(error)
