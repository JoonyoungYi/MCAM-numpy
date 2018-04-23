import numpy as np

from am import AlternatingMinimization as AM

m = 100
n = 20
k = 2

L = np.random.randn(m, k)
R = np.random.randn(k, n)
M = np.dot(L, R)

am = AM()
U, V = am.run(M, min_rank=k, max_rank=k)
X = np.matmul(U, V)
error = np.linalg.norm(
    np.subtract(M, X), ord='fro') / np.linalg.norm(
        M, ord='fro')
print(error)
