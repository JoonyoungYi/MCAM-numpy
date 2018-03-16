import numpy as np
from app.matrix_completion.alternating_minimization import AlternatingMinimization

# l = np.dot(5, np.random.rand(100, 2))
# r = np.random.rand(2, 20)
# m = np.dot(l, r)
#
# alternating_minimization = AlternatingMinimization()
# r, a = alternating_minimization.run(m)
# error = np.linalg.norm(np.subtract(m, r), ord='fro')
# print(error)

filename = 'files/ml-20m/ratings.csv'

MAX_USER_ID = 138494
MAX_MOVIE_ID = 131263

f = open(filename, 'r')
f.readline()

m = np.zeros((MAX_USER_ID, MAX_MOVIE_ID))
for line in f:
    cols = line.split(',')

    user_id = int(cols[0])
    movie_id = int(cols[1])
    rating = int(float(cols[2]) * 2)

    m[user_id, movie_id] = rating

print(m.shape)
m = m[~np.all(m == 0, axis=1)]
print(m.shape)
m = m[~np.all(m == 0, axis=2)]
print(m.shape)

# alternating_minimization = AlternatingMinimization()
# r, a = alternating_minimization.run(m)
# error = np.linalg.norm(np.subtract(m, r), ord='fro')
# print(error)
