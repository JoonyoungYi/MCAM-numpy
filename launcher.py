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

f = open(filename, 'r')
f.readline()

max_user_id = 0
max_movie_id = 0

user_ids = {}
movie_ids = {}

for line in f:
    cols = line.split(',')

    user_id = int(cols[0])
    movie_id = int(cols[1])
    # rating = int(float(cols[2]) * 2)

    if user_id > max_user_id:
        max_user_id = user_id

    if movie_id > max_movie_id:
        max_movie_id = movie_id

    user_ids[user_id] = 1
    movie_ids[movie_id] = 1

print(max_user_id)
print(max_movie_id)
print(len(user_ids))
print(len(movie_ids))

# alternating_minimization = AlternatingMinimization()
# r, a = alternating_minimization.run(m)
# error = np.linalg.norm(np.subtract(m, r), ord='fro')
# print(error)
