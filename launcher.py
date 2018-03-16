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

uid2rid_dict, mid2cid_dict = {}, {}
rid, cid = 0, 0
for line in f:
    cols = line.split(',')
    uid = int(cols[0])
    mid = int(cols[1])
    r = int(float(cols[2]) * 2)

    r_index = uid2rid_dict.get(uid, None)
    if r_index is None:
        uid2rid_dict[uid] = rid
        rid += 1

    c_index = mid2cid_dict.get(mid, None)
    if c_index is None:
        mid2cid_dict[mid] = cid
        cid += 1

print(rid, cid)
truth = np.zeros((rid, cid))  # answer

f = open(filename, 'r')
f.readline()

for line in f:
    cols = line.split(',')
    uid = int(cols[0])  # user id
    mid = int(cols[1])  # movie id
    score = int(float(cols[2]) * 2)  # score

    rid = uid2rid_dict[uid]  # row id
    cid = mid2cid_dict[mid]  # column id

    truth[rid, cid] = score

mask = np.random.randint(1, 10, truth.shape)
train = np.copy(truth)  # training matrix
train[mask == 1] = 0
test = np.copy(truth)  # test matrix
test[mask != 1] = 0

alternating_minimization = AlternatingMinimization()
answer, left = alternating_minimization.run(train)
error = np.linalg.norm(np.subtract(answer, test)[mask == 1], ord='fro')
print(error)
count = count_nonzero(test)
print(count)
print(error / count)
