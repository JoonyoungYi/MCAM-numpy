import random


def _get_row_from_line(line):
    line = line.strip()
    rows = line.split('\t')
    return [float(r) if j == 2 else int(r) for j, r in enumerate(rows)]


def init():
    f = open('app/data/raw/ml-100k/u.data', 'r')
    user_dict, item_dict = {}, {}
    for line in f:
        user_id, item_id, rating, timestamp = _get_row_from_line(line)

        rating_number = user_dict.get(user_id, 0)
        user_dict[user_id] = rating_number + 1
        rating_number = item_dict.get(item_id, 0)
        item_dict[item_id] = rating_number + 1
    f.close()

    f = open('app/data/processed/ml-100k/user_ids.dat', 'w')
    user_row_dict = {}
    for row_index, user_id in enumerate(user_dict.keys()):
        user_row_dict[user_id] = row_index
        f.write('{}\n'.format(user_id))
    f.close()

    f = open('app/data/processed/ml-100k/item_ids.dat', 'w')
    item_col_dict = {}
    for col_index, item_id in enumerate(item_dict.keys()):
        item_col_dict[item_id] = col_index
        f.write('{}\n'.format(item_id))
    f.close()

    # 5폴드 시행.
    fs = [None for i in range(5)]
    for i in range(5):
        fs[i] = open('app/data/processed/ml-100k/test_{}.dat'.format(i), 'w')
    f = open('app/data/raw/ml-100k/u.data', 'r')
    for line in f:
        user_id, item_id, rating, timestamp = _get_row_from_line(line)
        c = random.randint(0, 4)  # random class uniformly
        fs[c].write('{},{},{}\n'.format(user_id, item_id, rating))
    f.close()
    for i in range(5):
        fs[i].close()

    # 시행한 5 fold data로 training data 만듦.
    for i in range(5):
        f = open('app/data/processed/ml-100k/training_{}.dat'.format(i), 'w')
        for c in range(5):
            if i == c:
                continue
            _f = open('app/data/processed/ml-100k/test_{}.dat'.format(c), 'r')
            for line in _f:
                f.write(line)
            _f.close()
        f.close()
