import os
from ..configs.local import DATA_PROCESSED_DIR_PATH as DIR_PATH
from ..func.line import convert_rating


def test(algorithm, data_kind):
    data_index = 0
    model = algorithm.train(data_kind, data_index)

    f = open(os.path.join(DIR_PATH, 'test_{}.data'.format(data_index)))
    for line in f:
        r = convert_rating(line)
        score = model.predict(r.user_id, r.item_id)
        print(r - r.score)
