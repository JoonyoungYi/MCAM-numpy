import os
import random

from ..func.line import convert_to_rating
from ..configs import DATA_RAW_DIR_PATH
from ..configs import DATA_PROCESSED_DIR_PATH


def _get_raw_dir_path(folder):
    return os.path.join(DATA_RAW_DIR_PATH, folder)


def _get_processed_dir_path(folder):
    return os.path.join(DATA_PROCESSED_DIR_PATH, folder)


def _get_user_dict_and_item_dict(folder, file_name):
    file_path = os.path.join(_get_raw_dir_path(folder), file_name)
    with open(file_path, 'r') as f:
        user_dict, item_dict = {}, {}
        for line in f:
            r = convert_to_rating(line, token='\t')
            user_dict[r.user_id] = 1
            item_dict[r.item_id] = 1
        return user_dict, item_dict
    return None, None


def _get_and_save_index_dict(d, dir_path, file_name):
    with open(os.path.join(dir_path, '{}.dat'.format(file_name)), 'w') as f:
        index_dict = {}
        for index, id in enumerate(d.keys()):
            index_dict[id] = index
            f.write('{}\n'.format(id))
        return index_dict


def get_and_save_index_dicts(folder, file_name):
    assert folder

    user_dict, item_dict = _get_user_dict_and_item_dict(folder, file_name)
    if user_dict is None or item_dict is None:
        raise Exception('file_path로부터 user_dict와 item_dict를 얻어오지 못했습니다.')

    dir_path = _get_processed_dir_path(folder)
    user_row_dict = _get_and_save_index_dict(user_dict, dir_path, 'user_ids')
    if user_row_dict is None:
        raise Exception('user_dict로부터 index_dict를 얻어오지 못했습니다.')

    item_col_dict = _get_and_save_index_dict(item_dict, dir_path, 'item_ids')
    if item_col_dict is None:
        raise Exception('item_dict로부터 index_dict를 얻어오지 못했습니다.')

    return user_row_dict, item_col_dict


def _save_test_sets(dir_path, file_name, user_row_dict, item_col_dict):
    fs = [
        open(os.path.join(dir_path, 'test_{}.dat'.format(c)), 'w')
        for c in range(5)
    ]
    f = open('app/data/raw/ml-100k/u.data', 'r')
    for line in f:
        r = convert_to_rating(line, token='\t')
        c = random.randint(0, 4)  # random class uniformly
        fs[c].write('{},{},{}\n'.format(user_row_dict[r.user_id],
                                        item_col_dict[r.item_id], r.score))
    f.close()
    (f.close() for f in fs)


def _save_training_sets(dir_path):
    for i in range(5):
        f = open(os.path.join(dir_path, 'training_{}.dat'.format(i)), 'w')
        for c in range(5):
            if i == c:
                continue
            _f = open(os.path.join(dir_path, 'test_{}.dat'.format(c)), 'r')
            for line in _f:
                f.write(line)
            _f.close()
        f.close()


def save_5_folded_data(folder, file_name, user_row_dict, item_col_dict):
    dir_path = _get_processed_dir_path(folder)
    _save_test_sets(dir_path, file_name, user_row_dict, item_col_dict)
    _save_training_sets(dir_path)
