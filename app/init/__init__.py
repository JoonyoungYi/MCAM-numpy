import os
import random

from ..configs import DATA_RAW_DIR_PATH
from ..configs import DATA_PROCESSED_DIR_PATH
from ..models import Rating


def _convert_to_rating(line, token=','):
    rows = line.strip().split(token)
    user_id = int(rows[0])
    item_id = int(rows[1])
    score = float(rows[2])
    return Rating(score=score, user_id=user_id, item_id=item_id)


def _get_raw_dir_path(folder):
    return os.path.join(DATA_RAW_DIR_PATH, folder)


def _get_processed_dir_path(folder):
    return os.path.join(DATA_PROCESSED_DIR_PATH, folder)


def _get_user_dict_and_item_dict(folder, file_name, token, header_row):
    file_path = os.path.join(_get_raw_dir_path(folder), file_name)
    with open(file_path, 'r') as f:
        if header_row:
            f.readline()

        user_dict, item_dict = {}, {}
        for line in f:
            r = _convert_to_rating(line, token=token)
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


def _get_and_save_index_dicts(folder, file_name, token, header_row):
    assert folder

    user_dict, item_dict = _get_user_dict_and_item_dict(
        folder, file_name, token, header_row)
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


def _save_test_sets(
        folder,
        file_name,
        token,
        header_row,
        user_row_dict,
        item_col_dict, ):
    dir_path = _get_processed_dir_path(folder)
    fs = [
        open(os.path.join(dir_path, 'test_{}.dat'.format(c)), 'w')
        for c in range(5)
    ]
    f = open(os.path.join(_get_raw_dir_path(folder), file_name), 'r')
    if header_row:
        f.readline()

    for line in f:
        r = _convert_to_rating(line, token=token)
        c = random.randint(0, 4)  # random class uniformly
        fs[c].write('{},{},{}\n'.format(user_row_dict[r.user_id],
                                        item_col_dict[r.item_id], r.score))
    f.close()
    (f.close() for f in fs)


def _save_training_sets(folder):
    dir_path = _get_processed_dir_path(folder)
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


def _save_5_folded_data(
        folder,
        file_name,
        token,
        header_row,
        user_row_dict,
        item_col_dict, ):
    _save_test_sets(
        folder,
        file_name,
        token,
        header_row,
        user_row_dict,
        item_col_dict, )
    _save_training_sets(folder)


def base_init(folder, file_name, token, header_row=False):
    user_row_dict, item_col_dict = _get_and_save_index_dicts(
        folder,
        file_name,
        token,
        header_row, )
    _save_5_folded_data(
        folder,
        file_name,
        token,
        header_row,
        user_row_dict,
        item_col_dict, )
