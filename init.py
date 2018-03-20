import os
import itertools

from app.configs import DATA_DIR_PATH
from app.configs import DATA_RAW_DIR_PATH
from app.configs import DATA_PROCESSED_DIR_PATH
from app.func import log
from app.init import ml_100k
from app.init import ml_1m
from app.init import ml_10m
from app.init import ml_20m


def _create_dir_path_if_not_exist(dir_path):
    if os.path.exists(dir_path):
        return False
    os.makedirs(dir_path)
    return True


def main():
    # folder initialization
    for dir_path in [
            DATA_DIR_PATH,
            DATA_RAW_DIR_PATH,
            DATA_PROCESSED_DIR_PATH,
    ]:
        _create_dir_path_if_not_exist(dir_path)

    folders = ['ml-100k', 'ml-10m', 'ml-1m', 'ml-20m']
    for dir_path, folder in itertools.product([
            DATA_RAW_DIR_PATH,
            DATA_PROCESSED_DIR_PATH,
    ], folders):
        _create_dir_path_if_not_exist(os.path.join(dir_path, folder))

    # ml_100k.init()
    # ml_1m.init()
    # ml_10m.init()
    ml_20m.init()
    return True


if __name__ == '__main__':
    main()
