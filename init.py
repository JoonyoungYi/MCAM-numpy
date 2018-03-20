import os
import itertools

from app.func import log
from app.init import ml_100k


def _create_dir_path_if_not_exist(dir_path):
    if os.path.exists(dir_path):
        return False
    os.makedirs(dir_path)
    return True


def _init_ml_100k():
    pass


def main():
    # folder initialization
    for dir in ['', 'raw', 'processed']:
        _create_dir_path_if_not_exist('app/data/{}'.format(dir))
    folders = ['ml-100k', 'ml-10m', 'ml-1m', 'ml-20m']
    for dir, folder in itertools.product(['raw', 'processed'], folders):
        _create_dir_path_if_not_exist('app/data/{}/{}'.format(dir, folder))

    ml_100k.init()
    return False


if __name__ == '__main__':
    main()
