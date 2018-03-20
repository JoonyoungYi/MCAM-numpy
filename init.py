import os
import itertools

from app.func import log


def _dir_path_exist(dir_path):
    if os.path.exists(dir_path):
        return True
    log.e('{} 폴더가 존재하지 않습니다.'.format(dir_path))
    return False


def _create_dir_path_if_not_exist(dir_path):
    if os.path.exists(dir_path):
        return False
    os.makedirs(dir_path)
    return True


def main():
    for dir in ['', 'raw', 'processed']:
        _create_dir_path_if_not_exist('app/data/{}'.format(dir))

    folders = ['ml-100k', 'ml-10m', 'ml-1m', 'ml-20m']
    for dir, folder in itertools.product(['raw', 'processed'], folders):
        _create_dir_path_if_not_exist('app/data/{}/{}'.format(dir, folder))

    # if not _dir_path_exist('app/data/processed'):
    #     os.makedirs(directory)

    return False


if __name__ == '__main__':
    main()
