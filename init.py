import os

from app.func import log


def _dir_path_exist(dir_path):
    if os.path.exists(dir_path):
        return True
    log.e('{} 폴더가 존재하지 않습니다.'.format(dir_path))
    return False


def main():
    for dir_path in [
            'app/data',
            'app/data/raw',
            'app/data/raw/ml-100k',
            'app/data/raw/ml-10m',
            'app/data/raw/ml-1m',
            'app/data/raw/ml-20m',
    ]:
        if not _dir_path_exist(dir_path):
            return False

    
    return False


if __name__ == '__main__':
    main()
