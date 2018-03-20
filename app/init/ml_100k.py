from . import get_and_save_index_dicts
from . import save_5_folded_data


def init():
    FOLDER, FILE_NAME = 'ml-100k', 'u.data'
    user_row_dict, item_col_dict = get_and_save_index_dicts(FOLDER, FILE_NAME)
    save_5_folded_data(FOLDER, FILE_NAME, user_row_dict, item_col_dict)
