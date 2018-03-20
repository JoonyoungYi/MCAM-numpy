from . import base_init


def init():
    return base_init('ml-20m', 'ratings.csv', ',', header_row=True)
