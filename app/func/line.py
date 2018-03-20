from ..models import Rating


def convert_to_rating(line, token=','):
    rows = line.strip().split(token)
    user_id = int(rows[0])
    item_id = int(rows[1])
    score = float(rows[2])
    return Rating(user_id=user_id, item_id=item_id, score=score)
