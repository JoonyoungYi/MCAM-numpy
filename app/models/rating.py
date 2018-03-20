class Rating():
    def __init__(self,
                 score,
                 user_id=None,
                 item_id=None):
        self.score = score
        self.user_id = user_id
        self.item_id = item_id
