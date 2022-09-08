class DeepQMCError(Exception):
    pass


class NanError(DeepQMCError):
    def __init__(self):
        super().__init__()
