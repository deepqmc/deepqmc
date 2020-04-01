__all__ = ()


class InfoException(Exception):
    def __init__(self, info=None):
        self.info = info or {}
        super().__init__(self.info)


class DeepQMCError(Exception):
    pass


class NanLoss(DeepQMCError):
    pass


class NanGradients(DeepQMCError):
    pass


class TrainingBlowup(DeepQMCError):
    def __init__(self, step, chkpts):
        super().__init__()
        self.step = step
        self.chkpts = chkpts


class LUFactError(InfoException, DeepQMCError):
    pass
