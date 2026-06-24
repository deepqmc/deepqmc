class NanError(Exception):
    """Exception due to unexpected NaN."""

    def __init__(self):
        super().__init__()


class TrainingBlowup(Exception):
    """Exception due to sudden increase in energy."""

    def __init__(self):
        super().__init__()


class TrainingCrash(Exception):
    """Exception if training ends before completion of total training steps.

    Args:
        train_state (~deepqmc.types.TrainState): the last training state that was
            checkpointed before the crash.
    """

    def __init__(self, train_state):
        super().__init__()
        self.train_state = train_state
