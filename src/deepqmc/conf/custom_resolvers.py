import os

from omegaconf import Container
from omegaconf._impl import select_value

__all__ = ()


def get_hydra_subdir(_root_: Container) -> str:
    evaluate = select_value(_root_, 'task.evaluate', default=False)
    subdir = 'evaluation' if evaluate else 'training'
    return os.path.join(subdir, '.hydra')
