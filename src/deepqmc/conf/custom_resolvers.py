from omegaconf import Container
from omegaconf._impl import select_value

from ..parallel import get_process_count, get_process_index

__all__ = ()


def process_idx_suffix() -> str:
    process_count = get_process_count()
    process_idx = get_process_index()
    if process_count is None or process_idx is None or int(process_count) == 1:
        return ''
    return f'_{process_idx}'


def mode_subdir(_root_: Container) -> str:
    r"""Get the mode subdir for the current run."""
    evaluate = select_value(_root_, 'task.evaluate', default=False)
    return 'evaluation' if evaluate else 'training'
