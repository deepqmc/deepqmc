from .types import Batch


def batch_size_extractor(batch: Batch) -> int:
    r"""Compute the batch size for KFAC.

    KFAC requires a single batch dimension, we therefore flatten our batches resulting
    in batch dimensions that are a product of our various (molecule, electron)
    batch sizes. Note that each parameter receives gradients only from its samples,
    therefore the electronic state dimension is not included in this product.

    Args:
        batch (~deepqmc.types.Batch): a tuple containing a physical configuration,
            a set of sample weights and auxiliary data.

    Returns:
        int: the product of the molecule and electron batch size dimensions.
    """
    _, weights, _ = batch
    # product of the molecule and electron batch dims
    return weights.shape[0] * weights.shape[2]
