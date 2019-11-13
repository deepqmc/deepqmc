# DL-QMC

## Installing

```
poetry install -E test -E train
```

## Example

```python
from dlqmc.train import get_default_params, model, train

params = get_default_params()
params.model_kwargs.geomname = 'B'
params.model_kwargs.spin = 1
net, mf = model(**params.model_kwargs)
train(net, mf, cwd='runs/test', **params.train_kwargs)
```
