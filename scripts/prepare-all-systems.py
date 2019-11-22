#!/usr/bin/env python3
from pathlib import Path

import click

from dlqmc.utils import NestedDict
from prepare_runs import prepare_run

systems = [
    'H2',
    'B',
    'Be',
    'LiH',
    *({'name': 'Hn', 'n': 10, 'dist': d} for d in [1.2, 1.8, 3.6]),
]
cass = {'H2': [2, 2], 'B': [4, 3], 'LiH': [4, 2], 'Hn': [6, 4], 'Be': [4, 2]}
param_sets = ['SD-SJ', 'SD-SJBF', 'MD-SJ', 'MD-SJBF']


@click.command()
@click.argument('basedir')
def main(basedir):
    for system in systems:
        sys_name = system if isinstance(system, str) else system['name']
        sys_label = sys_name
        if sys_name == 'Hn':
            sys_label += f'-{system["dist"]}'
        for param_set in param_sets:
            path = Path(f'{basedir}/{sys_label}/{param_set}')
            if path.exists():
                continue
            params = NestedDict()
            params['system'] = system
            params['train_kwargs.optimizer'] = 'Adam'
            if 'MD' in param_set:
                params['model_kwargs.cas'] = cass[sys_name]
            if 'BF' not in param_set:
                params['model_kwargs.omni_kwargs.with_backflow'] = False
            if sys_name == 'Hn':
                params['train_kwargs.fit_kwargs.subbatch_size'] = 2_000
            prepare_run(path, params)


if __name__ == '__main__':
    main()
