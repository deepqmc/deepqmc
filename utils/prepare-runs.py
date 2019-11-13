#!/usr/bin/env python
import shutil
from datetime import datetime
from pathlib import Path

import click
import toml

from dlqmc.train import get_default_params

INIT_FILE = 'init.sh'
PARAM_FILE = 'params.toml'
ROOT = Path(__file__).resolve().parents[1]


class TOMLParam(click.ParamType):
    name = 'toml'

    def convert(self, value, param, ctx):
        try:
            return toml.loads(value)
        except toml.decoder.TomlDecodeError:
            self.fail(f'{value!r} is not a valid TOML expression', param, ctx)


def merge_into_defaults(options):
    params = get_default_params()
    for option in options:
        target = params
        while True:
            key, option = next(iter(option.items()))
            if not isinstance(option, dict):
                break
            target = target[key]
        print(f'Updating {key!r} from {target[key]!r} to {option!r}')
        target[key] = option
    return params


@click.command()
@click.option('--basedir', default='runs')
@click.option('--label')
@click.option('options', '-o', '--option', multiple=True, type=TOMLParam())
def prepare(basedir, label, options):
    basedir = Path(basedir)
    label = label or datetime.now().isoformat(timespec='seconds')
    params = merge_into_defaults(options)
    path = basedir / label
    metadata = toml.loads((ROOT / 'pyproject.toml').read_text())['tool']['poetry']
    pacakge_file = f'{metadata["name"]}-{metadata["version"]}.tar.gz'
    path.mkdir(parents=True)
    shutil.copy(ROOT / 'dist' / pacakge_file, path)
    (path / PARAM_FILE).write_text(toml.dumps(params))
    shutil.copy(ROOT / 'scripts/init-run.sh', path / INIT_FILE)


if __name__ == '__main__':
    prepare()
