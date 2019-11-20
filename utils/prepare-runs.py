#!/usr/bin/env python3
import shutil
from datetime import datetime
from pathlib import Path

import click
import toml

from dlqmc.train import get_default_params

INIT_FILE = 'init.sh'
PARAM_FILE = 'params.toml'
PACKAGE_FILE = 'dlqmc.tar.gz'
UTIL_DIR = Path(__file__).resolve().parent
ROOT = UTIL_DIR.parent


class TOMLParam(click.ParamType):
    name = 'toml'

    def convert(self, value, param, ctx):
        try:
            return toml.loads(value)
        except toml.decoder.TomlDecodeError:
            self.fail(f'{value!r} is not a valid TOML expression', param, ctx)


def merge_into(left, right):
    for key, val in right.items():
        if isinstance(val, dict):
            assert isinstance(left[key], dict)
            merge_into(left[key], val)
        else:
            if left.get(key) != val:
                print(f'Updating {key!r} from {left.get(key)!r} to {val!r}')
                left[key] = val


@click.command()
@click.option('--basedir', default='runs')
@click.option('--label')
@click.option('--conf', type=click.File())
@click.option('options', '-o', '--option', multiple=True, type=TOMLParam())
def prepare(basedir, label, options, conf):
    basedir = Path(basedir)
    label = label or datetime.now().isoformat(timespec='seconds')
    params = get_default_params()
    if conf:
        merge_into(params, toml.load(conf))
    for option in options:
        merge_into(params, option)
    path = basedir / label
    metadata = toml.loads((ROOT / 'pyproject.toml').read_text())['tool']['poetry']
    pacakge_file = f'{metadata["name"]}-{metadata["version"]}.tar.gz'
    path.mkdir(parents=True)
    shutil.copy(ROOT / 'dist' / pacakge_file, path / PACKAGE_FILE)
    (path / PARAM_FILE).write_text(toml.dumps(params))
    shutil.copy(UTIL_DIR / 'init-run.sh', path / INIT_FILE)


if __name__ == '__main__':
    prepare()
