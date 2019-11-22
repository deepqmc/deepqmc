#!/usr/bin/env python3
import shutil
from datetime import datetime
from pathlib import Path

import click
import importlib_metadata
import toml

from dlqmc.train import merge_into

INIT_FILE = 'init.sh'
PARAM_FILE = 'params.toml'
PACKAGE_NAME = 'dlqmc'
PACKAGE_FILE = f'{PACKAGE_NAME}.tar.gz'
PACKAGE_FILE_VER = f'{PACKAGE_NAME}-{importlib_metadata.version(PACKAGE_NAME)}.tar.gz'
UTIL_DIR = Path(__file__).resolve().parent
ROOT = UTIL_DIR.parent


def prepare_run(path, params):
    path = Path(path)
    path.mkdir(parents=True)
    shutil.copy(ROOT / 'dist' / PACKAGE_FILE_VER, path / PACKAGE_FILE)
    (path / PARAM_FILE).write_text(toml.dumps(params))
    shutil.copy(UTIL_DIR / 'init-run.sh', path / INIT_FILE)


class TOMLParam(click.ParamType):
    name = 'toml'

    def convert(self, value, param, ctx):
        try:
            return toml.loads(value)
        except toml.decoder.TomlDecodeError:
            self.fail(f'{value!r} is not a valid TOML expression', param, ctx)


@click.command()
@click.option('--basedir', default='runs')
@click.option('--label')
@click.option('--conf', type=click.File())
@click.option('options', '-o', '--option', multiple=True, type=TOMLParam())
def prepare(basedir, label, options, conf):
    basedir = Path(basedir)
    label = label or datetime.now().isoformat(timespec='seconds')
    params = toml.load(conf) if conf else {}
    for option in options:
        merge_into(params, option)
    prepare_run(basedir / label, params)


if __name__ == '__main__':
    prepare()
