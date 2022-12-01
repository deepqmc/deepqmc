from setuptools import find_packages, setup

requires = {
    'install': {
        'dm-haiku',
        'e3nn-jax',
        'h5py',
        'hydra-core',
        'jax[cuda]',
        'kfac-jax',
        'pyscf',
        'tensorboard',
        'toml',
        'tqdm',
        'uncertainties',
    },
    'test': {'pytest', 'pytest-regressions'},
    'format': {'black', 'flake8', 'isort'},
}

setup(
    name='deepqmc',
    version='1.0.0',
    author=(
        'Jan Hermann <jan.hermann@fu-berlin.de>, '
        'Zeno Schätzle <zenoone@physik.fu-berlin.de>, '
        'Péter Bernát Szabó <peter.bernat.szabo@fu-berlin.de>'
    ),
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=requires['install'],
    scripts=['bin/deepqmc'],
    extras=['dev', 'test'],
    extras_require={
        'test': requires['test'],
        'dev': {*requires['format'], *requires['test']},
    },
)
