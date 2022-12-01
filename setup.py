from setuptools import find_packages, setup

setup(
    name='deepqmc',
    version='1.0.0',
    author=(
        'Jan Hermann <jan.hermann@fu-berlin.de>'
        'Zeno Schätzle <zenoone@physik.fu-berlin.de>'
        'Péter Bernát Szabó <peter.bernat.szabo@fu-berlin.de>'
    ),
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
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
    ],
    scripts=['bin/deepqmc'],
    extras=['dev'],
    extras_require={
        'dev': ['black', 'flake8', 'isort', 'pytest', 'pytest-regressions']
    },
)
