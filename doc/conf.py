import datetime
import os
import sys

import toml

sys.path.insert(0, os.path.abspath('../src'))
with open('../pyproject.toml') as f:
    metadata = toml.load(f)['project']
project = 'DeepQMC'
author = ''  # ' '.join(metadata['authors'][0].split()[:-1])
release = version = '1.0.0'
description = ''  # metadata['description']
year_range = (2019, datetime.date.today().year)
year_str = (
    str(year_range[0])
    if year_range[0] == year_range[1]
    else f'{year_range[0]}-{year_range[1]}'
)
copyright = f'{year_str}, Frank Noé and collaborators'
extensions = [
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinxcontrib.katex',
    'sphinx.ext.autosectionlabel',
]
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jax': ('https://jax.readthedocs.io/en/latest', None),
    'haiku': ('https://dm-haiku.readthedocs.io/en/latest', None),
    'pyscf': ('http://pyscf.org', None),
}
exclude_patterns = ['build', '.DS_Store']
autosectionlabel_prefix_document = True
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'show_toc_level': 1,
    'secondary_sidebar_items': [],
    'footer_start': ['copyright', 'sphinx-version', 'theme-version', 'sourcelink'],
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/deepqmc/deepqmc',  # required
            'icon': 'fa-brands fa-github',
            'type': 'fontawesome',
        }
    ],
}
html_sidebars = {
    '**': [
        'page-toc',
    ]
}
html_static_path = ['_static']

autodoc_default_options = {'members': True}
autodoc_inherit_docstrings = False
autodoc_mock_imports = [
    'h5py',
    'numpy',
    'pyscf',
    'scipy',
    'tqdm',
    'uncertainties',
    'jax',
    'jax_dataclasses',
    'kfac_jax',
    'haiku',
    'optax',
    'e3nn_jax',
    'yaml',
    'tensorboard',
]
toc_object_entries = False
todo_include_todos = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True
