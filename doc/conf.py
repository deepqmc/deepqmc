import datetime
import os
import subprocess
import sys

import toml

sys.path.insert(0, os.path.abspath('../src'))
with open('../pyproject.toml') as f:
    metadata = toml.load(f)['tool']['poetry']

project = 'DeepQMC'
author = ' '.join(metadata['authors'][0].split()[:-1])
release = version = (
    subprocess.run(['poetry', 'version'], capture_output=True, cwd='..')
    .stdout.decode()
    .split()[1]
)
description = metadata['description']
year_range = (2019, datetime.date.today().year)
year_str = (
    str(year_range[0])
    if year_range[0] == year_range[1]
    else f'{year_range[0]}-{year_range[1]}'
)
copyright = f'{year_str}, Frank Noé and collaborators'

master_doc = 'index'
extensions = [
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinxcontrib.katex',
]
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'pyscf': ('http://pyscf.org/pyscf', None),
}
exclude_patterns = ['build', '.DS_Store']

html_theme = 'alabaster'
html_theme_options = {
    'description': description,
    'github_button': True,
    'github_user': 'deepqmc',
    'github_repo': 'deepqmc',
    'badge_branch': 'master',
    'codecov_button': True,
    'fixed_sidebar': True,
    'page_width': '60em',
}
html_sidebars = {
    '**': ['about.html', 'navigation.html', 'relations.html', 'searchbox.html']
}
html_static_path = ['_static']

autodoc_default_options = {'members': True}
autodoc_inherit_docstrings = False
autodoc_mock_imports = [
    'h5py',
    'numpy',
    'pyscf',
    'scipy',
    'torch',
    'tqdm',
    'uncertainties',
]
todo_include_todos = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True
