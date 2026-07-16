import datetime
import os
import sys
import types as _types
import toml


# Provide a minimal jax_dataclasses stub so that @jdc.pytree_dataclass
# does not replace classes with Mock objects during the Sphinx build.
# Without this, PhysicalConfiguration (and similar classes) would be
# replaced by MagicMock(name='pytree_dataclass()'), causing Sphinx to
# render their type annotations as "jax_dataclasses.pytree_dataclass".
_jdc_stub = _types.ModuleType('jax_dataclasses')
_jdc_stub.pytree_dataclass = lambda cls: cls  # identity decorator
_jdc_stub.replace = lambda obj, **changes: obj  # no-op; never called during docs build
sys.modules['jax_dataclasses'] = _jdc_stub
sys.path.insert(0, os.path.abspath('../src'))
with open('../pyproject.toml') as f:
    metadata = toml.load(f)['project']
project = 'DeepQMC'
author = ''  # ' '.join(metadata['authors'][0].split()[:-1])
release = version = metadata['version']
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
    'nbsphinx',
]
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jax': ('https://docs.jax.dev/en/latest', None),
    'haiku': ('https://dm-haiku.readthedocs.io/en/latest', None),
    'pyscf': ('https://pyscf.org', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}
exclude_patterns = ['build', '.DS_Store', '**.ipynb_checkpoints']

# Example notebooks are executed once by their authors and checked in with their
# outputs already saved (actually running a DeepQMC training during the docs build
# is not feasible); the build must therefore always use the stored outputs.
nbsphinx_execute = 'never'
# Force Python syntax highlighting for code cells regardless of whether a
# notebook carries its own kernelspec/language_info metadata (cleaned-up
# example notebooks intentionally omit it).
nbsphinx_codecell_lexer = 'ipython3'
# `!shell command` cells are valid ipython3 syntax and render correctly, but an
# auxiliary Sphinx pass (unrelated to nbsphinx's own rendering) also tries to
# lex them as plain Python for indexing purposes and warns when that fails,
# even though it then falls back gracefully. Harmless; suppressed so it
# doesn't fail `sphinx-build -W`.
suppress_warnings = ['misc.highlighting_failure']
autosectionlabel_prefix_document = True
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'show_toc_level': 1,
    'secondary_sidebar_items': [],
    'footer_start': ['copyright'],
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/deepqmc/deepqmc',  # required
            'icon': 'fa-brands fa-github',
            'type': 'fontawesome',
        }
    ],
    'navigation_with_keys': False,
    'navbar_persistent': ['search-button'],
    'header_links_before_dropdown': 6,
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
    'hydra',
    'numpy',
    'pyscf',
    'scipy',
    'tqdm',
    'uncertainties',
    'jax',
    'kfac_jax',
    'haiku',
    'omegaconf',
    'optax',
    'yaml',
    'tensorboardX',
    'folx',
]
toc_object_entries = False
todo_include_todos = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented_params'
