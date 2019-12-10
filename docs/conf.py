import datetime

import toml

with open('../pyproject.toml') as f:
    metadata = toml.load(f)['tool']['poetry']

project = 'DeepQMC'
version = metadata['version']
author = ' '.join(metadata['authors'][0].split()[:-1])
description = metadata['description']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}
source_suffix = '.rst'
master_doc = 'index'
copyright = f'2019-{datetime.date.today().year}, {author}'
release = version
language = None
exclude_patterns = ['build', '.DS_Store']
pygments_style = 'sphinx'
todo_include_todos = True
html_theme = 'alabaster'
html_theme_options = {
    'description': description,
    'github_button': True,
    'github_user': 'noegroup',
    'github_repo': 'deepqmc',
    'badge_branch': 'master',
    'codecov_button': True,
    'travis_button': True,
}
html_sidebars = {
    '**': ['about.html', 'navigation.html', 'relations.html', 'searchbox.html']
}
htmlhelp_basename = f'{project}doc'
autodoc_default_options = {'special-members': '__call__'}
