## Style guides

### Commit messages

- Imperative style (has to work in "If this commit is applied, it will \<commit message\>.")
- The first line <80 characters, the second line empty

### Python code

- [Black](https://github.com/psf/black) style
- No extra blank lines
- Has to pass [flake8](https://gitlab.com/pycqa/flake8)
- Imports sorted with [isort](https://github.com/timothycrosley/isort)
- [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings
- Has to pass [mypy](https://github.com/python/mypy)
- Has to pass [codespell](https://github.com/codespell-project/codespell)
