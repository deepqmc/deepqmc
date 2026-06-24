## Style guides

### Commit messages

- Imperative style (has to work in "If this commit is applied, it will \<commit message\>.")
- The first line <80 characters, the second line empty

### Python code

- [Black](https://github.com/psf/black) style
- No extra blank lines
- Has to pass [ruff](https://docs.astral.sh/ruff/) (linting and import sorting)
- [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings
- Has to pass [pyright](https://microsoft.github.io/pyright/) type checking
- Has to pass [codespell](https://github.com/codespell-project/codespell)
