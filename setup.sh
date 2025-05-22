#!/bin/bash

# install poetry
curl -sSL https://install.python-poetry.org | python3.10 -
export PATH="/home/{$USER}/.local/bin:$PATH"

# # https://stackoverflow.com/questions/59882884/vscode-doesnt-show-poetry-virtualenvs-in-select-interpreter-option
poetry config virtualenvs.in-project true

# Run with project python env when setting up projects
poetry env use $(pyenv which python)

# Install project to setup requirements
poetry install --no-interaction

# poetry environment
poetry env list

# # update environment
# poetry update
