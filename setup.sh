#!/bin/bash

# install poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/{$USER}/.local/bin:$PATH" 

# # https://stackoverflow.com/questions/59882884/vscode-doesnt-show-poetry-virtualenvs-in-select-interpreter-option
poetry config virtualenvs.in-project true

# Install project to setup requirements
poetry install --no-interaction

# poetry environment
poetry env list

# # update environment
# poetry update