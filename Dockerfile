# syntax=docker/dockerfile:1
# Keep this syntax directive! It's used to enable Docker BuildKit
# reference https://github.com/python-poetry/poetry/discussions/1879#discussioncomment-216865

################################
# PYTHON-BASE
# Sets up all our shared environment variables
################################
FROM python:3.9-slim AS python-base
# https://dev.to/oben/apple-silicon-mac-m1m2-how-to-deal-with-slow-docker-performance-58n0
# FROM arm64v8/python:3.9-slim AS python-base

# use the official lightweight python image
# https://hub.docker.com/_/python

# python environment variables
# https://docs.python.org/3/using/cmdline.html#environment-variables
# pip options
# https://pip.pypa.io/en/stable/cli/pip/#general-options
# poetry environmental variables
# https://python-poetry.org/docs/configuration/#available-settings
# https://python-poetry.org/docs/configuration/#using-environment-variables
ENV \
    # allows statements and log messages to immediately appear in the logs
    PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv" 

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

################################
# BUILDER
# Used to build deps + create our virtual environment
################################
FROM python-base AS builder

ENV POETRY_VERSION=1.8.2

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION

RUN pip install "virtualenv>=20.23.0"

# copy project requirements here to ensure they are cached
WORKDIR $PYSETUP_PATH
COPY pyproject.toml ./

# install runtime deps - uses $POETRY_VIRTUALENVS_IN_PROJECT internally
# https://python-poetry.org/docs/cli/#install
RUN poetry install



################################
# PRODUCTION
# Final image used for runtime
################################
FROM python-base AS production

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    libgl1 libglib2.0-0 \
    # libegl1 libgomp1 libglib2.0-0
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder $PYSETUP_PATH $PYSETUP_PATH

ENTRYPOINT ["python", "main.py"]