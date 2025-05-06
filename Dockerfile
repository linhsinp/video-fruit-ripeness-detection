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
# COPY ./app ./

# # explicitly assign port
# EXPOSE 8081

# run the web service on container startup
# here we use the gunicorn webserver
# with one worker process and 8 threads
# for environments with multiple CPU cores
# increase the number of workers
# timeout is set to 0 to disable the timeouts of the workers
# to allow Cloud Run to handle instance scaling.
# CMD exec gunicorn --bind :8081 --workers 1 --threads 8 --timeout 0 main:app
# CMD [ "gunicorn", "--bind=:8081", "--workers=1", "--threads=8", "--timeout=0", "main:app"]

# docker run --rm --volume=/Users/hsin-pei/Desktop/HarvestAi_GitHub/sandbox/yolov8-live/app/:/app -i streaming
ENTRYPOINT ["python", "main.py"]