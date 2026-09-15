#!/bin/sh
# Run the test suite inside the app image, next to the database container.
# Uses a separate database (futures_test); the real data is not touched and
# the language model is never called.
#
#   scripts/test.sh            # all tests
#   scripts/test.sh -k admin   # extra arguments go to pytest
set -eu
cd "$(dirname "$0")/.."

docker compose up -d db >/dev/null
docker compose build app >/dev/null
docker compose run --rm --no-deps -T \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$PWD":/src -w /src \
    --entrypoint sh app -c '
        pip install --quiet --user pytest==8.4.2 &&
        pybabel compile -d translations >/dev/null &&
        python -m pytest -p no:cacheprovider -q "$@"' sh "$@"
